"""Workflow definition helpers for the Tapis Workflows ETL path."""

from __future__ import annotations

import base64
from typing import Iterable


WORKFLOW_TASK_CODE = r'''
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from owe_python_sdk.runtime import execution_context as ctx


CKAN_WEL_STANDARD_VAR = "groundwater_well__recharge_volume_flux"
CKAN_RCH_STANDARD_VAR = "groundwater__recharge_volume_flux"
CKAN_GRID_STANDARD_VAR = "Modflow-Spatially-Distributed-Grid"


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slugify(value):
    value = str(value or "").strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "dataset"


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def _extract_standard_vars(resource):
    candidates = []
    for key in ("MINT Standard Variables", "mint_standard_variables", "standard_variables"):
        value = resource.get(key)
        if value is not None:
            candidates.append(value)
    extras = resource.get("extras")
    if isinstance(extras, list):
        for item in extras:
            if isinstance(item, dict) and item.get("key") == "MINT Standard Variables":
                candidates.append(item.get("value"))
    values = []
    for entry in candidates:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            values.extend(str(v) for v in entry if v is not None)
            continue
        text = str(entry).strip()
        if not text:
            continue
        if text[:1] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                values.extend(str(v) for v in parsed if v is not None)
                continue
        values.extend(v.strip() for v in text.replace(";", ",").split(",") if v.strip())
    return values


def _resource_has_standard_var(resource, target):
    target_key = str(target).strip().lower()
    return any(str(value).strip().lower() == target_key for value in _extract_standard_vars(resource))


def _package_show(ckan_url, token, dataset_name):
    response = requests.get(
        f"{ckan_url.rstrip('/')}/api/3/action/package_show",
        params={"id": dataset_name},
        headers=_headers(token),
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {payload}")
    return payload["result"]


def _create_dataset(ckan_url, token, dataset_dict):
    response = requests.post(
        f"{ckan_url.rstrip('/')}/api/3/action/package_create",
        json=dataset_dict,
        headers=_headers(token),
        timeout=120,
    )
    if response.status_code == 409:
        raise RuntimeError("dataset name conflict")
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_create failed: {payload}")
    return payload["result"]


def _create_resource_upload(ckan_url, token, dataset_id, file_path, resource_dict):
    data = {
        "package_id": dataset_id,
        "name": resource_dict.get("name", file_path.name),
        "description": resource_dict.get("description", ""),
        "format": resource_dict.get("format", file_path.suffix.lstrip(".").upper() or "DAT"),
    }
    if resource_dict.get("mint_standard_variables"):
        data["mint_standard_variables"] = resource_dict["mint_standard_variables"]
    extras = resource_dict.get("extras")
    if extras:
        data["extras"] = json.dumps(extras)
    with file_path.open("rb") as handle:
        response = requests.post(
            f"{ckan_url.rstrip('/')}/api/3/action/resource_create",
            data=data,
            files={"upload": handle},
            headers=_headers(token),
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN resource_create failed: {payload}")
    return payload["result"]


def _extras_to_dict(extras):
    result = {}
    for item in extras or []:
        if isinstance(item, dict) and item.get("key"):
            result[str(item["key"])] = "" if item.get("value") is None else str(item.get("value"))
    return result


def _merge_extras(existing, additions):
    merged = _extras_to_dict(existing)
    merged.update({k: v for k, v in additions.items() if v is not None})
    return [{"key": key, "value": value} for key, value in merged.items()]


def _extract_mint_svo(extras):
    extras_dict = _extras_to_dict(extras)
    for key in ("MINT_SVO", "MINT Standard Variables", "mint_standard_variables", "standard_variables"):
        value = extras_dict.get(key)
        if value:
            return value
    return None


def _resolve_mint_svo(resource, dataset, target_var):
    mint_svo = _extract_mint_svo(resource.get("extras", []))
    if mint_svo:
        return mint_svo
    if _resource_has_standard_var(resource, target_var):
        return target_var
    mint_svo = _extract_mint_svo(dataset.get("extras", []))
    if mint_svo:
        return mint_svo
    for value in _extract_standard_vars(dataset):
        if str(value).strip().lower() == str(target_var).strip().lower():
            return target_var
    return None


def _build_dataset_payload(source_dataset, new_name, provenance, mint_svo, source_url=None, change_summary=None, maintainer_username=None, new_title=None):
    copy_fields = [
        "title",
        "notes",
        "owner_org",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "license_id",
        "private",
        "url",
        "version",
        "spatial",
        "temporal_coverage_start",
        "temporal_coverage_end",
        "groups",
    ]
    dataset = {field: source_dataset.get(field) for field in copy_fields if source_dataset.get(field) is not None}
    dataset["name"] = new_name
    dataset["title"] = new_title.strip() if new_title else f"{source_dataset.get('title', new_name)} (updated {_now_iso()})"
    dataset["tags"] = [{"name": t["name"]} for t in source_dataset.get("tags", []) if isinstance(t, dict) and t.get("name")]
    dataset["extras"] = _merge_extras(
        source_dataset.get("extras", []),
        {
            "MINT_SVO": mint_svo,
            "provenance_timestamp": provenance.get("timestamp", _now_iso()),
            "provenance_action": provenance.get("action", ""),
            "provenance_details": provenance.get("details", ""),
            "derived_from_dataset_id": provenance.get("source_dataset_id", ""),
            "derived_from_dataset_name": provenance.get("source_dataset_name", ""),
            "source_url": source_url or "",
            "change_summary": change_summary or "",
        },
    )
    if change_summary:
        dataset["notes"] = (dataset.get("notes") or "") + f"\n\nChanges: {change_summary}"
    if maintainer_username:
        dataset["maintainer"] = maintainer_username
    return dataset


def _build_resource_payload(source_resource, provenance, mint_svo, source_url=None, change_summary=None, default_name="WEL", default_format="WEL", resource_name=None):
    description = source_resource.get("description", "")
    if source_url:
        description = f"{description}\nMetadata This file comes from {source_url}".strip()
    if change_summary:
        description = f"{description}\nMetadata Description of Changes Made: {change_summary}".strip()
    return {
        "name": resource_name or f"{source_resource.get('name', default_name)} (updated)",
        "description": description,
        "format": source_resource.get("format", default_format),
        "mint_standard_variables": mint_svo,
        "extras": _merge_extras(
            source_resource.get("extras", []),
            {
                "MINT_SVO": mint_svo,
                "provenance_timestamp": provenance.get("timestamp", _now_iso()),
                "provenance_action": provenance.get("action", ""),
                "provenance_details": provenance.get("details", ""),
                "derived_from_resource_id": provenance.get("source_resource_id", ""),
                "derived_from_resource_name": provenance.get("source_resource_name", ""),
                "source_url": source_url or "",
                "change_summary": change_summary or "",
            },
        ),
    }


def _download_resource(resource, dest_dir):
    url = resource.get("url")
    if not url:
        raise ValueError("CKAN resource missing URL.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?", 1)[0]).name or _slugify(resource.get("name") or resource.get("id") or "resource")
    path = dest_dir / filename
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _extract_zip(zip_path):
    extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def _find_grid_data_path(root):
    if root.is_dir():
        for path in root.rglob("*"):
            if path.suffix.lower() in (".shp", ".geojson", ".gpkg", ".json", ".csv"):
                return path
        for path in root.rglob("*"):
            if path.suffix.lower() == ".gdb":
                return path
        return None
    return root


def _prepare_grid_gdf(gdf):
    if gdf.crs is None:
        centroids = gdf.geometry.centroid
        gdf["_lon"] = centroids.x
        gdf["_lat"] = centroids.y
        return gdf
    try:
        projected = gdf.to_crs(gdf.estimate_utm_crs()) if gdf.crs.is_geographic else gdf
    except Exception:
        projected = gdf
    centroids = projected.geometry.centroid
    try:
        centroids_ll = centroids.to_crs("EPSG:4326")
    except Exception:
        centroids_ll = centroids
    gdf["_lon"] = centroids_ll.x
    gdf["_lat"] = centroids_ll.y
    try:
        return gdf.to_crs("EPSG:4326")
    except Exception:
        return gdf


def _load_grid_csv(csv_path):
    df = pd.read_csv(csv_path)
    columns = {col.lower(): col for col in df.columns}
    lon_key = columns.get("centroidx") or columns.get("node_x")
    lat_key = columns.get("centroidy") or columns.get("node_y")
    if lon_key is None or lat_key is None:
        raise ValueError("CSV grid missing centroid/node coordinate columns.")
    df["_lon"] = pd.to_numeric(df[lon_key], errors="coerce")
    df["_lat"] = pd.to_numeric(df[lat_key], errors="coerce")
    lon_abs = df["_lon"].abs().max()
    lat_abs = df["_lat"].abs().max()
    if (lon_abs <= 90 and lat_abs > 90) or (lon_abs > 180 and lat_abs <= 180):
        df["_lon"], df["_lat"] = df["_lat"], df["_lon"]
    geometry = [None if pd.isna(lon) or pd.isna(lat) else Point(float(lon), float(lat)) for lon, lat in zip(df["_lon"], df["_lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry)


def _load_grid_resource(resource, dest_dir):
    path = _download_resource(resource, dest_dir)
    if path.suffix.lower() == ".zip":
        path = _extract_zip(path)
    grid_path = _find_grid_data_path(path)
    if grid_path is None:
        raise ValueError("No grid dataset found after extracting resource.")
    if grid_path.suffix.lower() == ".csv":
        return _load_grid_csv(grid_path)
    gdf = gpd.read_file(grid_path)
    return _prepare_grid_gdf(gdf)


class _SimpleStressPeriodData:
    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype


class MfusgWel:
    is_mfusg = True

    def __init__(self, header_line, spd, dtype):
        self.header_line = header_line.rstrip("\n")
        self.stress_period_data = _SimpleStressPeriodData(spd, dtype)

    def write_file(self, output_path):
        lines = [self.header_line]
        for per in sorted(self.stress_period_data.data.keys()):
            recs = self.stress_period_data.data.get(per)
            itmp = 0 if recs is None else len(recs)
            lines.append(f"{itmp} 0 0                  Stress Period {per + 1}")
            if itmp > 0:
                for rec in recs:
                    lines.append(f"{int(rec['node'])} {float(rec['flux']):.6g}")
        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class LazyMfusgWel(MfusgWel):
    is_lazy = True

    def __init__(self, path, header_line, index, dtype):
        super().__init__(header_line, {}, dtype)
        self.path = path
        self.index = index
        self.period_count = len(index)

    def get_period(self, period):
        if period < 0 or period >= self.period_count:
            return np.recarray(0, dtype=self.stress_period_data.dtype)
        entry = self.index[period]
        itmp = entry["itmp"]
        if itmp < 0:
            return self.get_period(period - 1) if period > 0 else np.recarray(0, dtype=self.stress_period_data.dtype)
        if itmp == 0:
            return np.recarray(0, dtype=self.stress_period_data.dtype)
        records = []
        with Path(self.path).open(encoding="utf-8") as handle:
            handle.seek(entry["offset"])
            for _ in range(itmp):
                line = _strip_comment(handle.readline())
                parts = line.split()
                if len(parts) >= 2:
                    records.append((int(float(parts[0])), float(parts[1])))
        return np.rec.array(records, dtype=self.stress_period_data.dtype) if records else np.recarray(0, dtype=self.stress_period_data.dtype)

    def load_all(self):
        return {per: self.get_period(per) for per in range(self.period_count)}


def _strip_comment(line):
    for token in ("#", ";"):
        if token in line:
            line = line.split(token, 1)[0]
    return line.strip()


def _detect_mfusg_wel(path):
    data_lines = [_strip_comment(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    data_lines = [line for line in data_lines if line]
    if len(data_lines) < 3:
        return False
    first_record = data_lines[2].split()
    if len(first_record) < 2:
        return False
    try:
        int(first_record[1])
    except ValueError:
        return True
    return False


def _index_mfusg_wel(wel_path):
    dtype = np.dtype([("node", "i4"), ("flux", "f8")])
    index = []
    with Path(wel_path).open(encoding="utf-8") as handle:
        header = _strip_comment(handle.readline()).rstrip("\n")
        while True:
            line = handle.readline()
            if not line:
                break
            line = _strip_comment(line)
            if not line:
                continue
            tokens = line.split()
            itmp = int(tokens[0])
            offset = handle.tell()
            if itmp > 0:
                for _ in range(itmp):
                    handle.readline()
            index.append({"itmp": itmp, "offset": offset})
    return LazyMfusgWel(wel_path, header, index, dtype)


def _scan_wel_metadata(path):
    data_lines = [_strip_comment(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    data_lines = [line for line in data_lines if line]
    if not data_lines:
        raise ValueError("WEL file is empty or has no data.")
    data_lines.pop(0)
    nper = 0
    max_k = max_i = max_j = 1
    idx = 0
    while idx < len(data_lines):
        tokens = data_lines[idx].split()
        idx += 1
        if not tokens:
            continue
        nper += 1
        itmp = int(tokens[0])
        if itmp <= 0:
            continue
        for _ in range(itmp):
            if idx >= len(data_lines):
                raise ValueError("Unexpected end of WEL file.")
            parts = data_lines[idx].split()
            idx += 1
            k, i, j = int(parts[0]), int(parts[1]), int(parts[2])
            max_k, max_i, max_j = max(max_k, k), max(max_i, i), max(max_j, j)
    return nper, max_k, max_i, max_j


def _load_wel(wel_path):
    wel_path = Path(wel_path)
    if _detect_mfusg_wel(wel_path):
        return _index_mfusg_wel(wel_path)
    nper, nlay, nrow, ncol = _scan_wel_metadata(wel_path)
    model = flopy.modflow.Modflow(modelname="wel_read", model_ws=str(wel_path.parent))
    flopy.modflow.ModflowDis(model, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper, delr=1.0, delc=1.0, top=1.0, botm=[0.0] * nlay)
    return flopy.modflow.ModflowWel.load(str(wel_path), model)


def _apply_rate_update(wel, gdf, selected_ids, new_rate, rate_mode, add_missing, layers_for_new, periods_for_update, output_path):
    spd = wel.stress_period_data.data
    base_dtype = getattr(wel.stress_period_data, "dtype", None)
    if getattr(wel, "is_mfusg", False):
        node_lookup = dict(zip(gdf["CELL_ID"], gdf["NODE_NUM"]))
        selected_cells = {int(node_lookup[cid]) for cid in selected_ids if cid in node_lookup}
        if hasattr(wel, "load_all"):
            spd = wel.load_all()
            wel.stress_period_data.data = spd
    else:
        cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
        selected_cells = {cell_lookup[cid] for cid in selected_ids if cid in cell_lookup}
    updated_spd = spd
    for per, recs in spd.items():
        if base_dtype is None and hasattr(recs, "dtype") and recs.dtype.names:
            base_dtype = recs.dtype
        if base_dtype is None:
            raise ValueError("WEL stress_period_data dtype unavailable.")
        if periods_for_update and per not in periods_for_update:
            continue
        recs = np.rec.array(recs, dtype=base_dtype)
        if getattr(wel, "is_mfusg", False):
            mask = np.array([int(rec["node"]) in selected_cells for rec in recs], dtype=bool)
        else:
            mask = np.array([(int(rec["i"]), int(rec["j"])) in selected_cells for rec in recs], dtype=bool)
        if rate_mode == "scale_percent":
            recs["flux"][mask] *= 1.0 + (float(new_rate) / 100.0)
        else:
            recs["flux"][mask] = float(new_rate)
        if add_missing and selected_cells:
            if getattr(wel, "is_mfusg", False):
                existing = {int(r["node"]) for r in recs}
                to_add = [node for node in selected_cells if node not in existing]
                if to_add:
                    new_recs = np.recarray(len(recs) + len(to_add), dtype=base_dtype)
                    new_recs[: len(recs)] = recs
                    idx = len(recs)
                    for node in to_add:
                        new_recs[idx]["node"] = int(node)
                        new_recs[idx]["flux"] = float(new_rate)
                        idx += 1
                    recs = new_recs
            else:
                existing = {(int(r["i"]), int(r["j"])) for r in recs}
                to_add = [cell for cell in selected_cells if cell not in existing]
                if to_add:
                    layers = [int(layer) for layer in layers_for_new] or [1]
                    new_recs = np.recarray(len(recs) + len(to_add) * len(layers), dtype=base_dtype)
                    new_recs[: len(recs)] = recs
                    idx = len(recs)
                    for row, col in to_add:
                        for layer in layers:
                            new_recs[idx]["k"] = int(layer)
                            new_recs[idx]["i"] = int(row)
                            new_recs[idx]["j"] = int(col)
                            new_recs[idx]["flux"] = float(new_rate)
                            idx += 1
                    recs = new_recs
        updated_spd[per] = recs
    if base_dtype is not None:
        for per, recs in updated_spd.items():
            updated_spd[per] = np.recarray(0, dtype=base_dtype) if recs is None or len(recs) == 0 else np.rec.array(recs, dtype=base_dtype)
    wel.write_file(output_path if getattr(wel, "is_mfusg", False) else str(output_path))
    return len(selected_cells)


def _load_rch(rch_path, nrow, ncol, nper=1):
    model = flopy.modflow.Modflow(modelname="rch_read", model_ws=str(Path(rch_path).parent))
    flopy.modflow.ModflowDis(model, nlay=1, nrow=nrow, ncol=ncol, nper=nper, delr=1.0, delc=1.0, top=1.0, botm=[0.0])
    try:
        return flopy.modflow.ModflowRch.load(str(rch_path), model)
    except Exception:
        text = Path(rch_path).read_text(encoding="utf-8")
        values = []
        for token in text.replace(",", " ").split():
            try:
                values.append(float(token))
            except ValueError:
                pass
        if len(values) < nrow * ncol:
            raise ValueError("Not enough numeric values to build an RCH array.")
        count = len(values)
        return np.array(values, dtype=float).reshape((count // (nrow * ncol), nrow, ncol)) if count % (nrow * ncol) == 0 else np.array(values[-nrow * ncol :], dtype=float).reshape((nrow, ncol))


def _rch_to_arrays(rch):
    data = rch.rech if hasattr(rch, "rech") else rch
    if hasattr(data, "array"):
        arr = np.array(data.array)
    elif isinstance(data, dict):
        arr = np.stack([np.array(v) for v in data.values()])
    else:
        arr = np.array(data)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[idx] for idx in range(arr.shape[0])]
    if arr.ndim == 1:
        return [arr]
    if arr.ndim > 3:
        return [arr.reshape(-1)]
    return [arr]


def _normalize_rch_arrays(arrays, nrow, ncol):
    normalized = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size == nrow * ncol:
                normalized.append(arr.reshape((nrow, ncol)))
                continue
            if arr.size % (nrow * ncol) == 0:
                reshaped = arr.reshape((arr.size // (nrow * ncol), nrow, ncol))
                normalized.extend([reshaped[idx] for idx in range(reshaped.shape[0])])
                continue
        if arr.ndim == 2:
            normalized.append(arr)
            continue
        if arr.ndim == 3:
            normalized.extend([arr[idx] for idx in range(arr.shape[0])])
    return normalized


def _choose_rch_indexing(arrays, gdf):
    if not arrays:
        return 0, False
    cell_pairs = list(zip(gdf["ROW"], gdf["COL"]))
    candidates = [(0, False), (-1, False), (0, True), (-1, True)]
    best_score = -1
    best_choice = (0, False)
    for offset, swap in candidates:
        score = 0
        for row, col in cell_pairs:
            row_idx = int(col) + offset if swap else int(row) + offset
            col_idx = int(row) + offset if swap else int(col) + offset
            values = [float(arr[row_idx, col_idx]) for arr in arrays if 0 <= row_idx < arr.shape[0] and 0 <= col_idx < arr.shape[1]]
            if values and max(values, key=lambda value: abs(value)) != 0.0:
                score += 1
        if score > best_score:
            best_score = score
            best_choice = (offset, swap)
    return best_choice


def _write_rch_numeric(output_path, arrays):
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for idx, arr in enumerate(arrays):
            if idx:
                handle.write("\n")
            np.savetxt(handle, np.asarray(arr), fmt="%.6g")


def _apply_rch_rate_update(rch, gdf, selected_ids, new_rate, rate_mode, periods_for_update, output_path):
    arrays = _rch_to_arrays(rch)
    nrow = int(gdf["ROW"].max())
    ncol = int(gdf["COL"].max())
    normalized = _normalize_rch_arrays(arrays, nrow, ncol)
    if not normalized:
        _write_rch_numeric(output_path, [])
        return 0
    offset, swap = _choose_rch_indexing(normalized, gdf)
    cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
    selected_cells = [cell_lookup[cid] for cid in selected_ids if cid in cell_lookup]
    periods = list(range(len(normalized))) if not periods_for_update else [int(p) for p in periods_for_update if 0 <= int(p) < len(normalized)]
    if not periods:
        periods = list(range(len(normalized)))
    for per in periods:
        arr = np.array(normalized[per], dtype=float, copy=True)
        for row, col in selected_cells:
            row_idx = int(col) + offset if swap else int(row) + offset
            col_idx = int(row) + offset if swap else int(col) + offset
            if 0 <= row_idx < arr.shape[0] and 0 <= col_idx < arr.shape[1]:
                if rate_mode == "scale_percent":
                    arr[row_idx, col_idx] *= 1.0 + (float(new_rate) / 100.0)
                else:
                    arr[row_idx, col_idx] = float(new_rate)
        normalized[per] = arr
    wrote = False
    if hasattr(rch, "write_file") and hasattr(rch, "rech"):
        try:
            rch.rech = np.array(normalized[0]) if len(normalized) == 1 else np.stack([np.array(arr) for arr in normalized])
            rch.write_file(str(output_path))
            wrote = True
        except Exception:
            wrote = False
    if not wrote:
        _write_rch_numeric(output_path, normalized)
    return len(selected_cells)


def _publish_updated_flux(ckan_url, jwt_token, source_dataset_name, source_dataset, source_resource, output_path, provenance_details, target_var, new_dataset_name=None, new_dataset_title=None, source_url=None, change_summary=None, maintainer_username=None):
    mint_svo = _resolve_mint_svo(source_resource, source_dataset, target_var)
    if not mint_svo:
        raise RuntimeError("MINT_SVO missing in source metadata.")
    timestamp = _now_iso()
    new_name = _slugify(new_dataset_name) if new_dataset_name else _slugify(f"{source_dataset_name}-updated-{timestamp}")
    if new_name == source_dataset_name:
        raise RuntimeError("dataset name conflict with original; choose a new name")
    provenance = {
        "timestamp": timestamp,
        "action": "rch_rate_update" if target_var == CKAN_RCH_STANDARD_VAR else "wel_rate_update",
        "details": json.dumps(provenance_details, sort_keys=True),
        "source_dataset_id": source_dataset.get("id", ""),
        "source_dataset_name": source_dataset_name,
        "source_resource_id": source_resource.get("id", ""),
        "source_resource_name": source_resource.get("name", ""),
    }
    try:
        existing_dataset = _package_show(ckan_url, jwt_token, new_name)
    except Exception:
        existing_dataset = None
    if existing_dataset:
        if maintainer_username:
            existing_maintainer = str(existing_dataset.get("maintainer", "")).strip().lower()
            existing_email = str(existing_dataset.get("maintainer_email", "")).strip().lower()
            maintainer_key = str(maintainer_username).strip().lower()
            if existing_maintainer and maintainer_key != existing_maintainer and maintainer_key != existing_email:
                raise RuntimeError("dataset maintainer mismatch")
        dataset_id = existing_dataset["id"]
    else:
        created_dataset = _create_dataset(
            ckan_url,
            jwt_token,
            _build_dataset_payload(
                source_dataset,
                new_name,
                provenance,
                mint_svo,
                source_url=source_url,
                change_summary=change_summary,
                maintainer_username=maintainer_username,
                new_title=new_dataset_title,
            ),
        )
        existing_dataset = created_dataset
        dataset_id = created_dataset["id"]
    resource_payload = _build_resource_payload(
        source_resource,
        provenance,
        mint_svo,
        source_url=source_url,
        change_summary=change_summary,
        default_name="RCH" if target_var == CKAN_RCH_STANDARD_VAR else "WEL",
        default_format="RCH" if target_var == CKAN_RCH_STANDARD_VAR else "WEL",
        resource_name=Path(output_path).stem,
    )
    created_resource = _create_resource_upload(ckan_url, jwt_token, dataset_id, Path(output_path), resource_payload)
    return {"dataset": existing_dataset, "resource": created_resource}


def main():
    payload = json.loads(ctx.get_input("APPLY_PAYLOAD_JSON"))
    ckan_url = ctx.get_input("CKAN_URL")
    jwt_token = ctx.get_input("JWT_TOKEN")
    loaded_dataset = str(payload.get("dataset") or "").strip()
    if not loaded_dataset:
        raise ValueError("No dataset selected.")
    selected_ids = [int(value) for value in payload.get("selectedIds") or []]
    if not selected_ids:
        raise ValueError("No cells selected.")
    flux_source = str(payload.get("fluxSource") or "wel").strip().lower()
    new_rate = float(payload.get("newRate") or 0.0)
    rate_mode = str(payload.get("rateMode") or "set").strip()
    layers = [int(value) for value in payload.get("layers") or []]
    periods = [int(value) for value in payload.get("periods") or []]
    add_missing = bool(payload.get("addMissing"))
    dataset_name = str(payload.get("datasetName") or "").strip()
    dataset_title = str(payload.get("datasetTitle") or "").strip()
    output_name = str(payload.get("outputName") or "").strip()
    source_url = str(payload.get("sourceUrl") or "").strip()
    change_summary = str(payload.get("changeSummary") or "").strip()
    tapis_username = str(payload.get("tapisUsername") or "").strip()

    with tempfile.TemporaryDirectory(prefix="flopy-workflow-") as tmpdir:
        base_dir = Path(tmpdir)
        source_dataset = _package_show(ckan_url, jwt_token, loaded_dataset)
        resources = source_dataset.get("resources", [])
        target_var = CKAN_RCH_STANDARD_VAR if flux_source == "rch" else CKAN_WEL_STANDARD_VAR
        target_resource = next((res for res in resources if _resource_has_standard_var(res, target_var)), None)
        if not target_resource:
            raise RuntimeError(f"Could not find {flux_source.upper()} resource in source dataset.")
        grid_resource = next((res for res in resources if _resource_has_standard_var(res, CKAN_GRID_STANDARD_VAR)), None)
        if not grid_resource:
            raise RuntimeError("Could not find grid resource in source dataset.")

        grid_gdf = _load_grid_resource(grid_resource, base_dir / "grid")
        output_path = base_dir / (output_name or (f"{loaded_dataset}.{flux_source}"))
        if output_path.suffix.lower() != (".rch" if flux_source == "rch" else ".wel"):
            output_path = output_path.with_suffix(".rch" if flux_source == "rch" else ".wel")

        source_path = _download_resource(target_resource, base_dir / flux_source)
        if flux_source == "rch":
            nrow = int(grid_gdf["ROW"].max())
            ncol = int(grid_gdf["COL"].max())
            rch = _load_rch(source_path, nrow=nrow, ncol=ncol, nper=max(periods) + 1 if periods else 1)
            updated = _apply_rch_rate_update(rch, grid_gdf, selected_ids, new_rate, rate_mode, periods, output_path)
        else:
            wel = _load_wel(source_path)
            updated = _apply_rate_update(wel, grid_gdf, selected_ids, new_rate, rate_mode, add_missing, layers, periods, output_path)

        provenance = {
            "selected_cell_count": len(selected_ids),
            "rate_mode": rate_mode,
            "new_rate": new_rate,
            "periods": periods,
            "change_summary": change_summary,
            "source_url": source_url,
            "flux_source": flux_source,
        }
        if flux_source != "rch":
            provenance["layers"] = layers
            provenance["add_missing"] = add_missing

        publish_result = _publish_updated_flux(
            ckan_url=ckan_url,
            jwt_token=jwt_token,
            source_dataset_name=loaded_dataset,
            source_dataset=source_dataset,
            source_resource=target_resource,
            output_path=output_path,
            provenance_details=provenance,
            target_var=target_var,
            new_dataset_name=dataset_name or None,
            new_dataset_title=dataset_title or None,
            source_url=source_url or None,
            change_summary=change_summary or None,
            maintainer_username=tapis_username or None,
        )
        dataset_id = publish_result["dataset"].get("id", "")
        result = {
            "mode": "workflow",
            "updated": updated,
            "datasetId": dataset_id,
            "resourceId": publish_result["resource"].get("id", ""),
            "outputName": output_path.name,
            "message": f"Updated {updated} cells. Wrote {output_path.name}. Published CKAN dataset {dataset_id}.",
        }
        ctx.set_output("result_json", json.dumps(result))
        ctx.stdout(json.dumps(result))


main()
'''


def build_flux_percent_pipeline(pipeline_id: str, archive_ids: Iterable[str] | None = None) -> dict:
    """Return a single-task workflow definition for percent-based WEL/RCH updates."""
    archives = [archive_id.strip() for archive_id in (archive_ids or []) if str(archive_id).strip()]
    task = {
        "id": "apply-flux-percent",
        "type": "function",
        "description": "Load WEL or RCH data, apply set/scale updates for selected cells and periods, and publish the derived resource to CKAN.",
        "runtime": "python3.9",
        "installer": "pip",
        "packages": [
            "flopy==3.9.0",
            "numpy==2.1.2",
            "pandas==2.2.3",
            "geopandas==1.0.1",
            "fiona==1.10.1",
            "pyproj==3.7.0",
            "shapely==2.0.6",
            "requests==2.32.3",
        ],
        "code": base64.b64encode(WORKFLOW_TASK_CODE.encode("utf-8")).decode("ascii"),
        "input": {
            "APPLY_PAYLOAD_JSON": {
                "type": "string",
                "value_from": {"params": "apply_payload_json"},
            },
            "CKAN_URL": {
                "type": "string",
                "value_from": {"params": "ckan_url"},
            },
            "JWT_TOKEN": {
                "type": "string",
                "value_from": {"params": "jwt_token"},
            },
        },
        "output": {
            "result_json": {"type": "string"},
        },
    }
    pipeline = {
        "id": pipeline_id,
        "description": "FloPy interactive ETL workflow for percent-based WEL/RCH updates.",
        "tasks": [task],
    }
    if archives:
        pipeline["archives"] = archives
    return pipeline
