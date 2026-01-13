#!/usr/bin/env python3
"""Helpers for interactive WEL updates using the EBFZ grid and Plotly."""

from __future__ import annotations

import shutil
import zipfile
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.request import urlopen

import flopy
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from plotly import colors as pc
import ipywidgets as widgets
from IPython.display import display, clear_output

warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)


def _flatten_single_dir(root: Path) -> None:
    children = [p for p in root.iterdir() if p.is_dir()]
    if len(children) == 1:
        inner = children[0]
        for item in inner.iterdir():
            shutil.move(str(item), root)
        inner.rmdir()


def ensure_barton_springs_wel(wel_path: Path) -> None:
    if wel_path.exists():
        return
    url = (
        "https://ckan.tacc.utexas.edu/dataset/18400624-423c-42b5-ad56-6c73322584bd/"
        "resource/9c7b25c4-8cea-4965-a07a-d9b3867f18a9/"
        "download/barton_springs_2001_2010average.wel"
    )
    from urllib.request import urlretrieve

    urlretrieve(url, wel_path)


def ensure_ebfz_grid(grid_dir: Path) -> Path:
    grid_gdb = grid_dir / "ebfz_b_grid.gdb"
    if grid_gdb.exists():
        return grid_gdb
    url = (
        "https://ckan.tacc.utexas.edu/dataset/18400624-423c-42b5-ad56-6c73322584bd/"
        "resource/f07a257c-1d88-4819-bd5d-a104c5e3fe5b/"
        "download/ebfz_b_grid.zip"
    )
    from urllib.request import urlretrieve

    grid_zip = grid_dir.with_suffix(".zip")
    if not grid_zip.exists():
        urlretrieve(url, grid_zip)
    grid_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(grid_zip, "r") as zf:
        zf.extractall(grid_dir)
    _flatten_single_dir(grid_dir)
    return grid_gdb


def scan_wel_metadata(path: Path) -> Tuple[int, int, int, int]:
    def strip_comment(line: str) -> str:
        for token in ("#", ";"):
            if token in line:
                line = line.split(token, 1)[0]
        return line.strip()

    lines = [strip_comment(line) for line in path.read_text().splitlines()]
    data_lines = [line for line in lines if line]
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
                raise ValueError("Unexpected end of file while scanning wells.")
            parts = data_lines[idx].split()
            idx += 1
            k, i, j = (int(parts[0]), int(parts[1]), int(parts[2]))
            max_k = max(max_k, k)
            max_i = max(max_i, i)
            max_j = max(max_j, j)
    return nper, max_k, max_i, max_j


def load_wel(wel_path: Path) -> flopy.modflow.ModflowWel:
    nper, nlay, nrow, ncol = scan_wel_metadata(wel_path)
    model = flopy.modflow.Modflow(modelname="wel_read", model_ws=str(wel_path.parent))
    flopy.modflow.ModflowDis(
        model,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=[0.0] * nlay,
    )
    return flopy.modflow.ModflowWel.load(str(wel_path), model)


def load_grid_gdf(grid_gdb: Path, layer_name: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(grid_gdb, layer=layer_name)
    return _prepare_grid_gdf(gdf)


def _extract_standard_vars(resource: Dict) -> List[str]:
    candidates: List[str] = []
    direct_keys = ("MINT Standard Variables", "mint_standard_variables", "standard_variables")
    for key in direct_keys:
        if key in resource and resource[key] is not None:
            candidates.append(resource[key])
    extras = resource.get("extras")
    if isinstance(extras, list):
        for item in extras:
            if not isinstance(item, dict):
                continue
            if item.get("key") == "MINT Standard Variables":
                candidates.append(item.get("value"))
    elif isinstance(extras, dict) and "MINT Standard Variables" in extras:
        candidates.append(extras.get("MINT Standard Variables"))

    values: List[str] = []
    for entry in candidates:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            values.extend([str(v) for v in entry])
            continue
        if isinstance(entry, str):
            text = entry.strip()
            if not text:
                continue
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        values.extend([str(v) for v in parsed])
                        continue
                except json.JSONDecodeError:
                    pass
            values.extend([v.strip() for v in text.replace(";", ",").split(",") if v.strip()])
            continue
        values.append(str(entry))
    return values


def _resource_has_standard_var(resource: Dict, target: str) -> bool:
    target_key = target.strip().lower()
    for value in _extract_standard_vars(resource):
        if value.strip().lower() == target_key:
            return True
    return False


def _search_ckan_datasets() -> List[Dict]:
    base_url = "https://ckan.tacc.utexas.edu/api/3/action/package_search"
    start = 0
    rows = 100
    matched: List[Dict] = []
    targets = {
        "wel": "groundwater_well__recharge_volume_flux",
        "grid": "model_grid_cell_boundary_groundwater__interfacial_hydraulic_conductance",
        "rch": "groundwater__recharge_volume_flux",
    }

    while True:
        url = f"{base_url}?rows={rows}&start={start}"
        with urlopen(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError("CKAN search failed.")
        result = payload["result"]
        results = result.get("results", [])
        for pkg in results:
            resources = pkg.get("resources", [])
            matches = {"wel": [], "grid": [], "rch": []}
            for res in resources:
                if _resource_has_standard_var(res, targets["wel"]):
                    matches["wel"].append(res)
                if _resource_has_standard_var(res, targets["grid"]):
                    matches["grid"].append(res)
                if _resource_has_standard_var(res, targets["rch"]):
                    matches["rch"].append(res)
            if all(matches[key] for key in matches):
                matched.append(
                    {
                        "name": pkg.get("name") or pkg.get("id"),
                        "title": pkg.get("title") or pkg.get("name") or pkg.get("id"),
                        "matches": matches,
                    }
                )
        start += rows
        if start >= result.get("count", 0):
            break
    return matched


def _download_ckan_resource(resource: Dict, dest_dir: Path) -> Path:
    url = resource.get("url")
    if not url:
        raise ValueError("CKAN resource missing URL.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?", 1)[0]).name
    if not filename:
        filename = resource.get("name") or resource.get("id") or "resource"
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    from urllib.request import urlretrieve

    urlretrieve(url, dest_path)
    return dest_path


def _extract_zip(zip_path: Path) -> Path:
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    _flatten_single_dir(extract_dir)
    return extract_dir


def _prepare_grid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        centroids = gdf.geometry.centroid
        gdf["_lon"] = centroids.x
        gdf["_lat"] = centroids.y
        return gdf

    try:
        if gdf.crs.is_geographic:
            projected = gdf.to_crs(gdf.estimate_utm_crs())
        else:
            projected = gdf
    except Exception:
        projected = gdf

    centroids = projected.geometry.centroid
    try:
        centroids_ll = centroids.to_crs("EPSG:4326")
    except Exception:
        centroids_ll = centroids

    gdf["_lon"] = centroids_ll.x
    gdf["_lat"] = centroids_ll.y
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _load_grid_from_gdb(gdb_path: Path) -> gpd.GeoDataFrame:
    try:
        import fiona
    except ImportError as exc:
        raise ImportError("fiona is required to read a geodatabase.") from exc
    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"No layers found in {gdb_path}.")
    gdf = gpd.read_file(gdb_path, layer=layers[0])
    return _prepare_grid_gdf(gdf)


def _find_grid_data_path(root: Path) -> Path | None:
    if root.is_dir():
        gdbs = list(root.rglob("*.gdb"))
        if gdbs:
            return gdbs[0]
        for ext in (".shp", ".geojson", ".gpkg", ".json"):
            matches = list(root.rglob(f"*{ext}"))
            if matches:
                return matches[0]
        return None
    return root


def _load_grid_resource(resource: Dict, dest_dir: Path) -> gpd.GeoDataFrame:
    path = _download_ckan_resource(resource, dest_dir)
    if path.suffix.lower() == ".zip":
        path = _extract_zip(path)
    grid_path = _find_grid_data_path(path)
    if grid_path is None:
        raise ValueError("No grid dataset found after extracting resource.")
    if grid_path.suffix.lower() == ".gdb":
        return _load_grid_from_gdb(grid_path)
    gdf = gpd.read_file(grid_path)
    return _prepare_grid_gdf(gdf)


def _load_rch_numeric(rch_path: Path, nrow: int, ncol: int) -> np.ndarray:
    text = rch_path.read_text()
    values: List[float] = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if len(values) < nrow * ncol:
        raise ValueError("Not enough numeric values to build an RCH array.")
    count = len(values)
    if count % (nrow * ncol) == 0:
        arr = np.array(values, dtype=float).reshape((count // (nrow * ncol), nrow, ncol))
    else:
        arr = np.array(values[-nrow * ncol :], dtype=float).reshape((nrow, ncol))
    return arr


def load_rch(rch_path: Path, nrow: int, ncol: int, nper: int = 1):
    model = flopy.modflow.Modflow(modelname="rch_read", model_ws=str(rch_path.parent))
    flopy.modflow.ModflowDis(
        model,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=[0.0],
    )
    try:
        return flopy.modflow.ModflowRch.load(str(rch_path), model)
    except Exception:
        return _load_rch_numeric(rch_path, nrow, ncol)


def _rch_to_arrays(rch) -> List[np.ndarray]:
    if hasattr(rch, "rech"):
        data = rch.rech
    else:
        data = rch
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


def _map_rch_cells(
    arrays: Sequence[np.ndarray], gdf: gpd.GeoDataFrame
) -> Dict[int, float]:
    cell_lookup = dict(zip(zip(gdf["ROW"], gdf["COL"]), gdf["CELL_ID"]))
    nrow = int(gdf["ROW"].max())
    ncol = int(gdf["COL"].max())
    normalized: List[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size == nrow * ncol:
                normalized.append(arr.reshape((nrow, ncol)))
                continue
            if arr.size % (nrow * ncol) == 0:
                count = arr.size // (nrow * ncol)
                reshaped = arr.reshape((count, nrow, ncol))
                normalized.extend([reshaped[idx] for idx in range(count)])
                continue
        if arr.ndim == 2:
            normalized.append(arr)
            continue
        if arr.ndim == 3:
            normalized.extend([arr[idx] for idx in range(arr.shape[0])])
            continue

    def _collect(offset: int, swap: bool) -> Dict[int, float]:
        cells: Dict[int, float] = {}
        for row, col in cell_lookup:
            if swap:
                row_idx = int(col) + offset
                col_idx = int(row) + offset
            else:
                row_idx = int(row) + offset
                col_idx = int(col) + offset
            values = []
            for arr in normalized:
                if 0 <= row_idx < arr.shape[0] and 0 <= col_idx < arr.shape[1]:
                    values.append(float(arr[row_idx, col_idx]))
            if not values:
                continue
            flux = max(values, key=lambda v: abs(v))
            cells[int(cell_lookup[(row, col)])] = flux
        return cells

    candidates = [
        _collect(0, False),
        _collect(-1, False),
        _collect(0, True),
        _collect(-1, True),
    ]
    return max(candidates, key=lambda c: sum(1 for v in c.values() if v != 0.0))


def build_rch_cells(rch, gdf: gpd.GeoDataFrame) -> Dict[int, float]:
    arrays = _rch_to_arrays(rch)
    return _map_rch_cells(arrays, gdf)


def build_rch_cells_for_periods(
    rch,
    gdf: gpd.GeoDataFrame,
    periods: Sequence[int],
) -> Dict[int, float]:
    arrays = _rch_to_arrays(rch)
    if periods:
        selected = [arrays[p] for p in periods if 0 <= p < len(arrays)]
        if not selected:
            selected = arrays
    else:
        selected = arrays
    if not selected:
        return {}
    return _map_rch_cells(selected, gdf)


def _update_flux_customdata(
    fig: go.FigureWidget, gdf: gpd.GeoDataFrame, flux_values: Sequence[float]
) -> None:
    fig.data[0].customdata = np.stack(
        [
            gdf["CELL_ID"],
            flux_values,
            gdf["GCD_Name"].fillna("Unknown").astype(str),
            gdf["PGMA_Name"].fillna("Unknown").astype(str),
        ],
        axis=1,
    )
    fig.data[1].customdata = np.stack(
        [
            gdf["CELL_ID"],
            gdf["ROW"],
            gdf["COL"],
            flux_values,
            gdf["GCD_Name"].fillna("Unknown").astype(str),
            gdf["PGMA_Name"].fillna("Unknown").astype(str),
        ],
        axis=1,
    )


def _collect_wel_cells_for_period_data(
    wel: flopy.modflow.ModflowWel,
    cell_id_lookup: Dict[Tuple[int, int], int],
    period: int,
    row_offset: int,
    col_offset: int,
) -> Dict[int, float]:
    cells: Dict[int, float] = {}
    spd = wel.stress_period_data.data
    if period not in spd:
        return cells
    recs = spd[period]
    for rec in recs:
        row = int(rec["i"]) + row_offset
        col = int(rec["j"]) + col_offset
        cell_id = cell_id_lookup.get((row, col))
        if cell_id is None:
            continue
        flux = float(rec["flux"])
        cell_key = int(cell_id)
        if cell_key not in cells or abs(flux) > abs(cells[cell_key]):
            cells[cell_key] = flux
    return cells


def _signed_log(values: Sequence[float]) -> List[float]:
    return [float(np.sign(v) * np.log10(1.0 + abs(v))) for v in values]


def _signed_range(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    if vmin < 0.0 and vmax > 0.0:
        max_abs = max(abs(vmin), abs(vmax))
        return -max_abs, max_abs
    if vmax <= 0.0:
        return vmin, 0.0
    return 0.0, vmax


def _discrete_colorscale(colors: Sequence[str], count: int) -> List[Tuple[float, str]]:
    if count <= 1:
        return [(0.0, colors[0]), (1.0, colors[0])]
    scale: List[Tuple[float, str]] = []
    for idx, color in enumerate(colors):
        t0 = idx / (count - 1)
        t1 = min((idx + 1) / (count - 1), 1.0)
        scale.append((t0, color))
        scale.append((t1, color))
    return scale


def _apply_color_mode(
    fig: go.FigureWidget,
    gdf: gpd.GeoDataFrame,
    wel_cells: Dict[int, float],
    mode: str,
    normalize: bool = False,
    flux_label: str = "Flux",
    force_linear: bool = False,
) -> None:
    cell_ids = gdf["CELL_ID"].tolist()
    flux_values = [float(wel_cells.get(int(cid), 0.0)) for cid in cell_ids]
    diverging = [(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")]

    if mode == "flux":
        nonzero = [v for v in flux_values if v != 0.0]
        all_nonnegative = bool(nonzero) and min(flux_values) >= 0.0
        if (force_linear or normalize or all_nonnegative) and nonzero:
            fmin = min(flux_values)
            fmax = max(flux_values)
            z_vals = flux_values
            zmin, zmax = fmin, fmax
            if zmin == zmax:
                zmin -= 1.0
                zmax += 1.0
            tick_values = [fmin, (fmin + fmax) / 2.0, fmax]
            tick_vals = tick_values
            tick_text = [f"{v:.6g}" for v in tick_values]
            title = flux_label
        else:
            z_vals = _signed_log(flux_values)
            zmin, zmax = _signed_range(z_vals)
            if nonzero:
                fmin = min(nonzero)
                fmax = max(nonzero)
                if fmin < 0.0 and fmax > 0.0:
                    fmax_abs = max(abs(fmin), abs(fmax))
                    tick_values = [
                        -fmax_abs,
                        -fmax_abs / 10.0,
                        0.0,
                        fmax_abs / 10.0,
                        fmax_abs,
                    ]
                elif fmax <= 0.0:
                    tick_values = [fmin, fmin / 10.0, 0.0]
                else:
                    tick_values = [0.0, fmax / 10.0, fmax]
                tick_vals = [np.sign(v) * np.log10(1.0 + abs(v)) for v in tick_values]
                tick_text = [f"{v:.0f}" for v in tick_values]
            else:
                tick_vals, tick_text = None, None
            title = flux_label

        with fig.batch_update():
            fig.data[0].update(z=z_vals, zmin=zmin, zmax=zmax, colorscale=diverging)
            fig.data[1].marker.update(
                color=z_vals,
                colorscale=diverging,
                cmin=zmin,
                cmax=zmax,
                cauto=False,
                showscale=True,
                colorbar=dict(
                    title=title,
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickmode="array" if tick_vals is not None else "auto",
                    orientation="v",
                    x=1.02,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    len=0.8,
                ),
            )
        return

    col = "GCD_Name" if mode == "GCD_Name" else "PGMA_Name"
    categories = gdf[col].fillna("Unknown").astype(str).tolist()
    labels = list(dict.fromkeys(categories))
    label_map = {label: idx for idx, label in enumerate(labels)}
    codes = [label_map[value] for value in categories]
    palette = pc.qualitative.Safe
    colors = [palette[idx % len(palette)] for idx in range(len(labels))]
    colorscale = _discrete_colorscale(colors, max(1, len(labels)))
    cmax = max(1, len(labels) - 1)

    with fig.batch_update():
        fig.data[0].update(z=codes, zmin=0.0, zmax=cmax, colorscale=colorscale)
        fig.data[1].marker.update(
            color=codes,
            colorscale=colorscale,
            cmin=0.0,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=col,
                tickvals=list(range(len(labels))),
                ticktext=labels,
                orientation="v",
                x=1.02,
                xanchor="left",
                y=0.5,
                yanchor="middle",
                len=0.8,
            ),
        )


def build_plotly_selector(
    gdf: gpd.GeoDataFrame, wel_cells: Dict[int, float] | None = None
) -> Tuple[go.FigureWidget, widgets.Label, widgets.Label, set, callable]:
    gdf_map = gdf[["CELL_ID", "ROW", "COL", "geometry"]].copy()
    grid_geojson = gdf_map.set_index("CELL_ID").__geo_interface__
    center_lat = float(gdf["_lat"].median())
    center_lon = float(gdf["_lon"].median())

    fig = go.FigureWidget()
    fig._skip_invalid = True
    wel_cells = wel_cells or {}
    z_values = [float(wel_cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
    fig.add_trace(
        go.Choroplethmap(
            geojson=grid_geojson,
            locations=gdf["CELL_ID"],
            z=z_values,
            colorscale=[
                (0.0, "#2b6cb0"),
                (0.5, "#ffffff"),
                (1.0, "#c53030"),
            ],
            marker_opacity=0.35,
            marker_line_width=0.5,
            marker_line_color="#666",
            showscale=False,
            name="Grid",
            customdata=np.stack(
                [
                    gdf["CELL_ID"],
                    z_values,
                    gdf["GCD_Name"].fillna("Unknown").astype(str),
                    gdf["PGMA_Name"].fillna("Unknown").astype(str),
                ],
                axis=1,
            ),
            hovertemplate=(
                "CELL_ID=%{customdata[0]}<br>"
                "Flux=%{customdata[1]:.2f}<br>"
                "GCD=%{customdata[2]}<br>"
                "PGMA=%{customdata[3]}<extra></extra>"
            ),
        )
    )
    scatter_flux = [float(wel_cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
    fig.add_trace(
        go.Scattermap(
            lon=gdf["_lon"],
            lat=gdf["_lat"],
            mode="markers",
            marker=dict(
                size=6,
                color=scatter_flux,
                colorscale=[(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")],
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(
                    title="Flux",
                    tickvals=None,
                    ticktext=None,
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.15,
                    yanchor="top",
                    len=0.7,
                ),
            ),
            name="Cells",
            customdata=np.stack(
                [
                    gdf["CELL_ID"],
                    gdf["ROW"],
                    gdf["COL"],
                    scatter_flux,
                    gdf["GCD_Name"].fillna("Unknown").astype(str),
                    gdf["PGMA_Name"].fillna("Unknown").astype(str),
                ],
                axis=1,
            ),
            hovertemplate=(
                "CELL_ID=%{customdata[0]}<br>"
                "ROW=%{customdata[1]} COL=%{customdata[2]}<br>"
                "Flux=%{customdata[3]:.2f}<br>"
                "GCD=%{customdata[4]}<br>"
                "PGMA=%{customdata[5]}<extra></extra>"
            ),
            selected=dict(marker=dict(size=8, color="#d62728")),
            unselected=dict(marker=dict(opacity=0.5)),
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=[],
            lat=[],
            mode="markers",
            marker=dict(size=10, color="#d62728"),
            name="Selected",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    _apply_color_mode(fig, gdf, wel_cells, "flux", flux_label="Well")
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="lasso",
        clickmode="event+select",
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8,
        ),
    )

    selected_ids: set = set()
    id_to_index = {int(cid): idx for idx, cid in enumerate(gdf["CELL_ID"])}
    status = widgets.Label(value="Selected cells: 0")
    map_status = None

    def _apply_selection(cell_ids: Iterable[int]) -> None:
        selected_ids.clear()
        selected_ids.update(int(cid) for cid in cell_ids)
        selected_indices = [id_to_index[cid] for cid in selected_ids if cid in id_to_index]
        fig.data[1].selectedpoints = selected_indices
        selected_rows = gdf.iloc[selected_indices] if selected_indices else gdf.iloc[[]]
        fig.data[2].update(lon=selected_rows["_lon"], lat=selected_rows["_lat"])
        status.value = f"Selected cells: {len(selected_ids)}"

    def _update_selected_from_inds(point_inds):
        _apply_selection(int(fig.data[1].customdata[idx][0]) for idx in point_inds)

    def _on_selection(trace, points, state):
        if hasattr(points, "point_inds") and points.point_inds:
            _update_selected_from_inds(points.point_inds)
        else:
            _apply_selection([])

    def _on_click(trace, points, state):
        if hasattr(points, "point_inds") and points.point_inds:
            toggled = set(selected_ids)
            for idx in points.point_inds:
                cid = int(fig.data[1].customdata[idx][0])
                if cid in toggled:
                    toggled.remove(cid)
                else:
                    toggled.add(cid)
            _apply_selection(toggled)

    fig.data[1].on_selection(_on_selection)
    fig.data[1].on_click(_on_click)

    return fig, status, map_status, selected_ids, _apply_selection


def apply_rate_update(
    wel: flopy.modflow.ModflowWel,
    gdf: gpd.GeoDataFrame,
    selected_ids: Iterable[int],
    new_rate: float,
    rate_mode: str,
    add_missing: bool,
    layers_for_new: Sequence[int],
    periods_for_update: Sequence[int],
    output_path: Path,
) -> int:
    spd = wel.stress_period_data.data
    cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
    selected_cells = {cell_lookup[cid] for cid in selected_ids if cid in cell_lookup}
    new_spd = {}
    for per, recs in spd.items():
        recs = recs.copy()
        if periods_for_update and per not in periods_for_update:
            new_spd[per] = recs
            continue
        mask = []
        for rec in recs:
            i = int(rec["i"])
            j = int(rec["j"])
            mask.append((i, j) in selected_cells)
        mask = np.array(mask, dtype=bool)
        if rate_mode == "scale_percent":
            recs["flux"][mask] *= 1.0 + (float(new_rate) / 100.0)
        else:
            recs["flux"][mask] = float(new_rate)
        if add_missing and selected_cells:
            existing = set((int(r["i"]), int(r["j"])) for r in recs)
            to_add = [cell for cell in selected_cells if cell not in existing]
            if to_add:
                layers = [int(layer) for layer in layers_for_new] or [1]
                new_recs = np.zeros(len(recs) + len(to_add) * len(layers), dtype=recs.dtype)
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
        new_spd[per] = recs
    wel.stress_period_data = new_spd
    wel.write_file(str(output_path))
    return len(selected_cells)


def build_ui(
    wel: flopy.modflow.ModflowWel,
    gdf: gpd.GeoDataFrame,
    show_dataset_controls: bool = True,
    rch: flopy.modflow.ModflowRch | None = None,
) -> widgets.Widget:
    cell_id_lookup = dict(zip(zip(gdf["ROW"], gdf["COL"]), gdf["CELL_ID"]))

    def _collect_wel_cells(row_offset: int, col_offset: int) -> Dict[int, float]:
        cells: Dict[int, float] = {}
        for recs in wel.stress_period_data.data.values():
            for rec in recs:
                row = int(rec["i"]) + row_offset
                col = int(rec["j"]) + col_offset
                cell_id = cell_id_lookup.get((row, col))
                if cell_id is None:
                    continue
                flux = float(rec["flux"])
                cell_key = int(cell_id)
                if cell_key not in cells or abs(flux) > abs(cells[cell_key]):
                    cells[cell_key] = flux
        return cells

    def _collect_wel_cells_for_period(
        period: int, row_offset: int, col_offset: int
    ) -> Dict[int, float]:
        return _collect_wel_cells_for_period_data(
            wel, cell_id_lookup, period, row_offset, col_offset
        )

    wel_cells = _collect_wel_cells(0, 0)
    wel_cells_offset = _collect_wel_cells(1, 1)
    use_offset = len(wel_cells_offset) > len(wel_cells)
    if use_offset:
        wel_cells = wel_cells_offset

    rch_cells: Dict[int, float] = {}
    if rch is not None:
        try:
            rch_cells = build_rch_cells(rch, gdf)
        except Exception as exc:
            rch_cells = {}

    fig, status, map_status, selected_ids, apply_selection = build_plotly_selector(
        gdf, wel_cells
    )
    match_info = widgets.Label(
        value=(
            f"WEL/grid matches: {len(wel_cells)} cells"
            + (" (using +1 row/col offset)" if use_offset else "")
        )
    )

    help_text = widgets.HTML(
        """
        <b>Selection:</b> Use lasso/box on the map or click points to toggle selection.<br>
        <b>Rate mode:</b> <i>Set</i> replaces the rate, <i>Scale (%)</i> applies a percent change.<br>
        <b>New rate:</b> For <i>Set</i>, use the absolute pumping rate (negative for pumping). For
        <i>Scale (%)</i>, use percent change (e.g., 10 = +10%, -10 = -10%).<br>
        <b>Stress periods:</b> Controls which periods are displayed and updated.<br>
        <b>Layer (k):</b> Used only when adding missing wells. 1 = top layer.<br>
        <b>Add missing wells:</b> Add wells in selected cells not present in the WEL.
        """
    )
    rate_mode = widgets.Dropdown(
        options=[("Set", "set"), ("Scale (%)", "scale_percent")],
        value="set",
        description="Rate mode",
    )
    color_by = widgets.Dropdown(
        options=[("Flux", "flux"), ("GCD_Name", "GCD_Name"), ("PGMA_Name", "PGMA_Name")],
        value="flux",
        description="Color by",
    )
    spd_keys = sorted(list(wel.stress_period_data.data.keys()))
    if not spd_keys:
        spd_keys = [0]
    period_options = [(f"SP {idx + 1}", idx) for idx in spd_keys]
    period_select = widgets.SelectMultiple(
        options=period_options,
        value=(spd_keys[0],),
        description="Stress periods",
    )
    active_periods: List[int] = [int(spd_keys[0])]

    flux_source_options = [("WEL", "wel")]
    if rch is not None:
        flux_source_options.append(("RCH", "rch"))
    flux_source = widgets.Dropdown(
        options=flux_source_options,
        value="wel",
        description="Flux source",
    )
    rch_status = widgets.Label(
        value="RCH loaded: yes" if rch is not None else "RCH loaded: no"
    )
    rch_stats = widgets.Label(value="")
    if rch is not None:
        try:
            arrays = _rch_to_arrays(rch)
            if arrays:
                flat = np.concatenate([np.ravel(arr) for arr in arrays])
                if flat.size:
                    nonzero = int(np.count_nonzero(flat))
                    rmin = float(np.nanmin(flat))
                    rmax = float(np.nanmax(flat))
                    rch_stats.value = (
                        f"RCH stats: min={rmin:.6g}, max={rmax:.6g}, nonzero={nonzero}"
                    )
                else:
                    rch_stats.value = "RCH stats: empty array"
            else:
                rch_stats.value = "RCH stats: no arrays"
        except Exception as exc:
            rch_stats.value = f"RCH stats: error ({type(exc).__name__})"
    category_select = widgets.Combobox(
        options=[],
        description="Category",
        placeholder="Select a category",
    )
    select_category_btn = widgets.Button(description="Select category")
    rate_input = widgets.FloatText(value=-20.0, description="New rate")
    nlay = 1
    if hasattr(wel, "parent") and wel.parent is not None:
        nlay = int(getattr(wel.parent.dis, "nlay", 1))
    elif hasattr(wel, "model") and wel.model is not None:
        nlay = int(getattr(wel.model.dis, "nlay", 1))
    elif hasattr(wel, "_model") and wel._model is not None:
        nlay = int(getattr(wel._model.dis, "nlay", 1))
    layer_options = [(str(layer), layer) for layer in range(1, nlay + 1)]
    layer_input = widgets.SelectMultiple(
        options=layer_options,
        value=(1,),
        description="Layer (k)",
    )
    add_missing = widgets.Checkbox(value=False, description="Add missing wells")
    save_btn = widgets.Button(description="Apply + Save", button_style="primary")
    output = widgets.Output()
    dataset_fetch_btn = widgets.Button(description="Fetch CKAN datasets")
    dataset_dropdown = widgets.Dropdown(options=[], description="Dataset")
    dataset_status = widgets.Label(value="CKAN datasets: not loaded")
    dataset_lookup: Dict[str, Dict] = {}

    def _apply_and_save(_):
        with output:
            clear_output()
            if not selected_ids:
                print("No cells selected.")
                return
            updated = apply_rate_update(
                wel,
                gdf,
                selected_ids,
                rate_input.value,
                rate_mode.value,
                add_missing.value,
                list(layer_input.value),
                active_periods,
                Path("barton_springs_updated.wel"),
            )
            print(f"Updated {updated} cells. Wrote barton_springs_updated.wel")

    save_btn.on_click(_apply_and_save)

    def _on_color_change(change):
        if change.get("name") == "value":
            active_cells = wel_cells if flux_source.value == "wel" else rch_cells
            label = "Well" if flux_source.value == "wel" else "Recharge"
            _apply_color_mode(
                fig,
                gdf,
                active_cells,
                change["new"],
                normalize=False,
                flux_label=label,
                force_linear=(flux_source.value == "rch"),
            )
            if change["new"] == "flux":
                category_select.options = []
                category_select.value = ""
                category_select.disabled = True
                select_category_btn.disabled = True
            else:
                col = change["new"]
                categories = (
                    gdf[col].fillna("Unknown").astype(str).drop_duplicates().tolist()
                )
                category_select.options = categories
                category_select.value = ""
                category_select.disabled = False
                select_category_btn.disabled = False

    color_by.observe(_on_color_change, names="value")

    def _on_flux_source_change(change):
        if change.get("name") == "value":
            active_cells = wel_cells if change["new"] == "wel" else rch_cells
            flux_values = [float(active_cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
            label = "Well" if change["new"] == "wel" else "Recharge"
            _update_flux_customdata(fig, gdf, flux_values)
            _apply_color_mode(
                fig,
                gdf,
                active_cells,
                color_by.value,
                normalize=False,
                flux_label=label,
                force_linear=(change["new"] == "rch"),
            )

    flux_source.observe(_on_flux_source_change, names="value")

    def _refresh_period_cells() -> None:
        nonlocal wel_cells, rch_cells, use_offset
        if active_periods:
            wel_cells_all = {}
            for period in active_periods:
                period_cells = _collect_wel_cells_for_period(period, 0, 0)
                for cid, flux in period_cells.items():
                    if cid not in wel_cells_all or abs(flux) > abs(wel_cells_all[cid]):
                        wel_cells_all[cid] = flux
            wel_cells_offset = {}
            for period in active_periods:
                period_cells = _collect_wel_cells_for_period(period, 1, 1)
                for cid, flux in period_cells.items():
                    if cid not in wel_cells_offset or abs(flux) > abs(wel_cells_offset[cid]):
                        wel_cells_offset[cid] = flux
        else:
            wel_cells_all = _collect_wel_cells(0, 0)
            wel_cells_offset = _collect_wel_cells(1, 1)
        use_offset = len(wel_cells_offset) > len(wel_cells_all)
        wel_cells = wel_cells_offset if use_offset else wel_cells_all
        if rch is not None:
            try:
                rch_cells = build_rch_cells_for_periods(rch, gdf, active_periods)
            except Exception as exc:
                rch_cells = {}
        match_info.value = (
            f"WEL/grid matches: {len(wel_cells)} cells"
            + (" (using +1 row/col offset)" if use_offset else "")
        )
        active_cells = wel_cells if flux_source.value == "wel" else rch_cells
        flux_values = [float(active_cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
        label = "Well" if flux_source.value == "wel" else "Recharge"
        _update_flux_customdata(fig, gdf, flux_values)
        _apply_color_mode(
            fig,
            gdf,
            active_cells,
            color_by.value,
            normalize=False,
            flux_label=label,
            force_linear=(flux_source.value == "rch"),
        )

    def _on_period_change(change):
        if change.get("name") == "value":
            selected = list(change["new"])
            active_periods.clear()
            active_periods.extend([int(p) for p in selected])
            _refresh_period_cells()

    period_select.observe(_on_period_change, names="value")

    def _update_dataset_details(name: str) -> None:
        dataset = dataset_lookup.get(name)
        dataset_details.value = ""

    def _fetch_ckan(_):
        dataset_status.value = "CKAN datasets: loading..."
        dataset_dropdown.options = []
        dataset_lookup.clear()
        try:
            datasets = _search_ckan_datasets()
        except Exception as exc:
            dataset_status.value = f"CKAN datasets: error ({exc})"
            return
        options = [(d["title"], d["name"]) for d in datasets]
        dataset_dropdown.options = options
        dataset_status.value = f"CKAN datasets: {len(options)} found"
        for d in datasets:
            dataset_lookup[d["name"]] = d
        if options:
            dataset_dropdown.value = options[0][1]
            _update_dataset_details(options[0][1])

    dataset_fetch_btn.on_click(_fetch_ckan)

    def _on_dataset_change(change):
        if change.get("name") == "value":
            _update_dataset_details(change["new"])

    dataset_dropdown.observe(_on_dataset_change, names="value")

    def _select_by_category(_):
        mode = color_by.value
        if mode == "flux":
            return
        target = (category_select.value or "").strip()
        if not target:
            return
        col = mode
        series = gdf[col].fillna("Unknown").astype(str).str.strip()
        options = [str(opt).strip() for opt in category_select.options]
        options_map = {opt.lower(): opt for opt in options if opt}
        target_key = target.lower()
        if target_key in options_map:
            target = options_map[target_key]
            category_select.value = target
        else:
            # Fallback to case-insensitive match against data.
            matches = series.str.lower() == target_key
            cell_ids = gdf.loc[matches, "CELL_ID"].tolist()
            apply_selection(cell_ids)
            return
        matches = series == target
        cell_ids = gdf.loc[matches, "CELL_ID"].tolist()
        apply_selection(cell_ids)

    select_category_btn.on_click(_select_by_category)

    _on_color_change({"name": "value", "new": color_by.value})
    _on_flux_source_change({"name": "value", "new": flux_source.value})
    _refresh_period_cells()

    controls = widgets.VBox(
        [
            match_info,
            flux_source,
            period_select,
            rch_status,
            rch_stats,
            color_by,
            category_select,
            select_category_btn,
            rate_mode,
            rate_input,
            layer_input,
            add_missing,
            save_btn,
            status,
            output,
        ],
        layout=widgets.Layout(width="320px"),
    )
    top_widgets: List[widgets.Widget] = []
    if show_dataset_controls:
        top_widgets.append(
            widgets.VBox(
                [
                    widgets.HBox([dataset_fetch_btn, dataset_dropdown]),
                    dataset_status,
                ]
            )
        )
    return widgets.VBox(
        [
            *top_widgets,
            widgets.HBox([controls, fig]),
            help_text,
        ]
    )


def render_ui(
    wel: flopy.modflow.ModflowWel,
    gdf: gpd.GeoDataFrame,
    show_dataset_controls: bool = True,
    rch: flopy.modflow.ModflowRch | None = None,
) -> None:
    display(build_ui(wel, gdf, show_dataset_controls=show_dataset_controls, rch=rch))


def render_ui_from_ckan(data_dir: Path = Path("ckan_data")) -> None:
    dataset_dropdown = widgets.Dropdown(options=[], description="Dataset")
    dataset_status = widgets.Label(value="CKAN datasets: not loaded")
    dataset_details = widgets.HTML(value="")
    dataset_lookup: Dict[str, Dict] = {}
    load_btn = widgets.Button(description="Load dataset", button_style="primary")
    output = widgets.Output()

    def _update_dataset_details(name: str) -> None:
        dataset = dataset_lookup.get(name)
        return

    def _fetch_ckan(_):
        dataset_status.value = "CKAN datasets: loading..."
        dataset_dropdown.options = []
        dataset_lookup.clear()
        try:
            datasets = _search_ckan_datasets()
        except Exception as exc:
            dataset_status.value = f"CKAN datasets: error ({exc})"
            return
        options = [(d["title"], d["name"]) for d in datasets]
        dataset_dropdown.options = options
        dataset_status.value = f"CKAN datasets: {len(options)} found"
        for d in datasets:
            dataset_lookup[d["name"]] = d
        if options:
            dataset_dropdown.value = options[0][1]
            _update_dataset_details(options[0][1])

    def _on_dataset_change(change):
        if change.get("name") == "value":
            _update_dataset_details(change["new"])

    dataset_dropdown.observe(_on_dataset_change, names="value")

    def _load_selected(_):
        name = dataset_dropdown.value
        if not name:
            return
        dataset = dataset_lookup.get(name)
        if not dataset:
            return
        with output:
            clear_output()
            try:
                base_dir = data_dir / name
                wel_resource = dataset["matches"]["wel"][0]
                grid_resource = dataset["matches"]["grid"][0]
                rch_resource = dataset["matches"]["rch"][0]
                wel_path = _download_ckan_resource(wel_resource, base_dir / "wel")
                rch_path = _download_ckan_resource(rch_resource, base_dir / "rch")
                gdf = _load_grid_resource(grid_resource, base_dir / "grid")
                wel = load_wel(wel_path)
                nrow = int(gdf["ROW"].max())
                ncol = int(gdf["COL"].max())
                try:
                    nper = 1
                    spd = getattr(wel, "stress_period_data", None)
                    if spd is not None and hasattr(spd, "data"):
                        keys = list(spd.data.keys())
                        if keys:
                            nper = int(max(keys)) + 1
                    rch = load_rch(rch_path, nrow=nrow, ncol=ncol, nper=nper)
                except Exception as exc:
                    rch = None
                    print(f"RCH load failed: {exc}")
                ui = build_ui(wel, gdf, show_dataset_controls=False, rch=rch)
                display(ui)
            except Exception as exc:
                print(f"Failed to load dataset: {exc}")

    load_btn.on_click(_load_selected)

    display(
        widgets.VBox(
            [
                widgets.VBox(
                    [
                        widgets.HBox([dataset_dropdown, load_btn]),
                        dataset_status,
                    ]
                ),
                output,
            ]
        )
    )
    _fetch_ckan(None)
