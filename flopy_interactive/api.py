"""Flask API for the React + Leaflet FloPy interactive UI."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

from flask import Flask, jsonify, request, send_from_directory

from flopy_interactive import actor_gateway, workflow_gateway
from flopy_interactive.ckankit import publish as ckanp
from flopy_interactive.ckankit.search import (
    resource_has_standard_var,
    search_ckan_datasets,
    search_ckan_datasets_wel_rch,
)
from flopy_interactive.config import CKAN_BASE_URL, GRID_STANDARD_VAR
from flopy_interactive.data.download import download_ckan_resource
from flopy_interactive.data.grid import load_grid_resource
from flopy_interactive.data.rch import (
    apply_rch_rate_update,
    build_rch_cells_for_periods,
    load_rch,
)
from flopy_interactive.data.wel import (
    apply_rate_update,
    build_cell_id_lookup,
    collect_wel_cells_for_period_data,
    get_wel_period_keys,
    load_wel,
)


DATA_DIR = Path(os.environ.get("FLOPY_DATA_DIR", "ckan_data"))
OUTPUT_WEL = Path(os.environ.get("FLOPY_OUTPUT_WEL", "barton_springs_updated.wel"))
SUGGEST_TITLE_FILTER = "Barton Springs Edwards Aquifer"
CKAN_URL = os.environ.get("FLOPY_CKAN_URL", CKAN_BASE_URL)
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "dist"
APPLY_ACTOR_ID = os.environ.get("FLOPY_ACTOR_APPLY_ID", "").strip()
WORKFLOW_GROUP_ID = os.environ.get("FLOPY_WORKFLOW_GROUP_ID", "").strip()
WORKFLOW_PIPELINE_ID = os.environ.get("FLOPY_WORKFLOW_PIPELINE_ID", "flopy-apply-flux-percent").strip()
WORKFLOW_ARCHIVE_IDS = [
    value.strip()
    for value in os.environ.get("FLOPY_WORKFLOW_ARCHIVE_IDS", "").split(",")
    if value.strip()
]

_DATASET_CACHE: Dict[str, Dict] = {}


def _finite_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def get_datasets() -> List[Dict]:
    return search_ckan_datasets()


def _get_dataset_or_none(name: str | None) -> Dict | None:
    if not name:
        return None
    for dataset in get_datasets():
        if dataset.get("name") == name:
            return dataset
    return None


def load_dataset(name: str) -> Dict:
    cached = _DATASET_CACHE.get(name)
    if cached is not None:
        return cached
    dataset = _get_dataset_or_none(name)
    if not dataset:
        raise ValueError(f"Dataset not found: {name}")
    base_dir = DATA_DIR / name
    wel_resource = dataset["matches"]["wel"][0]
    grid_resource = dataset["matches"]["grid"][0]
    rch_resource = dataset["matches"]["rch"][0]
    wel_path = download_ckan_resource(wel_resource, base_dir / "wel")
    rch_path = download_ckan_resource(rch_resource, base_dir / "rch")
    gdf = load_grid_resource(grid_resource, base_dir / "grid")
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
    except Exception:
        rch = None
    nlay = 1
    if hasattr(wel, "parent") and wel.parent is not None:
        nlay = int(getattr(wel.parent.dis, "nlay", 1))
    elif hasattr(wel, "model") and wel.model is not None:
        nlay = int(getattr(wel.model.dis, "nlay", 1))
    elif hasattr(wel, "_model") and wel._model is not None:
        nlay = int(getattr(wel._model.dis, "nlay", 1))
    data = {
        "dataset": dataset,
        "gdf": gdf,
        "wel": wel,
        "rch": rch,
        "cell_id_lookup": build_cell_id_lookup(gdf, wel),
        "nlay": nlay,
    }
    _DATASET_CACHE[name] = data
    return data


def _collect_wel_cells_for_periods(wel, gdf, periods: Sequence[int]) -> Dict[int, float]:
    cell_id_lookup = build_cell_id_lookup(gdf, wel)
    if not periods:
        spd = getattr(wel, "stress_period_data", None)
        periods = list(spd.data.keys()) if spd is not None and hasattr(spd, "data") else []

    def _merge_periods(row_offset: int, col_offset: int) -> Dict[int, float]:
        cells: Dict[int, float] = {}
        for period in periods:
            period_cells = collect_wel_cells_for_period_data(
                wel, cell_id_lookup, int(period), row_offset, col_offset
            )
            for cid, flux in period_cells.items():
                if cid not in cells or abs(flux) > abs(cells[cid]):
                    cells[cid] = flux
        return cells

    if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
        return _merge_periods(0, 0)
    base = _merge_periods(0, 0)
    offset = _merge_periods(1, 1)
    return offset if len(offset) > len(base) else base


def _build_leaflet_payload(
    loaded_dataset: str,
    flux_source: str,
    color_by: str,
    color_period: int | None,
    periods: Sequence[int],
) -> Dict:
    data = load_dataset(loaded_dataset)
    gdf = data["gdf"]
    wel = data["wel"]
    rch = data["rch"]
    map_periods = [color_period] if color_by == "flux" and color_period is not None else list(periods or [])
    wel_cells = _collect_wel_cells_for_periods(wel, gdf, map_periods)
    rch_cells = {}
    if rch is not None:
        try:
            rch_cells = build_rch_cells_for_periods(rch, gdf, map_periods)
        except Exception:
            rch_cells = {}
    active_cells = rch_cells if flux_source == "rch" else wel_cells
    period_keys = get_wel_period_keys(wel) or [0]
    features = []
    for _, row in gdf.iterrows():
        lat = _finite_float(row.get("_lat"))
        lon = _finite_float(row.get("_lon"))
        if lat is None or lon is None:
            continue
        cell_id = int(row["CELL_ID"])
        flux = float(active_cells.get(cell_id, 0.0))
        if flux == 0.0 and color_by == "flux":
            continue
        features.append(
            {
                "cellId": cell_id,
                "lat": lat,
                "lon": lon,
                "row": int(row["ROW"]),
                "col": int(row["COL"]),
                "flux": flux,
                "gcd": str(row.get("GCD_Name") or "Unknown"),
                "pgma": str(row.get("PGMA_Name") or "Unknown"),
            }
        )
    categories = {}
    for column in ("GCD_Name", "PGMA_Name"):
        if column in gdf.columns:
            values = gdf[column].fillna("Unknown").astype(str).drop_duplicates().tolist()
            categories[column] = [{"label": value, "value": value} for value in values]
    if features:
        center = {
            "lat": sum(feature["lat"] for feature in features) / len(features),
            "lon": sum(feature["lon"] for feature in features) / len(features),
            "zoom": 8,
        }
    else:
        center = {"lat": 30.2672, "lon": -97.7431, "zoom": 8}
    return {
        "dataset": {
            "name": data["dataset"]["name"],
            "title": data["dataset"]["title"],
            "sourceUrl": f"{CKAN_URL}/dataset/{data['dataset']['name']}",
            "hasRch": rch is not None,
            "nlay": data["nlay"],
        },
        "controls": {
            "periodOptions": [{"label": f"SP {idx + 1}", "value": idx} for idx in period_keys],
            "layerOptions": [{"label": str(layer), "value": layer} for layer in range(1, int(data["nlay"]) + 1)],
            "colorOptions": [
                {"label": "Flux", "value": "flux"},
                {"label": "GCD_Name", "value": "GCD_Name"},
                {"label": "PGMA_Name", "value": "PGMA_Name"},
            ],
            "categoryOptions": categories.get(color_by, []),
            "colorPeriodOptions": [{"label": f"SP {idx + 1}", "value": idx} for idx in period_keys],
        },
        "summary": {
            "activeCellCount": len([feature for feature in features if feature["flux"] != 0.0]),
            "fluxSource": flux_source,
        },
        "mapData": {
            "center": center,
            "cells": features,
        },
    }


def _dataset_options_without_grid(datasets: List[Dict]) -> List[Dict[str, str]]:
    filtered = []
    for dataset in datasets:
        name = dataset.get("name")
        if not name:
            continue
        title = str(dataset.get("title") or "").strip()
        if SUGGEST_TITLE_FILTER.lower() not in title.lower():
            continue
        if any(
            resource_has_standard_var(res, GRID_STANDARD_VAR)
            for res in dataset.get("matches", {}).get("grid", [])
        ):
            continue
        filtered.append({"label": dataset.get("title", name), "value": name})
    return filtered


def _owned_gam_datasets(username: str, jwt_token: str) -> List[Dict[str, str]]:
    if not username or not jwt_token:
        return []
    options: List[Dict[str, str]] = []
    for dataset in get_datasets():
        name = dataset.get("name")
        if not name:
            continue
        title = str(dataset.get("title") or "").strip()
        if SUGGEST_TITLE_FILTER.lower() not in title.lower():
            continue
        try:
            details = ckanp.package_show(jwt_token, name)
        except Exception:
            continue
        resources = details.get("resources", [])
        if any(resource_has_standard_var(res, GRID_STANDARD_VAR) for res in resources):
            continue
        maintainer = str(details.get("maintainer", "")).strip().lower()
        maintainer_email = str(details.get("maintainer_email", "")).strip().lower()
        author = str(details.get("author", "")).strip().lower()
        author_email = str(details.get("author_email", "")).strip().lower()
        target = username.strip().lower()
        if maintainer and (maintainer == target or maintainer_email == target):
            options.append({"label": details.get("title", name), "value": name})
            continue
        if author and (author == target or author_email == target):
            options.append({"label": details.get("title", name), "value": name})
    return options


def _suggest_gam_datasets(username: str, jwt_token: str) -> List[Dict[str, str]]:
    datasets = search_ckan_datasets_wel_rch(no_grid=True)
    if jwt_token:
        owned = _owned_gam_datasets(username, jwt_token)
        if owned:
            return owned
    return _dataset_options_without_grid(datasets)


def _parse_int_list(value: str | None) -> List[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _json_int_list(payload: Dict, key: str) -> List[int]:
    values = payload.get(key) or []
    return [int(value) for value in values]


def _resolve_workflow_group_id(payload: Dict | None = None) -> str:
    if payload:
        candidate = str(payload.get("workflowGroupId") or "").strip()
        if candidate:
            return candidate
    return WORKFLOW_GROUP_ID


def _apply_direct(payload: Dict) -> Dict:
    loaded_dataset = str(payload.get("dataset") or "").strip()
    if not loaded_dataset:
        raise ValueError("No dataset selected.")
    selected_ids = _json_int_list(payload, "selectedIds")
    if not selected_ids:
        raise ValueError("No cells selected.")
    data = load_dataset(loaded_dataset)
    gdf = data["gdf"]
    wel = data["wel"]
    flux_source = str(payload.get("fluxSource") or "wel")
    new_rate = float(payload.get("newRate") or 0.0)
    rate_mode = str(payload.get("rateMode") or "set")
    layers = _json_int_list(payload, "layers")
    periods = _json_int_list(payload, "periods")
    jwt_token = str(payload.get("jwtToken") or "").strip()
    dataset_name = str(payload.get("datasetName") or "").strip()
    dataset_title = str(payload.get("datasetTitle") or "").strip()
    output_name = str(payload.get("outputName") or "").strip()
    source_url = str(payload.get("sourceUrl") or "").strip()
    change_summary = str(payload.get("changeSummary") or "").strip()
    tapis_username = str(payload.get("tapisUsername") or "").strip()
    add_missing = bool(payload.get("addMissing"))
    output_path = Path(output_name or OUTPUT_WEL)
    target_ext = ".rch" if flux_source == "rch" else ".wel"
    if output_path.suffix.lower() != target_ext:
        output_path = output_path.with_suffix(target_ext)

    if flux_source == "rch":
        rch = data["rch"]
        if rch is None:
            raise ValueError("RCH data not loaded.")
        updated = apply_rch_rate_update(rch, gdf, selected_ids, new_rate, rate_mode, periods, output_path)
    else:
        updated = apply_rate_update(
            wel,
            gdf,
            selected_ids,
            new_rate,
            rate_mode,
            add_missing,
            layers,
            periods,
            output_path,
        )

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

    try:
        if flux_source == "rch":
            result = ckanp.publish_updated_rch(
                loaded_dataset,
                output_path,
                provenance,
                jwt_token=jwt_token or None,
                new_dataset_name=dataset_name or None,
                new_dataset_title=dataset_title or None,
                source_url=source_url or None,
                change_summary=change_summary or None,
                maintainer_username=tapis_username or None,
            )
        else:
            result = ckanp.publish_updated_wel(
                loaded_dataset,
                output_path,
                provenance,
                jwt_token=jwt_token or None,
                new_dataset_name=dataset_name or None,
                new_dataset_title=dataset_title or None,
                source_url=source_url or None,
                change_summary=change_summary or None,
                maintainer_username=tapis_username or None,
            )
        dataset_id = result["dataset"].get("id", "")
        message = f"Updated {updated} cells. Wrote {output_path}. Published CKAN dataset {dataset_id}."
        return {"mode": "direct", "message": message, "updated": updated, "datasetId": dataset_id}
    except Exception as exc:
        return {
            "mode": "direct",
            "message": f"Updated {updated} cells. Wrote {output_path}. CKAN publish failed: {exc}",
            "updated": updated,
        }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")

    @app.get("/api/datasets")
    def list_datasets():
        datasets = [{"label": d["title"], "value": d["name"]} for d in get_datasets()]
        return jsonify({"datasets": datasets})

    @app.get("/api/datasets/<dataset_name>/view")
    def dataset_view(dataset_name: str):
        flux_source = request.args.get("fluxSource", "wel")
        color_by = request.args.get("colorBy", "flux")
        color_period = request.args.get("colorPeriod")
        payload = _build_leaflet_payload(
            dataset_name,
            flux_source,
            color_by,
            int(color_period) if color_period not in (None, "") else None,
            _parse_int_list(request.args.get("periods")),
        )
        return jsonify(payload)

    @app.post("/api/datasets/<dataset_name>/category-selection")
    def category_selection(dataset_name: str):
        payload = request.get_json(force=True) or {}
        color_by = str(payload.get("colorBy") or "")
        category_value = str(payload.get("categoryValue") or "")
        data = load_dataset(dataset_name)
        gdf = data["gdf"]
        if not category_value or color_by not in ("GCD_Name", "PGMA_Name") or color_by not in gdf.columns:
            return jsonify({"selectedIds": []})
        matches = gdf[color_by].fillna("Unknown").astype(str) == category_value
        selected_ids = sorted(int(cid) for cid in gdf.loc[matches, "CELL_ID"].tolist())
        return jsonify({"selectedIds": selected_ids})

    @app.post("/api/login")
    def login():
        payload = request.get_json(force=True) or {}
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "").strip()
        if not username or not password:
            return jsonify({"error": "Missing username or password."}), 400
        try:
            token = ckanp.get_tapis_token(username, password)
        except Exception as exc:
            return jsonify({"error": f"Login failed: {exc}"}), 400
        return jsonify({"jwtToken": token, "username": username})

    @app.get("/api/dataset-suggestions")
    def dataset_suggestions():
        username = str(request.args.get("username") or "")
        jwt_token = str(request.args.get("jwtToken") or "")
        options = [{"label": "New dataset", "value": "__new__"}] + _suggest_gam_datasets(username, jwt_token)
        return jsonify({"options": options})

    @app.post("/api/workflow/register")
    def register_workflow():
        payload = request.get_json(force=True) or {}
        jwt_token = str(payload.get("jwtToken") or "").strip()
        workflow_group_id = _resolve_workflow_group_id(payload)
        if not jwt_token:
            return jsonify({"error": "Missing jwtToken."}), 400
        if not workflow_group_id:
            return jsonify({"error": "Missing workflowGroupId."}), 400
        try:
            workflow_gateway.ensure_pipeline(
                workflow_group_id,
                WORKFLOW_PIPELINE_ID,
                jwt_token,
                archive_ids=WORKFLOW_ARCHIVE_IDS,
            )
        except Exception as exc:
            return jsonify({"error": f"Failed to register workflow pipeline: {exc}"}), 400
        return jsonify(
            {
                "groupId": workflow_group_id,
                "pipelineId": WORKFLOW_PIPELINE_ID,
                "message": f"Workflow pipeline {WORKFLOW_PIPELINE_ID} is ready in group {workflow_group_id}.",
            }
        )

    @app.post("/api/apply")
    def apply():
        payload = request.get_json(force=True) or {}
        jwt_token = str(payload.get("jwtToken") or "").strip()
        workflow_group_id = _resolve_workflow_group_id(payload)
        if workflow_group_id and jwt_token:
            try:
                workflow_gateway.ensure_pipeline(
                    workflow_group_id,
                    WORKFLOW_PIPELINE_ID,
                    jwt_token,
                    archive_ids=WORKFLOW_ARCHIVE_IDS,
                )
                submission = workflow_gateway.run_pipeline(
                    workflow_group_id,
                    WORKFLOW_PIPELINE_ID,
                    jwt_token,
                    args={
                        "apply_payload_json": {
                            "type": "string",
                            "value": json.dumps(payload),
                        },
                        "ckan_url": {
                            "type": "string",
                            "value": CKAN_URL,
                        },
                        "jwt_token": {
                            "type": "string",
                            "value": jwt_token,
                        },
                    },
                )
                run_id = workflow_gateway.extract_run_id(submission)
            except Exception as exc:
                return jsonify({"error": f"Failed to submit workflow run: {exc}"}), 400
            return jsonify(
                {
                    "mode": "workflow",
                    "message": f"Submitted workflow run {run_id}.",
                    "groupId": workflow_group_id,
                    "pipelineId": WORKFLOW_PIPELINE_ID,
                    "runId": run_id,
                }
            ), 202
        if APPLY_ACTOR_ID and jwt_token:
            actor_payload = dict(payload)
            actor_payload["operation"] = "apply_update_publish"
            submission = actor_gateway.submit_execution(APPLY_ACTOR_ID, jwt_token, actor_payload)
            return jsonify(
                {
                    "mode": "actor",
                    "message": f"Submitted ETL actor execution {submission['executionId']}.",
                    "actorId": submission["actorId"],
                    "executionId": submission["executionId"],
                }
            ), 202
        try:
            return jsonify(_apply_direct(payload))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/actor-executions/<actor_id>/<execution_id>")
    def actor_execution(actor_id: str, execution_id: str):
        token = str(request.args.get("jwtToken") or "").strip()
        if not token:
            return jsonify({"error": "Missing jwtToken."}), 400
        try:
            execution = actor_gateway.get_execution(actor_id, execution_id, token)
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch actor execution: {exc}"}), 400
        status = str(execution.get("status") or "")
        result = None
        logs = None
        if status == "COMPLETE":
            try:
                result = actor_gateway.get_result(actor_id, execution_id, token)
            except Exception:
                result = None
            try:
                logs = actor_gateway.get_logs(actor_id, execution_id, token)
            except Exception:
                logs = None
        return jsonify({"status": status, "execution": execution, "result": result, "logs": logs})

    @app.get("/api/workflow-runs/<group_id>/<pipeline_id>/<run_id>")
    def workflow_run(group_id: str, pipeline_id: str, run_id: str):
        token = str(request.args.get("jwtToken") or "").strip()
        if not token:
            return jsonify({"error": "Missing jwtToken."}), 400
        try:
            run_payload = workflow_gateway.get_pipeline_run(group_id, pipeline_id, run_id, token)
        except Exception as exc:
            return jsonify({"error": f"Failed to fetch workflow run: {exc}"}), 400
        status = workflow_gateway.extract_run_status(run_payload)
        result = workflow_gateway.extract_result_payload(run_payload)
        return jsonify({"status": status, "run": run_payload, "result": result})

    @app.get("/api/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.get("/")
    def serve_root():
        if FRONTEND_DIST.exists():
            return send_from_directory(FRONTEND_DIST, "index.html")
        return jsonify({"message": "Frontend build not found. Run `npm install` and `npm run build`."})

    @app.get("/<path:path>")
    def serve_frontend(path: str):
        if FRONTEND_DIST.exists() and (FRONTEND_DIST / path).exists():
            return send_from_directory(FRONTEND_DIST, path)
        if FRONTEND_DIST.exists():
            return send_from_directory(FRONTEND_DIST, "index.html")
        return jsonify({"error": f"Unknown path: {path}"}), 404

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
