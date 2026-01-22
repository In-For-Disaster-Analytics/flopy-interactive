#!/usr/bin/env python3
"""Dash dashboard for WEL/RCH visualization and updates."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
import re
import hashlib
import json
from typing import Dict, Iterable, List, Sequence

import dash
from dash import Input, Output, State, dcc, html, ctx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import ckan_publish as ckanp
from flopy_interactive.ckankit.search import (
    resource_has_standard_var,
    search_ckan_datasets,
    search_ckan_datasets_wel_rch,
)
from flopy_interactive.config import CKAN_BASE_URL, GRID_STANDARD_VAR
from flopy_interactive.data.download import download_ckan_resource
from flopy_interactive.data.grid import load_grid_resource
from flopy_interactive.data.rch import apply_rch_rate_update, build_rch_cells_for_periods, load_rch
from flopy_interactive.data.wel import (
    apply_rate_update,
    build_cell_id_lookup,
    collect_wel_cells_for_period_data,
    get_wel_period_keys,
    load_wel,
)
from flopy_interactive.utils.perf import perf_call, perf_note, perf_timer
from flopy_interactive.viz.color_modes import apply_color_mode


DATA_DIR = Path(os.environ.get("FLOPY_DATA_DIR", "ckan_data"))
OUTPUT_WEL = Path(os.environ.get("FLOPY_OUTPUT_WEL", "barton_springs_updated.wel"))
SUGGEST_TITLE_FILTER = "Barton Springs Edwards Aquifer"
CKAN_URL = os.environ.get("FLOPY_CKAN_URL", CKAN_BASE_URL)

_DATASET_CACHE: Dict[str, Dict] = {}
_MAP_FIG_CACHE: Dict[tuple, go.Figure] = {}


def get_datasets() -> List[Dict]:
    """Fetch CKAN datasets for the app session.

    Args:
        None.

    Returns:
        List of dataset metadata dicts.
    """
    return perf_call("search_ckan_datasets", search_ckan_datasets)


def _get_dataset_or_none(name: str | None) -> Dict | None:
    """Return dataset metadata matching a name, if present.

    Args:
        name: Dataset name string.

    Returns:
        Dataset metadata dict or None.
    """
    if not name:
        return None
    datasets = get_datasets()
    for dataset in datasets:
        if dataset.get("name") == name:
            return dataset
    return None


def load_dataset(name: str) -> Dict:
    """Download CKAN resources and assemble WEL/RCH/grid inputs.

    Args:
        name: CKAN dataset name.

    Returns:
        Dict with dataset, gdf, wel, rch, lookup, and nlay.
    """
    cached = _DATASET_CACHE.get(name)
    if cached is not None:
        perf_note(f"load_dataset cache hit: {name}")
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
    gdf = perf_call(f"load_grid_resource:{name}", load_grid_resource, grid_resource, base_dir / "grid")
    wel = perf_call(f"load_wel:{name}", load_wel, wel_path)
    nrow = int(gdf["ROW"].max())
    ncol = int(gdf["COL"].max())
    try:
        nper = 1
        spd = getattr(wel, "stress_period_data", None)
        if spd is not None and hasattr(spd, "data"):
            keys = list(spd.data.keys())
            if keys:
                nper = int(max(keys)) + 1
        rch = perf_call(f"load_rch:{name}", load_rch, rch_path, nrow=nrow, ncol=ncol, nper=nper)
    except Exception:
        rch = None
    cell_id_lookup = perf_call(f"build_cell_id_lookup:{name}", build_cell_id_lookup, gdf, wel)
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
        "cell_id_lookup": cell_id_lookup,
        "nlay": nlay,
    }
    _DATASET_CACHE[name] = data
    return data


def _collect_wel_cells_for_periods(
    wel, gdf, periods: Sequence[int]
) -> Dict[int, float]:
    """Aggregate WEL flux by cell across selected periods.

    Args:
        wel: FloPy WEL package.
        gdf: GeoDataFrame with ROW/COL/CELL_ID columns.
        periods: Stress period indices to include.

    Returns:
        Mapping of CELL_ID to flux value.
    """
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


def _build_map_figure(
    gdf,
    cells: Dict[int, float],
    flux_label: str,
    color_by: str,
    force_linear: bool,
    selected_ids: Iterable[int],
    zoom: float | None = None,
    show_grid: bool = False,
) -> go.Figure:
    """Build the Plotly map figure with grid and selection overlays.

    Args:
        gdf: GeoDataFrame with grid metadata.
        cells: Mapping of CELL_ID to flux values.
        flux_label: Label for the colorbar.
        color_by: ``flux`` or category column name.
        force_linear: Whether to force linear scaling.
        selected_ids: Iterable of selected CELL_ID values.
        zoom: Optional zoom level for downsampling.
        show_grid: Whether to render the grid choropleth layer.

    Returns:
        Plotly Figure for the map.
    """
    gdf_map = gdf[["CELL_ID", "ROW", "COL", "geometry"]].copy()
    gdf_valid = gdf_map[gdf_map["geometry"].notna()].copy()
    with perf_timer("downsample_for_choropleth"):
        gdf_choro = _downsample_for_choropleth(gdf_valid, gdf, zoom)
    with perf_timer("build_geojson"):
        grid_geojson = gdf_choro.set_index("CELL_ID").__geo_interface__
    center_lat = float(gdf["_lat"].median())
    center_lon = float(gdf["_lon"].median())

    fig = go.Figure()
    z_values = [float(cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
    z_values_choro = [float(cells.get(int(cid), 0.0)) for cid in gdf_choro["CELL_ID"]]
    z_values_valid = [float(cells.get(int(cid), 0.0)) for cid in gdf_valid["CELL_ID"]]
    gcd_values = (
        gdf["GCD_Name"].fillna("Unknown").astype(str)
        if "GCD_Name" in gdf.columns
        else pd.Series(["Unknown"] * len(gdf), index=gdf.index)
    )
    pgma_values = (
        gdf["PGMA_Name"].fillna("Unknown").astype(str)
        if "PGMA_Name" in gdf.columns
        else pd.Series(["Unknown"] * len(gdf), index=gdf.index)
    )
    if show_grid:
        fig.add_trace(
            go.Choroplethmap(
                geojson=grid_geojson,
                locations=gdf_choro["CELL_ID"],
                z=z_values_choro,
                colorscale=[(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")],
                marker_opacity=0.35,
                marker_line_width=0.5,
                marker_line_color="#666",
                showscale=False,
                name="Grid",
                customdata=np.stack(
                    [
                        gdf_choro["CELL_ID"],
                        z_values_choro,
                        gcd_values.loc[gdf_choro.index],
                        pgma_values.loc[gdf_choro.index],
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
    active_ids = {int(cid) for cid, flux in cells.items() if float(flux) != 0.0}
    if active_ids:
        gdf_scatter = gdf[gdf["CELL_ID"].isin(active_ids)].copy()
    else:
        gdf_scatter = gdf.copy()
    gdf_scatter = gdf_scatter[gdf_scatter["_lon"].notna() & gdf_scatter["_lat"].notna()].copy()
    fig.add_trace(
        go.Scattermap(
            lon=gdf_scatter["_lon"],
            lat=gdf_scatter["_lat"],
            mode="markers",
            marker=dict(
                size=6,
                color=z_values,
                colorscale=[(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")],
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(
                    title=flux_label,
                    tickvals=None,
                    ticktext=None,
                    orientation="v",
                    x=1.02,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    len=0.8,
                ),
            ),
            name="Cells",
            customdata=np.stack(
                [
                    gdf_scatter["CELL_ID"],
                    gdf_scatter["ROW"],
                    gdf_scatter["COL"],
                    [float(cells.get(int(cid), 0.0)) for cid in gdf_scatter["CELL_ID"]],
                    gcd_values.loc[gdf_scatter.index],
                    pgma_values.loc[gdf_scatter.index],
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

    color_gdf = gdf if show_grid else gdf_scatter
    with perf_timer("apply_color_mode"):
        apply_color_mode(
            fig,
            color_gdf,
            cells,
            color_by,
            normalize=False,
            flux_label=flux_label,
            force_linear=force_linear,
        )

    selected_ids = {int(cid) for cid in selected_ids}
    if selected_ids and len(fig.data) > 2:
        selected_rows = gdf[gdf["CELL_ID"].isin(selected_ids)]
        fig.data[2].update(lon=selected_rows["_lon"], lat=selected_rows["_lat"])

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="lasso",
        clickmode="event+select",
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8,
        ),
    )
    return fig


def _apply_selection(fig: go.Figure, gdf, selected_ids: Iterable[int]) -> None:
    """Apply selected cell overlay to an existing figure."""
    if not fig.data or len(fig.data) <= 2:
        return
    selected_set = {int(cid) for cid in (selected_ids or [])}
    if not selected_set:
        fig.data[2].update(lon=[], lat=[])
        return
    selected_rows = gdf[gdf["CELL_ID"].isin(selected_set)]
    fig.data[2].update(lon=selected_rows["_lon"], lat=selected_rows["_lat"])


def _dataset_options() -> List[Dict[str, str]]:
    """Build dataset dropdown options from CKAN search.

    Args:
        None.

    Returns:
        List of dicts with ``label`` and ``value`` keys.
    """
    try:
        datasets = get_datasets()
    except Exception:
        return []
    return [{"label": d["title"], "value": d["name"]} for d in datasets]


def _dataset_options_without_grid(datasets: List[Dict]) -> List[Dict[str, str]]:
    """Build dataset options excluding datasets that include grid resources."""
    filtered = []
    for dataset in datasets:
        name = dataset.get("name")
        if not name:
            continue
        title = str(dataset.get("title") or "").strip()
        if SUGGEST_TITLE_FILTER.lower() not in title.lower():
            continue
        if any(resource_has_standard_var(res, GRID_STANDARD_VAR) for res in dataset.get("matches", {}).get("grid", [])):
            continue
        filtered.append({"label": dataset.get("title", name), "value": name})
    return filtered


def _summarize_periods(periods: List[int], total: int | None) -> str:
    """Return a compact stress-period summary string."""
    if not periods:
        return f"All periods ({total})" if total else "All periods"
    unique = sorted({int(p) for p in periods})
    if total and len(unique) >= total:
        return f"All periods ({total})"
    if total and total > 0 and len(unique) / total >= 0.7:
        return f"Periods: {len(unique)}/{total}"
    if len(unique) > 1 and unique[-1] - unique[0] + 1 == len(unique):
        return f"Periods: {unique[0] + 1}-{unique[-1] + 1}"
    if len(unique) <= 5:
        return "Periods: " + ", ".join(str(p + 1) for p in unique)
    return "Periods: " + ", ".join(str(p + 1) for p in unique[:3]) + f" (+{len(unique) - 3} more)"


def _downsample_for_choropleth(gdf_valid, gdf_full, zoom: float | None) -> pd.DataFrame:
    """Downsample grid polygons for choropleth rendering based on zoom."""
    if zoom is None:
        bin_size = 0.5
    elif zoom < 6:
        bin_size = 0.5
    elif zoom < 7:
        bin_size = 0.25
    elif zoom < 8:
        bin_size = 0.125
    elif zoom < 9:
        bin_size = 0.06
    elif zoom < 10:
        bin_size = 0.03
    elif zoom < 11:
        bin_size = 0.015
    else:
        return gdf_valid
    if "_lon" not in gdf_full.columns or "_lat" not in gdf_full.columns:
        return gdf_valid
    gdf_bins = gdf_full.loc[gdf_valid.index, ["_lon", "_lat"]].copy()
    lon_bins = np.floor(gdf_bins["_lon"] / bin_size)
    lat_bins = np.floor(gdf_bins["_lat"] / bin_size)
    gdf_bins["_bin_key"] = list(zip(lon_bins, lat_bins))
    return gdf_valid.loc[gdf_bins.drop_duplicates("_bin_key").index]


def _owned_gam_datasets(username: str, jwt_token: str) -> List[Dict[str, str]]:
    """List datasets owned by a user for suggestion dropdowns.

    Args:
        username: Tapis username or email.
        jwt_token: CKAN JWT token.

    Returns:
        List of dicts with ``label`` and ``value`` keys.
    """
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
    """Return owned datasets when available, else all matched GAM datasets."""
    datasets = search_ckan_datasets_wel_rch(no_grid=True)
    if jwt_token:
        owned = _owned_gam_datasets(username, jwt_token)
        if owned:
            return owned
    return _dataset_options_without_grid(datasets)

def _slugify(value: str) -> str:
    """Normalize a string for use in dataset naming.

    Args:
        value: Input string to normalize.

    Returns:
        Slugified string.
    """
    value = value.strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-_.]", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "dataset"


app = dash.Dash(__name__)
server = app.server

dataset_options = _dataset_options()
default_dataset = dataset_options[0]["value"] if dataset_options else None

app.layout = html.Div(
    [
        dcc.Store(id="selected-store", data=[]),
        dcc.Store(id="update-store", data=0),
        dcc.Store(id="loaded-dataset", data=default_dataset),
        dcc.Store(id="load-counter", data=0),
        dcc.Store(id="ckan-jwt", data=os.environ.get("FLOPY_CKAN_JWT", "").strip()),
        dcc.Store(id="login-message", data=""),
        dcc.Store(id="tapis-username", data=""),
        dcc.Store(id="name-seed", data=str(uuid.uuid4())),
        dcc.Store(id="last-loaded-dataset", data=""),
        dcc.Store(id="selection-warning", data=False),
        dcc.Store(id="category-warning", data=False),
        html.Div(
            [
                html.Div("FloPy WEL/RCH Dashboard", className="header-title"),
                html.Form(
                    [
                        html.Div("CKAN Login", className="login-title"),
                        dcc.Input(
                            id="login-username",
                            type="text",
                            placeholder="Tapis username",
                            className="login-input",
                        ),
                        dcc.Input(
                            id="login-password",
                            type="password",
                            placeholder="Tapis password",
                            className="login-input",
                        ),
                        html.Button(
                            "Login",
                            id="login-submit",
                            n_clicks=0,
                            type="button",
                        ),
                        html.Div(id="login-status", className="status"),
                    ],
                    className="login-card",
                ),
            ],
            className="header-bar",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Dataset"),
                        dcc.Dropdown(
                            id="dataset",
                            options=dataset_options,
                            value=default_dataset,
                            clearable=False,
                        ),
                        html.Button("Load dataset", id="load-dataset", n_clicks=0),
                        html.Div(id="dataset-status", className="status"),
                        html.Label("Flux source"),
                        dcc.Dropdown(
                            id="flux-source",
                            options=[
                                {"label": "Well", "value": "wel"},
                                {"label": "Recharge", "value": "rch"},
                            ],
                            value="wel",
                            clearable=False,
                        ),
                        html.Label("Color by"),
                        dcc.Dropdown(
                            id="color-by",
                            options=[
                                {"label": "Flux", "value": "flux"},
                                {"label": "GCD_Name", "value": "GCD_Name"},
                                {"label": "PGMA_Name", "value": "PGMA_Name"},
                            ],
                            value="flux",
                            clearable=False,
                        ),
                        html.Div(
                            [
                                html.Label("Category"),
                                dcc.Dropdown(
                                    id="category-select",
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    className="dropdown-scroll",
                                ),
                                html.Button(
                                    "Select category",
                                    id="select-category",
                                    n_clicks=0,
                                ),
                            ],
                            id="category-wrap",
                            className="category-wrap",
                            style={"display": "none"},
                        ),
                        html.Div(
                            [
                                html.Label("Color period (flux)"),
                                dcc.Dropdown(
                                    id="color-period",
                                    options=[],
                                    value=None,
                                    clearable=True,
                                ),
                            ],
                            id="color-period-wrap",
                        ),
        html.Hr(),
        html.H4("Edit selection"),
        html.Label("Stress periods"),
        dcc.Dropdown(
            id="periods",
            options=[],
            value=[],
            multi=True,
            className="dropdown-scroll",
        ),
        html.Div(id="periods-summary", className="status"),
        html.Button("Select all periods", id="select-all-periods", n_clicks=0),
        html.Hr(),
        html.Label("Rate mode"),
                        dcc.Dropdown(
                            id="rate-mode",
                            options=[
                                {"label": "Set", "value": "set"},
                                {"label": "Scale (%)", "value": "scale_percent"},
                            ],
                            value="set",
                            clearable=False,
                        ),
                        html.Label("New rate"),
                        dcc.Input(id="new-rate", type="number", value=-20.0, step=0.1),
                        html.Label("Layer (k)"),
                        dcc.Dropdown(
                            id="layers",
                            options=[],
                            value=[1],
                            multi=True,
                            className="dropdown-scroll",
                        ),
                        html.Button("Select all layers", id="select-all-layers", n_clicks=0),
                        dcc.Checklist(
                            id="add-missing",
                            options=[{"label": "Add missing wells", "value": "yes"}],
                            value=[],
                        ),
                        html.Label("Dataset title"),
                        dcc.Input(
                            id="dataset-title",
                            type="text",
                            placeholder="Optional display title",
                            className="login-input",
                        ),
                        html.Label("New dataset name"),
                        dcc.Dropdown(
                            id="dataset-suggestions",
                            options=[],
                            placeholder="Suggest existing GAM datasets",
                            clearable=True,
                        ),
                        dcc.Input(
                            id="dataset-name",
                            type="text",
                            placeholder="auto-generated",
                            className="login-input",
                        ),
                        html.Label(id="output-label", children="Output WEL filename"),
                        dcc.Input(
                            id="output-wel",
                            type="text",
                            value=str(OUTPUT_WEL),
                            className="login-input",
                        ),
                        html.Label("Change summary"),
                        dcc.Input(
                            id="change-summary",
                            type="text",
                            placeholder="Selection by Category x = y",
                            className="login-input",
                        ),
                        html.Label("Source URL"),
                        dcc.Input(
                            id="source-url",
                            type="text",
                            placeholder="https://...",
                            className="login-input",
                        ),
                        html.Button("Apply + Save", id="apply-rate", n_clicks=0),
                        html.Button(
                            "Clear selection",
                            id="clear-selection",
                            n_clicks=0,
                            className="secondary",
                        ),
                        html.Div(
                            id="selection-status",
                            className="status",
                            children="Selected cells: 0",
                        ),
                        html.Div(id="status-message", className="status"),
                    ],
                    className="panel",
                ),
                html.Div([dcc.Graph(id="map", figure=go.Figure())], className="map-wrap"),
            ],
            className="app-shell",
        ),
    ],
    className="app-root",
)


@app.callback(
    Output("periods", "options"),
    Output("layers", "options"),
    Input("loaded-dataset", "data"),
)
def update_dataset_controls(loaded_dataset: str | None):
    """Populate stress period and layer controls for the dataset.

    Args:
        loaded_dataset: CKAN dataset name or None.

    Returns:
        Tuple of (period options, layer options).
    """
    if not loaded_dataset:
        return [], []
    data = load_dataset(loaded_dataset)
    wel = data["wel"]
    period_keys = get_wel_period_keys(wel) or [0]
    period_options = [{"label": f"SP {idx + 1}", "value": idx} for idx in period_keys]
    nlay = data["nlay"]
    layer_options = [{"label": str(layer), "value": layer} for layer in range(1, nlay + 1)]
    return period_options, layer_options


@app.callback(
    Output("color-period-wrap", "style"),
    Output("color-period", "options"),
    Output("color-period", "value"),
    Input("loaded-dataset", "data"),
    Input("color-by", "value"),
    State("color-period", "value"),
)
def update_color_period(loaded_dataset, color_by, current_value):
    """Populate color-period dropdown from dataset stress periods."""
    if not loaded_dataset:
        return {"display": "none"}, [], None
    if color_by != "flux":
        return {"display": "none"}, [], None
    data = load_dataset(loaded_dataset)
    wel = data["wel"]
    spd_keys = get_wel_period_keys(wel) or [0]
    options = [{"label": f"SP {idx + 1}", "value": idx} for idx in spd_keys]
    if current_value in spd_keys:
        value = current_value
    else:
        value = spd_keys[0] if spd_keys else None
    return {"display": "block"}, options, value


@app.callback(
    Output("loaded-dataset", "data"),
    Output("dataset-status", "children"),
    Output("load-counter", "data"),
    Output("last-loaded-dataset", "data"),
    Input("load-dataset", "n_clicks"),
    State("dataset", "value"),
    State("load-counter", "data"),
    prevent_initial_call=False,
)
def load_selected_dataset(n_clicks, selected_dataset, load_counter):
    """Load dataset resources and update load state messages.

    Args:
        n_clicks: Click count from the load button.
        selected_dataset: Selected dataset name.
        load_counter: Current load counter value.

    Returns:
        Tuple of (loaded dataset, status message, updated counter).
    """
    if not selected_dataset:
        return None, "No dataset selected.", load_counter, ""
    try:
        load_dataset(selected_dataset)
    except Exception as exc:
        return selected_dataset, f"Failed to load dataset: {exc}", load_counter, selected_dataset
    load_counter = (load_counter or 0) + 1
    return selected_dataset, f"Loaded dataset: {selected_dataset}", load_counter, selected_dataset


@app.callback(
    Output("periods", "value"),
    Output("layers", "value"),
    Input("loaded-dataset", "data"),
    Input("select-all-periods", "n_clicks"),
    Input("select-all-layers", "n_clicks"),
    State("periods", "options"),
    State("layers", "options"),
    prevent_initial_call=False,
)
def update_periods_layers(
    loaded_dataset, select_periods_clicks, select_layers_clicks, period_options, layer_options
):
    """Handle select-all interactions for periods and layers.

    Args:
        loaded_dataset: CKAN dataset name or None.
        select_periods_clicks: Click count for select-all periods.
        select_layers_clicks: Click count for select-all layers.
        period_options: Existing period option list.
        layer_options: Existing layer option list.

    Returns:
        Tuple of (period values, layer values).
    """
    if not loaded_dataset:
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered_id
    if triggered == "select-all-periods":
        period_values = [opt["value"] for opt in (period_options or [])]
        layer_values = dash.no_update
    elif triggered == "select-all-layers":
        period_values = dash.no_update
        layer_values = [opt["value"] for opt in (layer_options or [])]
    elif triggered == "loaded-dataset":
        if not period_options or not layer_options:
            data = load_dataset(loaded_dataset)
            wel = data["wel"]
            period_keys = get_wel_period_keys(wel) or [0]
            period_options = [
                {"label": f"SP {idx + 1}", "value": idx} for idx in period_keys
            ]
            nlay = data["nlay"]
            layer_options = [
                {"label": str(layer), "value": layer} for layer in range(1, nlay + 1)
            ]
        period_values = [period_options[0]["value"]] if period_options else []
        layer_values = [1] if layer_options else []
    else:
        raise dash.exceptions.PreventUpdate
    return period_values, layer_values


@app.callback(
    Output("ckan-jwt", "data"),
    Output("login-status", "children"),
    Output("login-status", "className"),
    Output("tapis-username", "data"),
    Output("login-username", "disabled"),
    Output("login-password", "disabled"),
    Output("login-submit", "children"),
    Input("login-submit", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def login_ckan(n_clicks, username, password):
    """Authenticate with Tapis and store the CKAN JWT.

    Args:
        n_clicks: Click count from the login button.
        username: Tapis username string.
        password: Tapis password string.

    Returns:
        Tuple of (jwt token, status text, status class, username, disable flags, button label).
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not username or not password:
        return "", "Missing username or password.", "status", "", False, False, "Login"
    try:
        token = ckanp.get_tapis_token(username, password)
    except Exception as exc:
        return "", f"Login failed: {exc}", "status", "", False, False, "Login"
    return token, f"Logged in as {username}", "status status-ok", username, True, True, "Logged in"


@app.callback(
    Output("apply-rate", "disabled"),
    Input("ckan-jwt", "data"),
)
def toggle_apply_rate(jwt_token):
    """Enable Apply + Save only when authenticated."""
    return not bool(jwt_token)


@app.callback(
    Output("category-select", "options"),
    Output("category-select", "value"),
    Output("category-wrap", "style"),
    Output("select-category", "disabled"),
    Input("color-by", "value"),
    Input("loaded-dataset", "data"),
)
def update_category_controls(color_by, loaded_dataset):
    """Show or hide category selection controls by color mode.

    Args:
        color_by: Current color-by mode.
        loaded_dataset: CKAN dataset name or None.

    Returns:
        Tuple of (options, value, style, disabled flag).
    """
    if not loaded_dataset or color_by == "flux":
        return [], None, {"display": "none"}, True
    data = load_dataset(loaded_dataset)
    gdf = data["gdf"]
    if color_by not in ("GCD_Name", "PGMA_Name") or color_by not in gdf.columns:
        return [], None, {"display": "none"}, True
    categories = (
        gdf[color_by].fillna("Unknown").astype(str).drop_duplicates().tolist()
    )
    options = [{"label": value, "value": value} for value in categories]
    return options, None, {"display": "block"}, False


@app.callback(
    Output("periods-summary", "children"),
    Input("periods", "value"),
    Input("loaded-dataset", "data"),
)
def update_periods_summary(periods, loaded_dataset):
    """Show the active stress period selection summary."""
    if not loaded_dataset:
        return "Stress periods: none"
    if not periods:
        return "Stress periods: all"
    values = sorted(int(p) for p in periods)
    labels = ", ".join(f"SP {p + 1}" for p in values)
    return f"Stress periods: {labels}"


@app.callback(
    Output("dataset-suggestions", "options"),
    Input("tapis-username", "data"),
    Input("ckan-jwt", "data"),
)
def update_dataset_suggestions(username, jwt_token):
    """Return dataset suggestions owned by the logged-in user.

    Args:
        username: Tapis username or email.
        jwt_token: CKAN JWT token.

    Returns:
        List of dataset suggestion dicts.
    """
    options = _suggest_gam_datasets(username or "", jwt_token or "")
    return [{"label": "New dataset", "value": "__new__"}] + options


@app.callback(
    Output("dataset-name", "value"),
    Output("output-wel", "value"),
    Output("source-url", "value"),
    Output("change-summary", "value"),
    Output("dataset-title", "value"),
    Input("loaded-dataset", "data"),
    Input("flux-source", "value"),
    Input("rate-mode", "value"),
    Input("new-rate", "value"),
    Input("dataset-suggestions", "value"),
    Input("ckan-jwt", "data"),
    Input("color-by", "value"),
    Input("category-select", "value"),
    Input("periods", "value"),
    Input("layers", "value"),
    Input("add-missing", "value"),
    Input("selected-store", "data"),
    Input("load-counter", "data"),
    State("name-seed", "data"),
    State("last-loaded-dataset", "data"),
    State("periods", "options"),
    State("dataset-name", "value"),
    State("output-wel", "value"),
    State("source-url", "value"),
    State("change-summary", "value"),
    State("dataset-title", "value"),
)
def suggest_names(
    loaded_dataset,
    flux_source,
    rate_mode,
    new_rate,
    suggested_name,
    jwt_token,
    color_by,
    category_value,
    periods,
    layers,
    add_missing,
    selected_ids,
    _load_counter,
    name_seed,
    last_loaded_dataset,
    period_options,
    current_dataset_name,
    current_output_name,
    current_source_url,
    current_change_summary,
    current_dataset_title,
):
    """Generate dataset/output names and change summary from UI state.

    Args:
        loaded_dataset: CKAN dataset name or None.
        flux_source: ``wel`` or ``rch``.
        rate_mode: ``set`` or ``scale_percent``.
        new_rate: New rate value.
        suggested_name: Suggested dataset name selection.
        jwt_token: CKAN JWT token.
        color_by: Color-by mode.
        category_value: Selected category value.
        periods: Selected stress periods.
        layers: Selected layers.
        add_missing: Checkbox values for add-missing.
        selected_ids: Selected CELL_ID values.
        name_seed: Stable name seed string.
        current_dataset_name: Current dataset name input.
        current_output_name: Current output filename input.
        current_source_url: Current source URL input.
        current_change_summary: Current change summary input.

    Returns:
        Tuple of (dataset name, output filename, source URL, change summary, dataset title).
    """
    if not loaded_dataset:
        return (
            current_dataset_name,
            current_output_name,
            current_source_url,
            current_change_summary,
            current_dataset_title,
        )
    triggered = ctx.triggered_id
    if suggested_name and suggested_name != "__new__":
        current_dataset_name = suggested_name
    elif triggered in ("loaded-dataset", "load-counter"):
        current_dataset_name = _slugify(f"{loaded_dataset}-{name_seed}")
    rate_value = float(new_rate or 0.0)
    if rate_mode == "scale_percent":
        if rate_value < 0:
            suffix = f"{abs(rate_value):.0f}% reduction"
        elif rate_value > 0:
            suffix = f"{abs(rate_value):.0f}% increase"
        else:
            suffix = "0% change"
    else:
        suffix = f"set-{rate_value:.0f}"
    period_total = len(period_options or [])
    period_summary = _summarize_periods(list(periods or []), period_total or None)
    change_spec = {
        "flux_source": flux_source,
        "rate_mode": rate_mode,
        "new_rate": rate_value,
        "periods": sorted(list(periods or [])),
        "layers": sorted(list(layers or [])),
        "add_missing": bool(add_missing),
        "selection_count": len(selected_ids or []),
        "color_by": color_by,
        "category": category_value,
    }
    change_hash = hashlib.sha1(json.dumps(change_spec, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    base_name = _slugify(f"{loaded_dataset}-{change_hash}")
    dataset_name = current_dataset_name or base_name
    output_ext = ".rch" if flux_source == "rch" else ".wel"
    output_name = f"{loaded_dataset}_{suffix}_{change_hash}{output_ext}"
    if not output_name.lower().endswith(output_ext):
        output_name = f"{Path(output_name).stem}{output_ext}"
    source_url = current_source_url
    if not source_url and loaded_dataset:
        source_url = f"{CKAN_URL}/dataset/{loaded_dataset}"
    selection_count = len(selected_ids or [])
    selection_desc = f"Selected cells: {selection_count}"
    if color_by in ("GCD_Name", "PGMA_Name") and category_value:
        selection_desc = f"Category {color_by} = {category_value}"
    period_desc = period_summary
    if flux_source == "rch":
        layer_desc = "Layers: n/a"
        add_desc = "Add missing wells: n/a"
    else:
        layer_desc = "All layers" if not layers else f"Layers: {', '.join(str(l) for l in layers)}"
        add_desc = "Add missing wells: yes" if (add_missing or []) else "Add missing wells: no"
    change_summary = (
        f"{selection_desc}; {period_desc}; {layer_desc}; "
        f"Rate mode: {rate_mode}, New rate: {new_rate}; {add_desc}"
    )
    dataset_title = current_dataset_title or ""
    if triggered in ("loaded-dataset", "load-counter", "dataset-suggestions") or not dataset_title.strip():
        if loaded_dataset == "gam-carrizo-wilcox-aquifer-central-portion-version-3-02":
            dataset_title = f"Carrizo-Wilcox (v3.02) – updated-{name_seed}"
        else:
            dataset_title = (current_dataset_name or dataset_name or "").strip()
            if not dataset_title and loaded_dataset:
                try:
                    dataset_title = str(load_dataset(loaded_dataset)["dataset"].get("title") or "").strip()
                except Exception:
                    dataset_title = ""
    return dataset_name, output_name, source_url, change_summary, dataset_title


@app.callback(
    Output("selected-store", "data"),
    Output("category-warning", "data"),
    Input("map", "selectedData"),
    Input("map", "clickData"),
    Input("clear-selection", "n_clicks"),
    Input("select-category", "n_clicks"),
    State("selected-store", "data"),
    State("loaded-dataset", "data"),
    State("color-by", "value"),
    State("category-select", "value"),
)
def update_selection(
    selected_data,
    click_data,
    clear_clicks,
    category_clicks,
    selected_ids,
    loaded_dataset,
    color_by,
    category_value,
):
    """Update selected cell IDs based on map interactions.

    Args:
        selected_data: Lasso/box selection payload.
        click_data: Click selection payload.
        clear_clicks: Clear selection click count.
        category_clicks: Select category click count.
        selected_ids: Current selected CELL_ID values.
        loaded_dataset: CKAN dataset name or None.
        color_by: Color-by mode.
        category_value: Selected category value.

    Returns:
        Tuple of (updated selected ids, category warning flag).
    """
    selected_ids = [int(cid) for cid in (selected_ids or [])]
    triggered = ctx.triggered_id
    if triggered == "clear-selection":
        return [], False
    if triggered == "select-category":
        if not loaded_dataset or not category_value:
            return selected_ids, True
        data = load_dataset(loaded_dataset)
        gdf = data["gdf"]
        if color_by not in ("GCD_Name", "PGMA_Name") or color_by not in gdf.columns:
            return selected_ids, False
        matches = gdf[color_by].fillna("Unknown").astype(str) == str(category_value)
        cell_ids = gdf.loc[matches, "CELL_ID"].tolist()
        return sorted({int(cid) for cid in cell_ids}), False
    if triggered == "map" and selected_data:
        ids = []
        for pt in selected_data.get("points", []):
            custom = pt.get("customdata") or []
            if custom:
                ids.append(int(custom[0]))
        return sorted(set(ids)), False
    if triggered == "map" and click_data:
        custom = (click_data.get("points") or [{}])[0].get("customdata") or []
        if not custom:
            return selected_ids, False
        cid = int(custom[0])
        if cid in selected_ids:
            selected_ids.remove(cid)
        else:
            selected_ids.append(cid)
        return sorted(set(selected_ids)), False
    return selected_ids, False


@app.callback(
    Output("selection-status", "children"),
    Output("selection-status", "className"),
    Input("selected-store", "data"),
    Input("selection-warning", "data"),
)
def update_selection_status(selected_ids, selection_warning):
    """Update selection status text and warning styling.

    Args:
        selected_ids: Selected CELL_ID values.
        selection_warning: Warning flag for empty selections.

    Returns:
        Tuple of (status text, class name).
    """
    count = len(selected_ids or [])
    class_name = "status"
    if selection_warning and count == 0:
        class_name = "status warn-pulse"
    return f"Selected cells: {count}", class_name


@app.callback(
    Output("category-wrap", "className"),
    Input("category-warning", "data"),
    Input("color-by", "value"),
)
def update_category_warning(category_warning, color_by):
    """Highlight the category selector when a warning is active.

    Args:
        category_warning: Warning flag for category selection.
        color_by: Color-by mode.

    Returns:
        Class name string for the category wrapper.
    """
    base = "category-wrap"
    if color_by != "flux" and category_warning:
        return f"{base} warn-pulse"
    return base


@app.callback(
    Output("output-label", "children"),
    Input("flux-source", "value"),
)
def update_output_label(flux_source):
    """Switch the output label based on flux source.

    Args:
        flux_source: ``wel`` or ``rch``.

    Returns:
        Output label string.
    """
    return "Output RCH filename" if flux_source == "rch" else "Output WEL filename"


@app.callback(
    Output("status-message", "children"),
    Output("update-store", "data"),
    Output("selection-warning", "data"),
    Input("apply-rate", "n_clicks"),
    State("loaded-dataset", "data"),
    State("selected-store", "data"),
    State("flux-source", "value"),
    State("new-rate", "value"),
    State("rate-mode", "value"),
    State("add-missing", "value"),
    State("layers", "value"),
    State("periods", "value"),
    State("update-store", "data"),
    State("ckan-jwt", "data"),
    State("dataset-name", "value"),
    State("dataset-title", "value"),
    State("output-wel", "value"),
    State("source-url", "value"),
    State("change-summary", "value"),
    State("tapis-username", "data"),
)
def apply_rate(
    n_clicks,
    loaded_dataset,
    selected_ids,
    flux_source,
    new_rate,
    rate_mode,
    add_missing,
    layers,
    periods,
    update_counter,
    jwt_token,
    dataset_name,
    dataset_title,
    output_wel,
    source_url,
    change_summary,
    tapis_username,
):
    """Apply rate updates and optionally publish to CKAN.

    Args:
        n_clicks: Click count from the apply button.
        loaded_dataset: CKAN dataset name.
        selected_ids: Selected CELL_ID values.
        flux_source: ``wel`` or ``rch``.
        new_rate: New rate value.
        rate_mode: ``set`` or ``scale_percent``.
        add_missing: Checkbox values for add-missing.
        layers: Selected layers.
        periods: Selected stress periods.
        update_counter: Current update counter.
        jwt_token: CKAN JWT token.
        dataset_name: Output dataset name.
        dataset_title: Output dataset title.
        output_wel: Output WEL/RCH filename.
        source_url: Source URL string.
        change_summary: Change summary string.
        tapis_username: Tapis username for maintainer checks.

    Returns:
        Tuple of (status message, updated counter, selection warning flag).
    """
    if not n_clicks:
        return "", update_counter, False
    if not loaded_dataset:
        return "No dataset selected.", update_counter, True
    if not selected_ids:
        return "No cells selected.", update_counter, True
    data = load_dataset(loaded_dataset)
    wel = data["wel"]
    gdf = data["gdf"]
    print(
        "[apply] "
        f"dataset={loaded_dataset} flux_source={flux_source} "
        f"rate_mode={rate_mode} new_rate={new_rate} "
        f"periods={periods} layers={layers} add_missing={add_missing} "
        f"selected_count={len(selected_ids)} "
        f"dataset_name={dataset_name} output={output_wel} "
        f"source_url={source_url} change_summary={change_summary} "
        f"jwt={'yes' if jwt_token else 'no'}"
    )
    output_path = Path(output_wel or OUTPUT_WEL)
    target_ext = ".rch" if flux_source == "rch" else ".wel"
    if output_path.suffix.lower() != target_ext:
        output_path = output_path.with_suffix(target_ext)
    if flux_source == "rch":
        rch = data["rch"]
        if rch is None:
            return "RCH data not loaded.", update_counter, True
        updated = apply_rch_rate_update(
            rch,
            gdf,
            selected_ids,
            float(new_rate or 0.0),
            rate_mode,
            list(periods or []),
            output_path,
        )
    else:
        updated = apply_rate_update(
            wel,
            gdf,
            selected_ids,
            float(new_rate or 0.0),
            rate_mode,
            "yes" in (add_missing or []),
            list(layers or []),
            list(periods or []),
            output_path,
        )
    provenance = {
        "selected_cell_count": len(selected_ids),
        "rate_mode": rate_mode,
        "new_rate": float(new_rate or 0.0),
        "periods": list(periods or []),
        "change_summary": change_summary or "",
        "source_url": source_url or "",
        "flux_source": flux_source or "wel",
    }
    if flux_source != "rch":
        provenance["layers"] = list(layers or [])
        provenance["add_missing"] = "yes" in (add_missing or [])
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
        return (
            f"Updated {updated} cells. Wrote {output_path}. "
            f"Published CKAN dataset {dataset_id}.",
            (update_counter or 0) + 1,
            False,
        )
    except Exception as exc:
        if "dataset name conflict with original" in str(exc):
            return (
                f"Updated {updated} cells. Wrote {output_path}. "
                "CKAN publish failed: dataset name matches the original. "
                "Choose a new dataset name.",
                (update_counter or 0) + 1,
                False,
            )
        if "dataset maintainer mismatch" in str(exc):
            return (
                f"Updated {updated} cells. Wrote {output_path}. "
                "CKAN publish failed: dataset is owned by a different maintainer.",
                (update_counter or 0) + 1,
                False,
            )
        if "dataset name conflict" in str(exc):
            return (
                f"Updated {updated} cells. Wrote {output_path}. "
                "CKAN publish failed: dataset name already exists. "
                "Edit the dataset name and try again.",
                (update_counter or 0) + 1,
                False,
            )
        return (
            f"Updated {updated} cells. Wrote {output_path}. "
            f"CKAN publish failed: {exc}",
            (update_counter or 0) + 1,
            False,
        )


@app.callback(
    Output("map", "figure"),
    Input("loaded-dataset", "data"),
    Input("load-counter", "data"),
    Input("flux-source", "value"),
    Input("color-by", "value"),
    Input("color-period", "value"),
    Input("periods", "value"),
    Input("selected-store", "data"),
    Input("update-store", "data"),
    Input("map", "relayoutData"),
)
def update_map(
    loaded_dataset,
    _load_counter,
    flux_source,
    color_by,
    color_period,
    periods,
    selected_ids,
    _update,
    relayout,
):
    """Refresh the map when dataset or settings change.

    Args:
        loaded_dataset: CKAN dataset name or None.
        _load_counter: Load counter (unused).
        flux_source: ``wel`` or ``rch``.
        color_by: Color-by mode.
        periods: Selected stress periods.
        selected_ids: Selected CELL_ID values.
        _update: Update counter (unused).

    Returns:
        Plotly Figure for the map.
    """
    if not loaded_dataset:
        return go.Figure()
    data = load_dataset(loaded_dataset)
    gdf = data["gdf"]
    try:
        geom_types = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
        valid_mask = gdf.geometry.notna()
        invalid_count = int((~valid_mask).sum())
        print(f"[grid] geometry types: {geom_types}")
        print(f"[grid] null geometry rows: {invalid_count}")
        if valid_mask.any():
            sample = gdf.loc[valid_mask].geometry.iloc[0]
            print(f"[grid] sample geometry: {sample}")
    except Exception as exc:
        print(f"[grid] geometry debug failed: {exc}")
    zoom = None
    if isinstance(relayout, dict):
        zoom = relayout.get("map.zoom")
        if zoom is None:
            zoom = relayout.get("mapbox.zoom")
    cache_key = (
        loaded_dataset,
        flux_source,
        color_by,
        int(color_period) if color_period is not None else None,
        tuple(int(p) for p in (periods or [])),
        float(zoom) if zoom is not None else None,
        int(_update or 0),
    )
    cached_fig = _MAP_FIG_CACHE.get(cache_key)
    if cached_fig is not None:
        fig = go.Figure(cached_fig)
        _apply_selection(fig, gdf, selected_ids)
        return fig
    wel = data["wel"]
    rch = data["rch"]
    periods = periods or []
    map_periods = [color_period] if color_by == "flux" and color_period is not None else periods
    wel_cells = _collect_wel_cells_for_periods(wel, gdf, map_periods)
    rch_cells = {}
    if rch is not None:
        try:
            rch_cells = build_rch_cells_for_periods(rch, gdf, map_periods)
        except Exception:
            rch_cells = {}
    if flux_source == "rch":
        active_cells = rch_cells
        label = "Recharge"
        force_linear = True
    else:
        active_cells = wel_cells
        label = "Well"
        force_linear = False
    if color_by == "flux":
        values = [float(v) for v in active_cells.values()]
        nonzero = [v for v in values if v != 0.0]
        count = len(values)
        nz_count = len(nonzero)
        if nonzero:
            vmin = min(nonzero)
            vmax = max(nonzero)
        else:
            vmin = vmax = 0.0
        period_label = (
            f"SP {int(color_period) + 1}" if color_period is not None else "multi"
        )
        print(
            f"[flux] period={period_label} cells={count} nonzero={nz_count} "
            f"min={vmin:.6g} max={vmax:.6g}"
        )
    fig = _build_map_figure(
        gdf,
        active_cells,
        label,
        color_by,
        force_linear,
        [],
        zoom=zoom,
        show_grid=False,
    )
    _MAP_FIG_CACHE[cache_key] = fig
    _apply_selection(fig, gdf, selected_ids)
    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
