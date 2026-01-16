#!/usr/bin/env python3
"""Dash dashboard for WEL/RCH visualization and updates."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import dash
from dash import Input, Output, State, dcc, html, ctx
import numpy as np
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import ckan_publish as ckanp
from flopy_interactive.ckankit.search import resource_has_standard_var, search_ckan_datasets
from flopy_interactive.data.download import download_ckan_resource
from flopy_interactive.data.grid import load_grid_resource
from flopy_interactive.data.rch import build_rch_cells_for_periods, load_rch
from flopy_interactive.data.wel import apply_rate_update, collect_wel_cells_for_period_data, load_wel
from flopy_interactive.viz.color_modes import apply_color_mode


DATA_DIR = Path(os.environ.get("FLOPY_DATA_DIR", "ckan_data"))
OUTPUT_WEL = Path(os.environ.get("FLOPY_OUTPUT_WEL", "barton_springs_updated.wel"))


def get_datasets() -> List[Dict]:
    """Fetch CKAN datasets for the app session.

    Args:
        None.

    Returns:
        List of dataset metadata dicts.
    """
    return search_ckan_datasets()


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
    cell_id_lookup = dict(zip(zip(gdf["ROW"], gdf["COL"]), gdf["CELL_ID"]))
    nlay = 1
    if hasattr(wel, "parent") and wel.parent is not None:
        nlay = int(getattr(wel.parent.dis, "nlay", 1))
    elif hasattr(wel, "model") and wel.model is not None:
        nlay = int(getattr(wel.model.dis, "nlay", 1))
    elif hasattr(wel, "_model") and wel._model is not None:
        nlay = int(getattr(wel._model.dis, "nlay", 1))
    return {
        "dataset": dataset,
        "gdf": gdf,
        "wel": wel,
        "rch": rch,
        "cell_id_lookup": cell_id_lookup,
        "nlay": nlay,
    }


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
    cell_id_lookup = dict(zip(zip(gdf["ROW"], gdf["COL"]), gdf["CELL_ID"]))
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

    if periods:
        base = _merge_periods(0, 0)
        offset = _merge_periods(1, 1)
    else:
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
) -> go.Figure:
    """Build the Plotly map figure with grid and selection overlays.

    Args:
        gdf: GeoDataFrame with grid metadata.
        cells: Mapping of CELL_ID to flux values.
        flux_label: Label for the colorbar.
        color_by: ``flux`` or category column name.
        force_linear: Whether to force linear scaling.
        selected_ids: Iterable of selected CELL_ID values.

    Returns:
        Plotly Figure for the map.
    """
    gdf_map = gdf[["CELL_ID", "ROW", "COL", "geometry"]].copy()
    grid_geojson = gdf_map.set_index("CELL_ID").__geo_interface__
    center_lat = float(gdf["_lat"].median())
    center_lon = float(gdf["_lon"].median())

    fig = go.Figure()
    z_values = [float(cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
    fig.add_trace(
        go.Choroplethmap(
            geojson=grid_geojson,
            locations=gdf["CELL_ID"],
            z=z_values,
            colorscale=[(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")],
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
    fig.add_trace(
        go.Scattermap(
            lon=gdf["_lon"],
            lat=gdf["_lat"],
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
                    gdf["CELL_ID"],
                    gdf["ROW"],
                    gdf["COL"],
                    z_values,
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

    apply_color_mode(
        fig,
        gdf,
        cells,
        color_by,
        normalize=False,
        flux_label=flux_label,
        force_linear=force_linear,
    )

    selected_ids = {int(cid) for cid in selected_ids}
    if selected_ids:
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
        try:
            details = ckanp.package_show(jwt_token, name)
        except Exception:
            continue
        maintainer = str(details.get("maintainer", "")).strip().lower()
        maintainer_email = str(details.get("maintainer_email", "")).strip().lower()
        target = username.strip().lower()
        if maintainer and (maintainer == target or maintainer_email == target):
            options.append({"label": details.get("title", name), "value": name})
    return options

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
        dcc.Store(id="ckan-jwt", data=""),
        dcc.Store(id="login-message", data=""),
        dcc.Store(id="tapis-username", data=""),
        dcc.Store(id="name-seed", data=str(uuid.uuid4())),
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
                        html.Label("Stress periods"),
                        dcc.Dropdown(
                            id="periods",
                            options=[],
                            value=[],
                            multi=True,
                            className="dropdown-scroll",
                        ),
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
    spd = wel.stress_period_data.data
    period_keys = sorted(list(spd.keys())) if spd else [0]
    period_options = [{"label": f"SP {idx + 1}", "value": idx} for idx in period_keys]
    nlay = data["nlay"]
    layer_options = [{"label": str(layer), "value": layer} for layer in range(1, nlay + 1)]
    return period_options, layer_options


@app.callback(
    Output("loaded-dataset", "data"),
    Output("dataset-status", "children"),
    Output("load-counter", "data"),
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
        return None, "No dataset selected.", load_counter
    try:
        load_dataset(selected_dataset)
    except Exception as exc:
        return selected_dataset, f"Failed to load dataset: {exc}", load_counter
    load_counter = (load_counter or 0) + 1
    return selected_dataset, f"Loaded dataset: {selected_dataset}", load_counter


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
            spd = wel.stress_period_data.data
            period_keys = sorted(list(spd.keys())) if spd else [0]
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
    if color_by not in ("GCD_Name", "PGMA_Name"):
        return [], None, {"display": "none"}, True
    categories = (
        gdf[color_by].fillna("Unknown").astype(str).drop_duplicates().tolist()
    )
    options = [{"label": value, "value": value} for value in categories]
    return options, None, {"display": "block"}, False


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
    return _owned_gam_datasets(username or "", jwt_token or "")


@app.callback(
    Output("dataset-name", "value"),
    Output("output-wel", "value"),
    Output("source-url", "value"),
    Output("change-summary", "value"),
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
    State("name-seed", "data"),
    State("dataset-name", "value"),
    State("output-wel", "value"),
    State("source-url", "value"),
    State("change-summary", "value"),
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
    name_seed,
    current_dataset_name,
    current_output_name,
    current_source_url,
    current_change_summary,
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
        Tuple of (dataset name, output filename, source URL, change summary).
    """
    if not loaded_dataset:
        return current_dataset_name, current_output_name, current_source_url, current_change_summary
    if suggested_name:
        current_dataset_name = suggested_name
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
    base_name = _slugify(f"{loaded_dataset}-{name_seed}")
    dataset_name = current_dataset_name or base_name
    output_ext = ".rch" if flux_source == "rch" else ".wel"
    output_name = current_output_name or f"{loaded_dataset}_{suffix}{output_ext}"
    if not output_name.lower().endswith(output_ext):
        output_name = f"{Path(output_name).stem}{output_ext}"
    source_url = current_source_url
    if not source_url and jwt_token and loaded_dataset:
        try:
            details = ckanp.package_show(jwt_token, loaded_dataset)
            source_url = details.get("url")
            if not source_url:
                resources = details.get("resources", [])
                wel_res = next(
                    (res for res in resources if resource_has_standard_var(res, ckanp.WEL_STANDARD_VAR)),
                    None,
                )
                if wel_res:
                    source_url = wel_res.get("url")
        except Exception:
            source_url = current_source_url
    selection_count = len(selected_ids or [])
    selection_desc = f"Selected cells: {selection_count}"
    if color_by in ("GCD_Name", "PGMA_Name") and category_value:
        selection_desc = f"Category {color_by} = {category_value}"
    period_desc = "All periods" if not periods else f"Periods: {', '.join(str(p) for p in periods)}"
    layer_desc = "All layers" if not layers else f"Layers: {', '.join(str(l) for l in layers)}"
    add_desc = "Add missing wells: yes" if (add_missing or []) else "Add missing wells: no"
    change_summary = (
        f"{selection_desc}; {period_desc}; {layer_desc}; "
        f"Rate mode: {rate_mode}, New rate: {new_rate}; {add_desc}"
    )
    return dataset_name, output_name, source_url, change_summary


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
        if color_by not in ("GCD_Name", "PGMA_Name"):
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
    State("new-rate", "value"),
    State("rate-mode", "value"),
    State("add-missing", "value"),
    State("layers", "value"),
    State("periods", "value"),
    State("update-store", "data"),
    State("ckan-jwt", "data"),
    State("dataset-name", "value"),
    State("output-wel", "value"),
    State("source-url", "value"),
    State("change-summary", "value"),
    State("tapis-username", "data"),
)
def apply_rate(
    n_clicks,
    loaded_dataset,
    selected_ids,
    new_rate,
    rate_mode,
    add_missing,
    layers,
    periods,
    update_counter,
    jwt_token,
    dataset_name,
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
        new_rate: New rate value.
        rate_mode: ``set`` or ``scale_percent``.
        add_missing: Checkbox values for add-missing.
        layers: Selected layers.
        periods: Selected stress periods.
        update_counter: Current update counter.
        jwt_token: CKAN JWT token.
        dataset_name: Output dataset name.
        output_wel: Output WEL filename.
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
    output_path = Path(output_wel or OUTPUT_WEL)
    if output_path.suffix.lower() != ".wel":
        output_path = output_path.with_suffix(".wel")
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
        "layers": list(layers or []),
        "periods": list(periods or []),
        "change_summary": change_summary or "",
        "source_url": source_url or "",
    }
    try:
        result = ckanp.publish_updated_wel(
            loaded_dataset,
            output_path,
            provenance,
            jwt_token=jwt_token or None,
            new_dataset_name=dataset_name or None,
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
    Input("periods", "value"),
    Input("selected-store", "data"),
    Input("update-store", "data"),
)
def update_map(
    loaded_dataset, _load_counter, flux_source, color_by, periods, selected_ids, _update
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
    wel = data["wel"]
    rch = data["rch"]
    periods = periods or []
    wel_cells = _collect_wel_cells_for_periods(wel, gdf, periods)
    rch_cells = {}
    if rch is not None:
        try:
            rch_cells = build_rch_cells_for_periods(rch, gdf, periods)
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
    return _build_map_figure(
        gdf,
        active_cells,
        label,
        color_by,
        force_linear,
        selected_ids or [],
    )


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
