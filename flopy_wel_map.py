#!/usr/bin/env python3
"""Helpers for interactive WEL updates using the EBFZ grid and Plotly."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import flopy
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output


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
    gdf = gdf.to_crs("EPSG:4326")
    centroids = gdf.geometry.centroid
    gdf["_lon"] = centroids.x
    gdf["_lat"] = centroids.y
    return gdf


def build_plotly_selector(gdf: gpd.GeoDataFrame) -> Tuple[go.FigureWidget, widgets.Label, set]:
    gdf_map = gdf[["CELL_ID", "ROW", "COL", "geometry"]].copy()
    grid_geojson = gdf_map.set_index("CELL_ID").__geo_interface__
    center_lat = float(gdf["_lat"].median())
    center_lon = float(gdf["_lon"].median())

    fig = go.FigureWidget()
    fig.add_trace(
        go.Choroplethmap(
            geojson=grid_geojson,
            locations=gdf["CELL_ID"],
            z=[1] * len(gdf),
            colorscale=["#e0e0e0", "#e0e0e0"],
            marker_opacity=0.35,
            marker_line_width=0.5,
            marker_line_color="#666",
            showscale=False,
            name="Grid",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=gdf["_lon"],
            lat=gdf["_lat"],
            mode="markers",
            marker=dict(size=6, color="#1f77b4"),
            name="Cells",
            customdata=np.stack([gdf["CELL_ID"], gdf["ROW"], gdf["COL"]], axis=1),
            hovertemplate="CELL_ID=%{customdata[0]}<br>ROW=%{customdata[1]} COL=%{customdata[2]}<extra></extra>",
            selected=dict(marker=dict(size=8, color="#d62728")),
            unselected=dict(marker=dict(opacity=0.5)),
        )
    )
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
    status = widgets.Label(value="Selected cells: 0")

    def _update_selected_from_inds(point_inds):
        selected_ids.clear()
        for idx in point_inds:
            cid = int(fig.data[1].customdata[idx][0])
            selected_ids.add(cid)
        status.value = f"Selected cells: {len(selected_ids)}"

    def _on_selection(trace, points, state):
        if hasattr(points, "point_inds") and points.point_inds:
            _update_selected_from_inds(points.point_inds)
        else:
            _update_selected_from_inds([])

    def _on_click(trace, points, state):
        if hasattr(points, "point_inds") and points.point_inds:
            for idx in points.point_inds:
                cid = int(fig.data[1].customdata[idx][0])
                if cid in selected_ids:
                    selected_ids.remove(cid)
                else:
                    selected_ids.add(cid)
            status.value = f"Selected cells: {len(selected_ids)}"

    fig.data[1].on_selection(_on_selection)
    fig.data[1].on_click(_on_click)

    return fig, status, selected_ids


def apply_rate_update(
    wel: flopy.modflow.ModflowWel,
    gdf: gpd.GeoDataFrame,
    selected_ids: Iterable[int],
    new_rate: float,
    rate_mode: str,
    add_missing: bool,
    layer_for_new: int,
    output_path: Path,
) -> int:
    spd = wel.stress_period_data.data
    cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
    selected_cells = {cell_lookup[cid] for cid in selected_ids if cid in cell_lookup}
    new_spd = {}
    for per, recs in spd.items():
        recs = recs.copy()
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
                new_recs = np.zeros(len(recs) + len(to_add), dtype=recs.dtype)
                new_recs[: len(recs)] = recs
                for idx, (row, col) in enumerate(to_add, start=len(recs)):
                    new_recs[idx]["k"] = int(layer_for_new)
                    new_recs[idx]["i"] = int(row)
                    new_recs[idx]["j"] = int(col)
                    new_recs[idx]["flux"] = float(new_rate)
                recs = new_recs
        new_spd[per] = recs
    wel.stress_period_data = new_spd
    wel.write_file(str(output_path))
    return len(selected_cells)


def render_ui(wel: flopy.modflow.ModflowWel, gdf: gpd.GeoDataFrame) -> None:
    fig, status, selected_ids = build_plotly_selector(gdf)

    help_text = widgets.HTML(
        """
        <b>Selection:</b> Use lasso/box on the map or click points to toggle selection.<br>
        <b>Rate mode:</b> <i>Set</i> replaces the rate, <i>Scale (%)</i> applies a percent change.<br>
        <b>New rate:</b> For <i>Set</i>, use the absolute pumping rate (negative for pumping). For
        <i>Scale (%)</i>, use percent change (e.g., 10 = +10%, -10 = -10%).<br>
        <b>Layer (k):</b> Used only when adding missing wells. 1 = top layer.<br>
        <b>Add missing wells:</b> Add wells in selected cells not present in the WEL.
        """
    )
    rate_mode = widgets.Dropdown(
        options=[("Set", "set"), ("Scale (%)", "scale_percent")],
        value="set",
        description="Rate mode",
    )
    rate_input = widgets.FloatText(value=-20.0, description="New rate")
    layer_input = widgets.IntText(value=1, description="Layer (k)")
    add_missing = widgets.Checkbox(value=False, description="Add missing wells")
    save_btn = widgets.Button(description="Apply + Save", button_style="primary")
    output = widgets.Output()

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
                layer_input.value,
                Path("barton_springs_updated.wel"),
            )
            print(f"Updated {updated} cells. Wrote barton_springs_updated.wel")

    save_btn.on_click(_apply_and_save)

    display(
        widgets.VBox(
            [
                fig,
                help_text,
                widgets.HBox([rate_mode, rate_input, layer_input, add_missing, save_btn]),
                status,
                output,
            ]
        )
    )
