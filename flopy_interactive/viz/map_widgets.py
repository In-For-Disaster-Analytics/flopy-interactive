"""Plotly + ipywidgets map builders."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from flopy_interactive.viz.color_modes import apply_color_mode


def build_plotly_selector(
    gdf, wel_cells: Dict[int, float] | None = None
) -> Tuple[go.FigureWidget, widgets.Label, widgets.Label, set, callable]:
    """Build a Plotly selector widget with selection callbacks.

    Args:
        gdf: GeoDataFrame with grid metadata.
        wel_cells: Optional mapping of CELL_ID to flux values.

    Returns:
        Tuple of (figure, status label, map status label, selected ids set, apply-selection fn).
    """
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
    apply_color_mode(fig, gdf, wel_cells, "flux", flux_label="Well")
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
