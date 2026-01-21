"""Color scaling and update helpers for Plotly maps."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly import colors as pc


def _signed_log(values: Sequence[float]) -> List[float]:
    """Apply a signed log transform for diverging scales."""
    return [float(np.sign(v) * np.log10(1.0 + abs(v))) for v in values]


def _signed_range(values: Sequence[float]) -> Tuple[float, float]:
    """Compute symmetric ranges for diverging color scales."""
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
    """Expand a discrete palette into Plotly colorscale tuples."""
    if count <= 1:
        return [(0.0, colors[0]), (1.0, colors[0])]
    scale: List[Tuple[float, str]] = []
    for idx, color in enumerate(colors):
        t0 = idx / (count - 1)
        t1 = min((idx + 1) / (count - 1), 1.0)
        scale.append((t0, color))
        scale.append((t1, color))
    return scale


def apply_color_mode(
    fig: go.FigureWidget,
    gdf,
    wel_cells: Dict[int, float],
    mode: str,
    normalize: bool = False,
    flux_label: str = "Flux",
    force_linear: bool = False,
) -> None:
    """Apply a color mode to an existing Plotly map figure.

    Args:
        fig: Plotly FigureWidget to update.
        gdf: GeoDataFrame with grid metadata.
        wel_cells: Mapping of CELL_ID to flux values.
        mode: ``flux`` or category column name.
        normalize: Whether to force linear normalization.
        flux_label: Colorbar label text.
        force_linear: Whether to skip signed-log scaling.

    Returns:
        None.
    """
    cell_ids = gdf["CELL_ID"].tolist()
    flux_values = [float(wel_cells.get(int(cid), 0.0)) for cid in cell_ids]
    diverging = [(0.0, "#2b6cb0"), (0.5, "#ffffff"), (1.0, "#c53030")]
    has_choropleth = bool(fig.data) and fig.data[0].type.startswith("choropleth")
    scatter_idx = 1 if has_choropleth else 0

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
            if has_choropleth:
                fig.data[0].update(z=z_vals, zmin=zmin, zmax=zmax, colorscale=diverging)
            fig.data[scatter_idx].marker.update(
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
        if has_choropleth:
            fig.data[0].update(z=codes, zmin=0.0, zmax=cmax, colorscale=colorscale)
        fig.data[scatter_idx].marker.update(
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


def update_flux_customdata(fig: go.FigureWidget, gdf, flux_values: Sequence[float]) -> None:
    """Update hover customdata after a flux change.

    Args:
        fig: Plotly FigureWidget to update.
        gdf: GeoDataFrame with grid metadata.
        flux_values: Sequence of flux values aligned to gdf rows.

    Returns:
        None.
    """
    has_choropleth = bool(fig.data) and fig.data[0].type.startswith("choropleth")
    scatter_idx = 1 if has_choropleth else 0
    if has_choropleth:
        fig.data[0].customdata = np.stack(
            [
                gdf["CELL_ID"],
                flux_values,
                gdf["GCD_Name"].fillna("Unknown").astype(str),
                gdf["PGMA_Name"].fillna("Unknown").astype(str),
            ],
            axis=1,
        )
    fig.data[scatter_idx].customdata = np.stack(
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
