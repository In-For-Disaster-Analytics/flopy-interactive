"""Notebook UI for interactive WEL updates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display

from flopy_interactive.ckankit.search import search_ckan_datasets
from flopy_interactive.data.download import download_ckan_resource
from flopy_interactive.data.grid import load_grid_resource
from flopy_interactive.data.rch import (
    _rch_to_arrays,
    build_rch_cells,
    build_rch_cells_for_periods,
    load_rch,
)
from flopy_interactive.data.wel import (
    apply_rate_update,
    build_cell_id_lookup,
    collect_wel_cells_for_period_data,
    load_wel,
)
from flopy_interactive.viz.color_modes import apply_color_mode, update_flux_customdata
from flopy_interactive.viz.map_widgets import build_plotly_selector


def build_ui(
    wel,
    gdf,
    show_dataset_controls: bool = True,
    rch=None,
) -> widgets.Widget:
    """Build the ipywidgets UI for editing WEL rates.

    Args:
        wel: FloPy WEL package.
        gdf: GeoDataFrame with grid metadata.
        show_dataset_controls: Whether to show CKAN dataset selectors.
        rch: Optional FloPy RCH package or array-like data.

    Returns:
        ipywidgets container widget.
    """
    cell_id_lookup = build_cell_id_lookup(gdf, wel)

    def _collect_wel_cells(row_offset: int, col_offset: int) -> Dict[int, float]:
        cells: Dict[int, float] = {}
        for recs in wel.stress_period_data.data.values():
            for rec in recs:
                if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
                    cell_id = cell_id_lookup.get(int(rec["node"]))
                else:
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
        return collect_wel_cells_for_period_data(
            wel, cell_id_lookup, period, row_offset, col_offset
        )

    if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
        wel_cells = _collect_wel_cells(0, 0)
        wel_cells_offset = {}
        use_offset = False
    else:
        wel_cells = _collect_wel_cells(0, 0)
        wel_cells_offset = _collect_wel_cells(1, 1)
        use_offset = len(wel_cells_offset) > len(wel_cells)
        if use_offset:
            wel_cells = wel_cells_offset

    rch_cells: Dict[int, float] = {}
    if rch is not None:
        try:
            rch_cells = build_rch_cells(rch, gdf)
        except Exception:
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
    dataset_details = widgets.HTML(value="")
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
            apply_color_mode(
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
            update_flux_customdata(fig, gdf, flux_values)
            apply_color_mode(
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
            if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
                wel_cells_offset = {}
            else:
                wel_cells_offset = {}
                for period in active_periods:
                    period_cells = _collect_wel_cells_for_period(period, 1, 1)
                    for cid, flux in period_cells.items():
                        if cid not in wel_cells_offset or abs(flux) > abs(wel_cells_offset[cid]):
                            wel_cells_offset[cid] = flux
        else:
            wel_cells_all = _collect_wel_cells(0, 0)
            if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
                wel_cells_offset = {}
            else:
                wel_cells_offset = _collect_wel_cells(1, 1)
        use_offset = len(wel_cells_offset) > len(wel_cells_all)
        wel_cells = wel_cells_offset if use_offset else wel_cells_all
        if rch is not None:
            try:
                rch_cells = build_rch_cells_for_periods(rch, gdf, active_periods)
            except Exception:
                rch_cells = {}
        match_info.value = (
            f"WEL/grid matches: {len(wel_cells)} cells"
            + (" (using +1 row/col offset)" if use_offset else "")
        )
        active_cells = wel_cells if flux_source.value == "wel" else rch_cells
        flux_values = [float(active_cells.get(int(cid), 0.0)) for cid in gdf["CELL_ID"]]
        label = "Well" if flux_source.value == "wel" else "Recharge"
        update_flux_customdata(fig, gdf, flux_values)
        apply_color_mode(
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
            datasets = search_ckan_datasets()
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
    wel,
    gdf,
    show_dataset_controls: bool = True,
    rch=None,
) -> None:
    """Render the WEL editor UI in a notebook.

    Args:
        wel: FloPy WEL package.
        gdf: GeoDataFrame with grid metadata.
        show_dataset_controls: Whether to show CKAN dataset selectors.
        rch: Optional FloPy RCH package or array-like data.

    Returns:
        None.
    """
    display(build_ui(wel, gdf, show_dataset_controls=show_dataset_controls, rch=rch))


def render_ui_from_ckan(data_dir: Path = Path("ckan_data")) -> None:
    """Fetch a CKAN dataset and render the notebook UI.

    Args:
        data_dir: Directory used to store downloaded CKAN resources.

    Returns:
        None.
    """
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
            datasets = search_ckan_datasets()
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
