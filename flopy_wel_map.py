#!/usr/bin/env python3
"""Compatibility layer for the reorganized flopy_interactive package."""

from __future__ import annotations

from flopy_interactive.ckankit.search import (
    extract_standard_vars as _extract_standard_vars,
    resource_has_standard_var as _resource_has_standard_var,
    search_ckan_datasets as _search_ckan_datasets,
)
from flopy_interactive.data.download import (
    _flatten_single_dir,
    download_ckan_resource as _download_ckan_resource,
    extract_zip as _extract_zip,
    find_grid_data_path as _find_grid_data_path,
)
from flopy_interactive.data.grid import (
    _load_grid_from_gdb,
    _prepare_grid_gdf,
    load_grid_gdf,
    load_grid_resource as _load_grid_resource,
)
from flopy_interactive.data.rch import (
    _load_rch_numeric,
    _map_rch_cells,
    _rch_to_arrays,
    build_rch_cells,
    build_rch_cells_for_periods,
    load_rch,
)
from flopy_interactive.data.sample import ensure_barton_springs_wel, ensure_ebfz_grid
from flopy_interactive.data.wel import (
    apply_rate_update,
    collect_wel_cells_for_period_data as _collect_wel_cells_for_period_data,
    load_wel,
    scan_wel_metadata,
)
from flopy_interactive.viz.color_modes import (
    _discrete_colorscale,
    _signed_log,
    _signed_range,
    apply_color_mode as _apply_color_mode,
    update_flux_customdata as _update_flux_customdata,
)
from flopy_interactive.viz.map_widgets import build_plotly_selector
from flopy_interactive.app.notebook_ui import build_ui, render_ui, render_ui_from_ckan

__all__ = [
    "_apply_color_mode",
    "_collect_wel_cells_for_period_data",
    "_discrete_colorscale",
    "_download_ckan_resource",
    "_extract_standard_vars",
    "_extract_zip",
    "_find_grid_data_path",
    "_flatten_single_dir",
    "_load_grid_from_gdb",
    "_load_grid_resource",
    "_load_rch_numeric",
    "_map_rch_cells",
    "_prepare_grid_gdf",
    "_rch_to_arrays",
    "_resource_has_standard_var",
    "_search_ckan_datasets",
    "_signed_log",
    "_signed_range",
    "_update_flux_customdata",
    "apply_rate_update",
    "build_plotly_selector",
    "build_rch_cells",
    "build_rch_cells_for_periods",
    "build_ui",
    "ensure_barton_springs_wel",
    "ensure_ebfz_grid",
    "load_grid_gdf",
    "load_rch",
    "load_wel",
    "render_ui",
    "render_ui_from_ckan",
    "scan_wel_metadata",
]
