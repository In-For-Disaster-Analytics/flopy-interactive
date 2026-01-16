"""WEL file loading and update helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import flopy
import numpy as np


def scan_wel_metadata(path: Path) -> Tuple[int, int, int, int]:
    """Scan a MODFLOW WEL file for grid dimensions.

    Args:
        path: Path to the WEL file.

    Returns:
        Tuple of (nper, nlay, nrow, ncol).
    """
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
    """Load a MODFLOW WEL package from disk.

    Args:
        wel_path: Path to the WEL file.

    Returns:
        FloPy WEL package.
    """
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


def collect_wel_cells_for_period_data(
    wel: flopy.modflow.ModflowWel,
    cell_id_lookup: Dict[Tuple[int, int], int],
    period: int,
    row_offset: int,
    col_offset: int,
) -> Dict[int, float]:
    """Collect WEL flux values for a single stress period.

    Args:
        wel: FloPy WEL package.
        cell_id_lookup: Mapping of (row, col) to CELL_ID.
        period: Stress period index.
        row_offset: Row offset applied to WEL indices.
        col_offset: Column offset applied to WEL indices.

    Returns:
        Mapping of CELL_ID to flux value.
    """
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


def apply_rate_update(
    wel: flopy.modflow.ModflowWel,
    gdf,
    selected_ids: Iterable[int],
    new_rate: float,
    rate_mode: str,
    add_missing: bool,
    layers_for_new: Sequence[int],
    periods_for_update: Sequence[int],
    output_path: Path,
) -> int:
    """Apply rate updates to a WEL package and write a new file.

    Args:
        wel: FloPy WEL package to update.
        gdf: GeoDataFrame with ROW/COL/CELL_ID columns.
        selected_ids: CELL_ID values to update.
        new_rate: New rate value or scale percent.
        rate_mode: ``set`` or ``scale_percent``.
        add_missing: Whether to add new wells for selected cells.
        layers_for_new: Layers to use when adding new wells.
        periods_for_update: Stress periods to update; empty for all.
        output_path: Path to write the updated WEL file.

    Returns:
        Count of selected cells processed.
    """
    spd = wel.stress_period_data.data
    spd_dtype = getattr(wel.stress_period_data, "dtype", None)
    cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
    selected_cells = {cell_lookup[cid] for cid in selected_ids if cid in cell_lookup}
    new_spd = {}
    for per, recs in spd.items():
        if spd_dtype is not None:
            recs = np.array(recs, dtype=spd_dtype)
        else:
            recs = np.array(recs)
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
    if spd_dtype is not None:
        cleaned = {}
        for per, recs in new_spd.items():
            if recs is None or len(recs) == 0:
                cleaned[per] = np.zeros(0, dtype=spd_dtype)
            else:
                cleaned[per] = np.array(recs, dtype=spd_dtype)
        new_spd = cleaned
    wel.stress_period_data = new_spd
    wel.write_file(str(output_path))
    return len(selected_cells)
