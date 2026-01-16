"""RCH file loading and mapping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import flopy
import numpy as np


def _load_rch_numeric(rch_path: Path, nrow: int, ncol: int) -> np.ndarray:
    """Parse numeric values from a text RCH file.

    Args:
        rch_path: Path to the RCH file.
        nrow: Number of grid rows.
        ncol: Number of grid columns.

    Returns:
        NumPy array of RCH values.
    """
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
    """Load a MODFLOW RCH package from disk.

    Args:
        rch_path: Path to the RCH file.
        nrow: Number of grid rows.
        ncol: Number of grid columns.
        nper: Number of stress periods for the model.

    Returns:
        FloPy RCH package if load succeeds, otherwise a NumPy array.
    """
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
    """Normalize an RCH package or array-like into 2D arrays.

    Args:
        rch: FloPy RCH package or array-like input.

    Returns:
        List of 2D NumPy arrays for each period.
    """
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


def _map_rch_cells(arrays: Sequence[np.ndarray], gdf) -> Dict[int, float]:
    """Project RCH arrays onto grid cell IDs with offset checks.

    Args:
        arrays: Sequence of 2D NumPy arrays.
        gdf: GeoDataFrame with ROW/COL/CELL_ID columns.

    Returns:
        Mapping of CELL_ID to RCH flux value.
    """
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


def build_rch_cells(rch, gdf) -> Dict[int, float]:
    """Map RCH arrays to grid cell IDs.

    Args:
        rch: FloPy RCH package or array-like input.
        gdf: GeoDataFrame with ROW/COL/CELL_ID columns.

    Returns:
        Mapping of CELL_ID to RCH flux value.
    """
    arrays = _rch_to_arrays(rch)
    return _map_rch_cells(arrays, gdf)


def build_rch_cells_for_periods(
    rch,
    gdf,
    periods: Sequence[int],
) -> Dict[int, float]:
    """Map RCH arrays for selected stress periods to grid cell IDs.

    Args:
        rch: FloPy RCH package or array-like input.
        gdf: GeoDataFrame with ROW/COL/CELL_ID columns.
        periods: Stress period indices to include.

    Returns:
        Mapping of CELL_ID to RCH flux value.
    """
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
