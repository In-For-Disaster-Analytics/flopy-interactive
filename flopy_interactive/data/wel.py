"""WEL file loading and update helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import flopy
import numpy as np


class _SimpleStressPeriodData:
    """Lightweight container for MFUSG stress period data."""

    def __init__(self, data: Dict[int, np.recarray], dtype: np.dtype) -> None:
        self.data = data
        self.dtype = dtype


class MfusgWel:
    """Minimal MFUSG WEL representation with node-based records."""

    is_mfusg = True

    def __init__(self, header_line: str, spd: Dict[int, np.recarray], dtype: np.dtype) -> None:
        self.header_line = header_line.rstrip("\n")
        self.stress_period_data = _SimpleStressPeriodData(spd, dtype)

    def write_file(self, output_path: Path) -> None:
        """Write MFUSG WEL records back to disk."""
        lines = [self.header_line]
        periods = sorted(self.stress_period_data.data.keys())
        for per in periods:
            recs = self.stress_period_data.data.get(per)
            itmp = 0 if recs is None else len(recs)
            lines.append(f"{itmp} 0 0                  Stress Period {per + 1}")
            if itmp > 0:
                for rec in recs:
                    lines.append(f"{int(rec['node'])} {float(rec['flux']):.6g}")
        output_path.write_text("\n".join(lines) + "\n")


class LazyMfusgWel(MfusgWel):
    """Lazy MFUSG WEL reader that loads selected stress periods on demand."""

    is_lazy = True

    def __init__(self, path: Path, header_line: str, index: list, dtype: np.dtype) -> None:
        super().__init__(header_line, {}, dtype)
        self.path = path
        self.index = index
        self.period_count = len(index)

    def get_period(self, period: int) -> np.recarray:
        if period < 0 or period >= self.period_count:
            return np.recarray(0, dtype=self.stress_period_data.dtype)
        entry = self.index[period]
        itmp = entry["itmp"]
        if itmp < 0:
            return self.get_period(period - 1) if period > 0 else np.recarray(0, dtype=self.stress_period_data.dtype)
        if itmp == 0:
            return np.recarray(0, dtype=self.stress_period_data.dtype)
        records = []
        with self.path.open() as handle:
            handle.seek(entry["offset"])
            for _ in range(itmp):
                line = _strip_comment(handle.readline())
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                node = int(float(parts[0]))
                flux = float(parts[1])
                records.append((node, flux))
        return np.rec.array(records, dtype=self.stress_period_data.dtype) if records else np.recarray(0, dtype=self.stress_period_data.dtype)

    def load_all(self) -> Dict[int, np.recarray]:
        spd: Dict[int, np.recarray] = {}
        for per in range(self.period_count):
            spd[per] = self.get_period(per)
        return spd


def _strip_comment(line: str) -> str:
    for token in ("#", ";"):
        if token in line:
            line = line.split(token, 1)[0]
    return line.strip()


def _detect_mfusg_wel(path: Path) -> bool:
    """Return True if the WEL file looks like MFUSG node-based format."""
    lines = [_strip_comment(line) for line in path.read_text().splitlines()]
    data_lines = [line for line in lines if line]
    if len(data_lines) < 3:
        return False
    # First line is header; next line is stress-period header; third line is first record
    first_record = data_lines[2].split()
    if len(first_record) < 2:
        return False
    try:
        int(first_record[1])
    except ValueError:
        return True
    return False


def _load_mfusg_wel(wel_path: Path) -> MfusgWel:
    """Load MFUSG node-based WEL data from disk."""
    lines = [_strip_comment(line) for line in wel_path.read_text().splitlines()]
    data_lines = [line for line in lines if line]
    if len(data_lines) < 2:
        raise ValueError("WEL file is empty or has no data.")
    header = data_lines[0]
    spd: Dict[int, np.recarray] = {}
    dtype = np.dtype([("node", "i4"), ("flux", "f8")])
    idx = 1
    per = 0
    while idx < len(data_lines):
        tokens = data_lines[idx].split()
        idx += 1
        if not tokens:
            continue
        itmp = int(tokens[0])
        if itmp < 0 and (per - 1) in spd:
            spd[per] = np.rec.array(spd[per - 1], dtype=dtype)
            per += 1
            continue
        records = []
        if itmp > 0:
            for _ in range(itmp):
                if idx >= len(data_lines):
                    raise ValueError("Unexpected end of file while scanning wells.")
                parts = data_lines[idx].split()
                idx += 1
                if len(parts) < 2:
                    continue
                node = int(float(parts[0]))
                flux = float(parts[1])
                records.append((node, flux))
        spd[per] = np.rec.array(records, dtype=dtype) if records else np.recarray(0, dtype=dtype)
        per += 1
    return MfusgWel(header, spd, dtype)


def _index_mfusg_wel(wel_path: Path) -> LazyMfusgWel:
    """Index MFUSG WEL file for lazy period access."""
    dtype = np.dtype([("node", "i4"), ("flux", "f8")])
    index = []
    with wel_path.open() as handle:
        header = _strip_comment(handle.readline()).rstrip("\n")
        per = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line = _strip_comment(line)
            if not line:
                continue
            tokens = line.split()
            itmp = int(tokens[0])
            offset = handle.tell()
            if itmp > 0:
                for _ in range(itmp):
                    handle.readline()
            index.append({"itmp": itmp, "offset": offset})
            per += 1
    return LazyMfusgWel(wel_path, header, index, dtype)


def scan_wel_metadata(path: Path) -> Tuple[int, int, int, int]:
    """Scan a MODFLOW WEL file for grid dimensions.

    Args:
        path: Path to the WEL file.

    Returns:
        Tuple of (nper, nlay, nrow, ncol).
    """
    lines = [_strip_comment(line) for line in path.read_text().splitlines()]
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
    if _detect_mfusg_wel(wel_path):
        return _index_mfusg_wel(wel_path)
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


def build_cell_id_lookup(gdf, wel) -> Dict:
    """Build a cell-id lookup keyed by node or (row, col)."""
    if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
        return dict(zip(gdf["NODE_NUM"], gdf["CELL_ID"]))
    return dict(zip(zip(gdf["ROW"], gdf["COL"]), gdf["CELL_ID"]))


def collect_wel_cells_for_period_data(
    wel: flopy.modflow.ModflowWel,
    cell_id_lookup: Dict,
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
    if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
        if hasattr(wel, "get_period"):
            recs = wel.get_period(period)
        else:
            spd = wel.stress_period_data.data
            if period not in spd:
                return cells
            recs = spd[period]
    else:
        spd = wel.stress_period_data.data
        if period not in spd:
            return cells
        recs = spd[period]
    for rec in recs:
        if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
            node = int(rec["node"])
            cell_id = cell_id_lookup.get(node)
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
    base_dtype = getattr(wel.stress_period_data, "dtype", None)
    if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
        node_lookup = dict(zip(gdf["CELL_ID"], gdf["NODE_NUM"]))
        selected_cells = {int(node_lookup[cid]) for cid in selected_ids if cid in node_lookup}
        if hasattr(wel, "load_all"):
            spd = wel.load_all()
            wel.stress_period_data.data = spd
    else:
        cell_lookup = dict(zip(gdf["CELL_ID"], zip(gdf["ROW"], gdf["COL"])))
        selected_cells = {cell_lookup[cid] for cid in selected_ids if cid in cell_lookup}
    updated_spd = spd
    for per, recs in spd.items():
        if base_dtype is None and hasattr(recs, "dtype") and recs.dtype.names:
            base_dtype = recs.dtype
        if base_dtype is None:
            raise ValueError("WEL stress_period_data dtype unavailable.")
        if periods_for_update and per not in periods_for_update:
            continue
        recs = np.rec.array(recs, dtype=base_dtype)
        if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
            mask = np.array([int(rec["node"]) in selected_cells for rec in recs], dtype=bool)
        else:
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
            if hasattr(wel, "is_mfusg") and getattr(wel, "is_mfusg"):
                existing = set(int(r["node"]) for r in recs)
                to_add = [node for node in selected_cells if node not in existing]
                if to_add:
                    new_recs = np.recarray(len(recs) + len(to_add), dtype=base_dtype)
                    new_recs[: len(recs)] = recs
                    idx = len(recs)
                    for node in to_add:
                        new_recs[idx]["node"] = int(node)
                        new_recs[idx]["flux"] = float(new_rate)
                        idx += 1
                    recs = new_recs
            else:
                existing = set((int(r["i"]), int(r["j"])) for r in recs)
                to_add = [cell for cell in selected_cells if cell not in existing]
                if to_add:
                    layers = [int(layer) for layer in layers_for_new] or [1]
                    new_recs = np.recarray(len(recs) + len(to_add) * len(layers), dtype=base_dtype)
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
        updated_spd[per] = recs
    if base_dtype is not None:
        for per, recs in updated_spd.items():
            if recs is None or len(recs) == 0:
                updated_spd[per] = np.recarray(0, dtype=base_dtype)
            else:
                updated_spd[per] = np.rec.array(recs, dtype=base_dtype)
    wel.write_file(str(output_path))
    return len(selected_cells)


def get_wel_period_keys(wel) -> list[int]:
    """Return stress period indices for WEL, including lazy MFUSG."""
    if hasattr(wel, "period_count"):
        return list(range(int(wel.period_count)))
    spd = getattr(wel, "stress_period_data", None)
    if spd is not None and hasattr(spd, "data"):
        return sorted(list(spd.data.keys()))
    return []
