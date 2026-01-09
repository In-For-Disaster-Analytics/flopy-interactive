#!/usr/bin/env python3
"""
Update a MODFLOW-2005 WEL file and write an MF6 WEL package using FloPy.

Example:
  python3 update_wel_mf6.py \
    --input barton_springs_2001_2010average.wel \
    --output barton_springs_mf6.wel \
    --multiplier 1.1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


Well = Tuple[int, int, int, float]  # (k, i, j, q) 1-based indices


def _strip_comment(line: str) -> str:
    for token in ("#", ";"):
        if token in line:
            line = line.split(token, 1)[0]
    return line.strip()


def scan_wel_metadata(path: Path) -> Tuple[int, int, int, int]:
    lines = [_strip_comment(line) for line in path.read_text().splitlines()]
    data_lines = [line for line in lines if line]
    if not data_lines:
        raise ValueError("WEL file is empty or has no data.")

    header = data_lines.pop(0).split()
    if len(header) < 2:
        raise ValueError("WEL header must contain at least two integers.")

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
            if len(parts) < 4:
                raise ValueError(f"Invalid well line: {data_lines[idx-1]!r}")
            k, i, j = (int(parts[0]), int(parts[1]), int(parts[2]))
            max_k = max(max_k, k)
            max_i = max(max_i, i)
            max_j = max(max_j, j)

    return nper, max_k, max_i, max_j


def _record_to_tuple(record) -> Tuple[int, int, int, float]:
    if hasattr(record, "dtype") and record.dtype.names:
        k = int(record["k"])
        i = int(record["i"])
        j = int(record["j"])
        if "flux" in record.dtype.names:
            q = float(record["flux"])
        else:
            q = float(record["q"])
        return k, i, j, q
    k, i, j, q = record[:4]
    return int(k), int(i), int(j), float(q)


def read_mf2005_wel(path: Path, nlay: int, nrow: int, ncol: int, nper: int) -> List[List[Well]]:
    try:
        import flopy
    except ImportError as exc:
        raise SystemExit("FloPy is required. Install with: pip install flopy") from exc

    model = flopy.modflow.Modflow(modelname="wel_read", model_ws=str(path.parent))
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
    wel = flopy.modflow.ModflowWel.load(str(path), model)
    spd = wel.stress_period_data.data or {}

    min_index = None
    for per in spd.values():
        for rec in per:
            k, i, j, _ = _record_to_tuple(rec)
            min_index = min(k, i, j) if min_index is None else min(min_index, k, i, j)
    zero_based = min_index == 0

    per_data: List[List[Well]] = []
    for per in range(nper):
        records = spd.get(per, [])
        wells: List[Well] = []
        for rec in records:
            k, i, j, q = _record_to_tuple(rec)
            if zero_based:
                k += 1
                i += 1
                j += 1
            wells.append((k, i, j, q))
        per_data.append(wells)

    return per_data


def load_updates_csv(path: Path) -> Dict[Tuple[int, int, int], float]:
    updates: Dict[Tuple[int, int, int], float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"k", "i", "j", "q"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must include headers: k,i,j,q")
        for row in reader:
            key = (int(row["k"]), int(row["i"]), int(row["j"]))
            updates[key] = float(row["q"])
    return updates


def apply_updates(
    per_data: List[List[Well]],
    multiplier: float | None,
    set_rate: float | None,
    updates: Dict[Tuple[int, int, int], float] | None,
    add_missing: bool,
) -> List[List[Well]]:
    updated: List[List[Well]] = []
    for wells in per_data:
        wells_out: List[Well] = []
        seen = set()
        for k, i, j, q in wells:
            if updates and (k, i, j) in updates:
                q = updates[(k, i, j)]
                seen.add((k, i, j))
            if multiplier is not None:
                q *= multiplier
            if set_rate is not None:
                q = set_rate
            wells_out.append((k, i, j, q))
        if updates and add_missing:
            for key, q in updates.items():
                if key in seen:
                    continue
                k, i, j = key
                wells_out.append((k, i, j, q))
        updated.append(wells_out)
    return updated


def to_mf6_spd(per_data: List[List[Well]]) -> Dict[int, List[Tuple[Tuple[int, int, int], float]]]:
    spd: Dict[int, List[Tuple[Tuple[int, int, int], float]]] = {}
    for per, wells in enumerate(per_data):
        items = []
        for k, i, j, q in wells:
            # FloPy expects zero-based indices for MF6
            items.append(((k - 1, i - 1, j - 1), q))
        spd[per] = items
    return spd


def write_mf6_wel(
    output_path: Path,
    spd: Dict[int, List[Tuple[Tuple[int, int, int], float]]],
    nlay: int,
    nrow: int,
    ncol: int,
) -> None:
    try:
        import flopy
    except ImportError as exc:
        raise SystemExit("FloPy is required. Install with: pip install flopy") from exc

    out_dir = output_path.parent
    sim = flopy.mf6.MFSimulation(sim_name="wel_update", version="mf6", sim_ws=str(out_dir))
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(spd),
        perioddata=[(1.0, 1, 1.0)] * len(spd),
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf")
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=[0.0] * nlay,
    )
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=spd,
        filename=output_path.name,
    )
    wel.write()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update MF2005 WEL and write MF6 WEL using FloPy.")
    parser.add_argument("--input", required=True, type=Path, help="Input MODFLOW-2005 WEL file")
    parser.add_argument("--output", required=True, type=Path, help="Output MF6 WEL file")
    parser.add_argument("--multiplier", type=float, help="Multiply all pumping rates by this value")
    parser.add_argument("--set-rate", type=float, help="Set all pumping rates to this value")
    parser.add_argument("--updates-csv", type=Path, help="CSV with columns k,i,j,q to update wells")
    parser.add_argument("--add-missing", action="store_true", help="Add wells from CSV if not present")
    parser.add_argument("--nlay", type=int, help="Number of layers (default: max from WEL data)")
    parser.add_argument("--nrow", type=int, help="Number of rows (default: max from WEL data)")
    parser.add_argument("--ncol", type=int, help="Number of columns (default: max from WEL data)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nper, max_k, max_i, max_j = scan_wel_metadata(args.input)
    nlay = args.nlay or max_k
    nrow = args.nrow or max_i
    ncol = args.ncol or max_j
    per_data = read_mf2005_wel(args.input, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)
    updates = load_updates_csv(args.updates_csv) if args.updates_csv else None
    per_data = apply_updates(per_data, args.multiplier, args.set_rate, updates, args.add_missing)
    spd = to_mf6_spd(per_data)
    write_mf6_wel(args.output, spd, nlay, nrow, ncol)


if __name__ == "__main__":
    main()
