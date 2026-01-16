"""Sample data bootstrapping helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

from flopy_interactive.data.download import _flatten_single_dir


def ensure_barton_springs_wel(wel_path: Path) -> None:
    """Download the Barton Springs WEL file if it is missing.

    Args:
        wel_path: Local path for the WEL file.

    Returns:
        None.
    """
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
    """Download and unzip the EBFZ grid geodatabase if needed.

    Args:
        grid_dir: Directory to place the grid data.

    Returns:
        Path to the extracted geodatabase.
    """
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
    if not grid_zip.exists():
        urlretrieve(url, grid_zip)
    grid_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(grid_zip, "r") as zf:
        zf.extractall(grid_dir)
    _flatten_single_dir(grid_dir)
    return grid_gdb
