"""Download and extract CKAN resources."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict


def download_ckan_resource(resource: Dict, dest_dir: Path) -> Path:
    """Download a CKAN resource to a local directory.

    Args:
        resource: CKAN resource metadata dict with a ``url``.
        dest_dir: Destination directory for the download.

    Returns:
        Path to the downloaded file.
    """
    url = resource.get("url")
    if not url:
        raise ValueError("CKAN resource missing URL.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?", 1)[0]).name
    if not filename:
        filename = resource.get("name") or resource.get("id") or "resource"
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    from urllib.request import urlretrieve

    urlretrieve(url, dest_path)
    return dest_path


def _flatten_single_dir(root: Path) -> None:
    """Flatten a single nested directory in place."""
    if not root.is_dir():
        return
    entries = list(root.iterdir())
    subdirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    if files or len(subdirs) != 1:
        return
    nested = subdirs[0]
    for entry in nested.iterdir():
        shutil.move(str(entry), root / entry.name)
    nested.rmdir()


def extract_zip(zip_path: Path) -> Path:
    """Extract a zip to a folder alongside the archive.

    Args:
        zip_path: Path to the zip archive.

    Returns:
        Path to the extracted directory.
    """
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_grid_data_path(root: Path) -> Path | None:
    """Locate a grid dataset within a folder or return the file path.

    Args:
        root: Directory or file path to search.

    Returns:
        Path to the grid data file, or None if not found.
    """
    if root.is_dir():
        if (root / "gdb").exists() or list(root.glob("*.gdbtable")):
            return root
        for path in root.rglob("*"):
            if path.suffix.lower() == ".gdb":
                return path
        for path in root.rglob("*"):
            if path.is_dir():
                if (path / "gdb").exists():
                    return path
                if list(path.glob("*.gdbtable")):
                    return path
        for path in root.rglob("*"):
            if path.suffix.lower() in (".shp", ".geojson", ".gpkg", ".json"):
                return path
        return None
    return root
