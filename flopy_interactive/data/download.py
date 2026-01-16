"""Download and extract CKAN resources."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict


def _flatten_single_dir(root: Path) -> None:
    """Collapse a single nested directory into its parent.

    Args:
        root: Directory to inspect and flatten.

    Returns:
        None.
    """
    children = [p for p in root.iterdir() if p.is_dir()]
    if len(children) == 1:
        inner = children[0]
        for item in inner.iterdir():
            shutil.move(str(item), root)
        inner.rmdir()


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
    _flatten_single_dir(extract_dir)
    return extract_dir


def find_grid_data_path(root: Path) -> Path | None:
    """Locate a grid dataset within a folder or return the file path.

    Args:
        root: Directory or file path to search.

    Returns:
        Path to the grid data file, or None if not found.
    """
    if root.is_dir():
        gdbs = list(root.rglob("*.gdb"))
        if gdbs:
            return gdbs[0]
        for ext in (".shp", ".geojson", ".gpkg", ".json"):
            matches = list(root.rglob(f"*{ext}"))
            if matches:
                return matches[0]
        return None
    return root
