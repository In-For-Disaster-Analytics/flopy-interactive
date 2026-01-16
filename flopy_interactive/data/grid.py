"""Load and normalize model grid data."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd

from flopy_interactive.data.download import extract_zip, find_grid_data_path, download_ckan_resource

warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)


def load_grid_gdf(grid_gdb: Path, layer_name: str) -> gpd.GeoDataFrame:
    """Load a named layer from a file geodatabase.

    Args:
        grid_gdb: Path to the geodatabase.
        layer_name: Layer name to load.

    Returns:
        GeoDataFrame with normalized CRS and centroid columns.
    """
    gdf = gpd.read_file(grid_gdb, layer=layer_name)
    return _prepare_grid_gdf(gdf)


def _prepare_grid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize CRS and add lon/lat centroid columns.

    Args:
        gdf: GeoDataFrame with geometry and CRS.

    Returns:
        GeoDataFrame with ``_lon`` and ``_lat`` columns in EPSG:4326.
    """
    if gdf.crs is None:
        centroids = gdf.geometry.centroid
        gdf["_lon"] = centroids.x
        gdf["_lat"] = centroids.y
        return gdf

    try:
        if gdf.crs.is_geographic:
            projected = gdf.to_crs(gdf.estimate_utm_crs())
        else:
            projected = gdf
    except Exception:
        projected = gdf

    centroids = projected.geometry.centroid
    try:
        centroids_ll = centroids.to_crs("EPSG:4326")
    except Exception:
        centroids_ll = centroids

    gdf["_lon"] = centroids_ll.x
    gdf["_lat"] = centroids_ll.y
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _load_grid_from_gdb(gdb_path: Path) -> gpd.GeoDataFrame:
    """Load the first layer from a file geodatabase.

    Args:
        gdb_path: Path to the geodatabase.

    Returns:
        GeoDataFrame for the first layer.
    """
    try:
        import fiona
    except ImportError as exc:
        raise ImportError("fiona is required to read a geodatabase.") from exc
    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"No layers found in {gdb_path}.")
    gdf = gpd.read_file(gdb_path, layer=layers[0])
    return _prepare_grid_gdf(gdf)


def load_grid_resource(resource: dict, dest_dir: Path) -> gpd.GeoDataFrame:
    """Download and load a grid resource from CKAN.

    Args:
        resource: CKAN resource metadata dict.
        dest_dir: Destination directory for the download.

    Returns:
        GeoDataFrame for the grid resource.
    """
    path = download_ckan_resource(resource, dest_dir)
    if path.suffix.lower() == ".zip":
        path = extract_zip(path)
    grid_path = find_grid_data_path(path)
    if grid_path is None:
        raise ValueError("No grid dataset found after extracting resource.")
    if grid_path.suffix.lower() == ".gdb":
        return _load_grid_from_gdb(grid_path)
    gdf = gpd.read_file(grid_path)
    return _prepare_grid_gdf(gdf)
