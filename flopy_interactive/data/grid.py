"""Load and normalize model grid data."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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
    layers = []
    try:
        layers = fiona.listlayers(gdb_path, driver="OpenFileGDB")
    except Exception:
        layers = []
    if not layers:
        layers = fiona.listlayers(gdb_path)
    if not layers:
        raise ValueError(f"No layers found in {gdb_path}.")
    try:
        gdf = gpd.read_file(gdb_path, layer=layers[0], driver="OpenFileGDB")
    except Exception:
        gdf = gpd.read_file(gdb_path, layer=layers[0])
    return _prepare_grid_gdf(gdf)


def _load_grid_from_csv(csv_path: Path) -> gpd.GeoDataFrame:
    """Load a grid CSV and construct lon/lat columns.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        GeoDataFrame with ``_lon`` and ``_lat`` columns.
    """
    df = pd.read_csv(csv_path)
    columns = {col.lower(): col for col in df.columns}
    lon_key = columns.get("centroidx") or columns.get("node_x")
    lat_key = columns.get("centroidy") or columns.get("node_y")
    if lon_key is None or lat_key is None:
        raise ValueError("CSV grid missing centroid/node coordinate columns.")
    df["_lon"] = pd.to_numeric(df[lon_key], errors="coerce")
    df["_lat"] = pd.to_numeric(df[lat_key], errors="coerce")
    # Heuristic: swap if lon/lat ranges look flipped.
    lon_abs = df["_lon"].abs().max()
    lat_abs = df["_lat"].abs().max()
    if (lon_abs <= 90 and lat_abs > 90) or (lon_abs > 180 and lat_abs <= 180):
        df["_lon"], df["_lat"] = df["_lat"], df["_lon"]
    geometry = []
    for lon, lat in zip(df["_lon"], df["_lat"]):
        if pd.isna(lon) or pd.isna(lat):
            geometry.append(None)
        else:
            geometry.append(Point(float(lon), float(lat)))
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    return gdf


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
    if grid_path.is_dir() or grid_path.suffix.lower() == ".gdb":
        return _load_grid_from_gdb(grid_path)
    if grid_path.suffix.lower() == ".csv":
        return _load_grid_from_csv(grid_path)
    gdf = gpd.read_file(grid_path)
    return _prepare_grid_gdf(gdf)
