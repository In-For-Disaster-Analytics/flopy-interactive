#!/usr/bin/env python3
"""Quick import check for the environment."""

import importlib
import sys


MODULES = [
    "flopy",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "geopandas",
    "shapely",
    "pyproj",
    "rasterio",
    "netCDF4",
    "ipywidgets",
    "plotly",
]


def main() -> None:
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"OK: {name}")
        except Exception as exc:  # pragma: no cover - quick diagnostic
            failures.append((name, exc))
            print(f"FAIL: {name} -> {exc}")

    if failures:
        print("\nSome imports failed.")
        sys.exit(1)

    print("\nAll imports succeeded.")


if __name__ == "__main__":
    main()
