# FloPy Dash App

Dash dashboard for visualizing WEL/RCH data, selecting cells on a map, applying rate updates, and publishing updated WEL files back to CKAN with provenance metadata.

## Quick start

1. Create a virtual environment.
2. Install dependencies:

```
pip install -r dash/requirements-dash.txt
```

3. Run the app:

```
python dash/dash_app.py
```

The app listens on http://localhost:8050 by default.

## Environment variables

- `FLOPY_DATA_DIR`: Directory where CKAN resources are downloaded (default: `ckan_data`).
- `FLOPY_OUTPUT_WEL`: Default output WEL path (default: `barton_springs_updated.wel`).
- `FLOPY_CKAN_URL`: CKAN base URL (default: `https://ckan.tacc.utexas.edu`).
- `FLOPY_CKAN_JWT`: CKAN JWT to skip login flow.
- `FLOPY_TAPIS_USERNAME`: Tapis username (used if `FLOPY_CKAN_JWT` is not set).
- `FLOPY_TAPIS_PASSWORD`: Tapis password (used if `FLOPY_CKAN_JWT` is not set).

## App flow

- Select a dataset, flux source (WEL or RCH), and stress periods.
- Lasso or click cells on the map to build a selection.
- Set a rate update mode (set or scale) and apply changes.
- Provide a dataset name, output filename, and change summary.
- Click Apply + Save to write the updated WEL and publish to CKAN.

## Files

- `dash/dash_app.py`: Dash UI and callbacks.
- `dash/ckan_publish.py`: CKAN/Tapis helpers and publish logic.
- `dash/assets/style.css`: Dash styling.
