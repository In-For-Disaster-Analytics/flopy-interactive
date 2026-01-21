"""Search CKAN datasets and standard variable metadata."""

from __future__ import annotations

import json
from typing import Dict, List
from urllib.request import urlopen

from flopy_interactive.config import (
    CKAN_BASE_URL,
    GRID_STANDARD_VAR,
    RCH_STANDARD_VAR,
    WEL_STANDARD_VAR,
)


def extract_standard_vars(resource: Dict) -> List[str]:
    """Extract MINT standard variable values from CKAN resource metadata.

    Args:
        resource: CKAN resource metadata dict.

    Returns:
        List of standard variable strings.
    """
    candidates: List[str] = []
    direct_keys = ("MINT Standard Variables", "mint_standard_variables", "standard_variables")
    for key in direct_keys:
        if key in resource and resource[key] is not None:
            candidates.append(resource[key])
    extras = resource.get("extras")
    if isinstance(extras, list):
        for item in extras:
            if not isinstance(item, dict):
                continue
            if item.get("key") == "MINT Standard Variables":
                candidates.append(item.get("value"))
    elif isinstance(extras, dict) and "MINT Standard Variables" in extras:
        candidates.append(extras.get("MINT Standard Variables"))

    values: List[str] = []
    for entry in candidates:
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            values.extend([str(v) for v in entry])
            continue
        if isinstance(entry, str):
            text = entry.strip()
            if not text:
                continue
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        values.extend([str(v) for v in parsed])
                        continue
                except json.JSONDecodeError:
                    pass
            values.extend([v.strip() for v in text.replace(";", ",").split(",") if v.strip()])
            continue
        values.append(str(entry))
    return values


def resource_has_standard_var(resource: Dict, target: str) -> bool:
    """Return True if a resource advertises the target standard variable.

    Args:
        resource: CKAN resource metadata dict.
        target: Standard variable to match (case-insensitive).

    Returns:
        True if the resource contains the target variable.
    """
    target_key = target.strip().lower()
    for value in extract_standard_vars(resource):
        if value.strip().lower() == target_key:
            return True
    return False


def search_ckan_datasets() -> List[Dict]:
    """Return CKAN datasets that have WEL, RCH, and grid resources.

    Args:
        None.

    Returns:
        List of dataset dicts with ``name``, ``title``, and ``matches``.
    """
    base_url = f"{CKAN_BASE_URL}/api/3/action/package_search"
    start = 0
    rows = 100
    matched: List[Dict] = []
    targets = {
        "wel": WEL_STANDARD_VAR,
        "grid": GRID_STANDARD_VAR,
        "rch": RCH_STANDARD_VAR,
    }

    while True:
        url = f"{base_url}?rows={rows}&start={start}"
        with urlopen(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError("CKAN search failed.")
        result = payload["result"]
        results = result.get("results", [])
        for pkg in results:
            resources = pkg.get("resources", [])
            matches = {"wel": [], "grid": [], "rch": []}
            for res in resources:
                if resource_has_standard_var(res, targets["wel"]):
                    matches["wel"].append(res)
                if resource_has_standard_var(res, targets["grid"]):
                    matches["grid"].append(res)
                if resource_has_standard_var(res, targets["rch"]):
                    matches["rch"].append(res)
            if all(matches[key] for key in matches):
                matched.append(
                    {
                        "name": pkg.get("name") or pkg.get("id"),
                        "title": pkg.get("title") or pkg.get("name") or pkg.get("id"),
                        "matches": matches,
                    }
                )
        start += rows
        if start >= result.get("count", 0):
            break
    return matched


def search_ckan_datasets_wel_rch(no_grid: bool = False) -> List[Dict]:
    """Return CKAN datasets that have WEL or RCH resources.

    Args:
        no_grid: When True, exclude datasets that also have grid resources.

    Returns:
        List of dataset dicts with ``name``, ``title``, and ``matches``.
    """
    base_url = f"{CKAN_BASE_URL}/api/3/action/package_search"
    start = 0
    rows = 100
    matched: List[Dict] = []
    targets = {
        "wel": WEL_STANDARD_VAR,
        "rch": RCH_STANDARD_VAR,
        "grid": GRID_STANDARD_VAR,
    }

    while True:
        url = f"{base_url}?rows={rows}&start={start}"
        with urlopen(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError("CKAN search failed.")
        result = payload["result"]
        results = result.get("results", [])
        for pkg in results:
            resources = pkg.get("resources", [])
            matches = {"wel": [], "rch": [], "grid": []}
            for res in resources:
                if resource_has_standard_var(res, targets["wel"]):
                    matches["wel"].append(res)
                if resource_has_standard_var(res, targets["rch"]):
                    matches["rch"].append(res)
                if resource_has_standard_var(res, targets["grid"]):
                    matches["grid"].append(res)
            has_target = bool(matches["wel"] or matches["rch"])
            if not has_target:
                continue
            if no_grid and matches["grid"]:
                continue
            matched.append(
                {
                    "name": pkg.get("name") or pkg.get("id"),
                    "title": pkg.get("title") or pkg.get("name") or pkg.get("id"),
                    "matches": matches,
                }
            )
        start += rows
        if start >= result.get("count", 0):
            break
    return matched
