#!/usr/bin/env python3
"""Publish updated WEL outputs back to CKAN with provenance metadata."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from tapipy.tapis import Tapis

import flopy_wel_map as fwm

CKAN_URL = os.environ.get("FLOPY_CKAN_URL", "https://ckan.tacc.utexas.edu")
WEL_STANDARD_VAR = "groundwater_well__recharge_volume_flux"


def _now_iso() -> str:
    """Return current UTC time in ISO8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    """Normalize a string for CKAN dataset/resource naming."""
    value = value.strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-_.]", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "dataset"


def _extras_to_dict(extras: Iterable[Dict]) -> Dict[str, str]:
    """Convert CKAN extras list into a key/value dict."""
    result: Dict[str, str] = {}
    for item in extras or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        value = item.get("value")
        if key:
            result[key] = "" if value is None else str(value)
    return result


def _merge_extras(existing: Iterable[Dict], additions: Dict[str, str]) -> List[Dict]:
    """Merge extras with updates, returning CKAN list-of-dict format."""
    merged = _extras_to_dict(existing)
    merged.update({k: v for k, v in additions.items() if v is not None})
    return [{"key": key, "value": value} for key, value in merged.items()]


def _extract_mint_svo(extras: Iterable[Dict]) -> Optional[str]:
    """Find the MINT SVO value from known extras keys."""
    candidates = [
        "MINT_SVO",
        "MINT Standard Variables",
        "mint_standard_variables",
        "standard_variables",
    ]
    extras_dict = _extras_to_dict(extras)
    for key in candidates:
        if key in extras_dict and extras_dict[key]:
            return extras_dict[key]
    return None


def _resolve_mint_svo(resource: Dict, dataset: Dict) -> Optional[str]:
    """Resolve the MINT SVO from resource or dataset metadata."""
    mint_svo = _extract_mint_svo(resource.get("extras", []))
    if mint_svo:
        return mint_svo
    if fwm._resource_has_standard_var(resource, WEL_STANDARD_VAR):
        return WEL_STANDARD_VAR
    mint_svo = _extract_mint_svo(dataset.get("extras", []))
    if mint_svo:
        return mint_svo
    for value in fwm._extract_standard_vars(dataset):
        if value.strip().lower() == WEL_STANDARD_VAR.lower():
            return WEL_STANDARD_VAR
    return None


def get_tapis_token(username: str, password: str) -> str:
    """Authenticate to Tapis and return the access token."""
    tapis = Tapis(base_url="https://portals.tapis.io", username=username, password=password)
    tapis.get_tokens()
    return tapis.access_token.access_token


def get_jwt_token() -> str:
    """Return a CKAN JWT from env or Tapis credentials."""
    jwt_token = os.environ.get("FLOPY_CKAN_JWT", "").strip()
    if jwt_token:
        return jwt_token
    username = os.environ.get("FLOPY_TAPIS_USERNAME", "").strip()
    password = os.environ.get("FLOPY_TAPIS_PASSWORD", "").strip()
    if not username or not password:
        raise ValueError("Missing CKAN JWT or Tapis credentials.")
    return get_tapis_token(username, password)


def _headers(jwt_token: str) -> Dict[str, str]:
    """Build auth headers for CKAN API requests."""
    return {"Authorization": f"Bearer {jwt_token}"}


def package_show(jwt_token: str, dataset_name: str) -> Dict:
    """Fetch CKAN dataset metadata by name."""
    url = f"{CKAN_URL}/api/3/action/package_show"
    response = requests.get(url, params={"id": dataset_name}, headers=_headers(jwt_token), timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {payload}")
    return payload["result"]


def create_dataset(jwt_token: str, dataset_dict: Dict) -> Dict:
    """Create a new CKAN dataset."""
    url = f"{CKAN_URL}/api/3/action/package_create"
    response = requests.post(url, json=dataset_dict, headers=_headers(jwt_token), timeout=60)
    if response.status_code == 409:
        raise RuntimeError("dataset name conflict")
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_create failed: {payload}")
    return payload["result"]


def create_resource_upload(
    jwt_token: str,
    dataset_id: str,
    file_path: Path,
    resource_dict: Dict,
) -> Dict:
    """Upload a file as a CKAN resource."""
    url = f"{CKAN_URL}/api/3/action/resource_create"
    data = {
        "package_id": dataset_id,
        "name": resource_dict.get("name", file_path.name),
        "description": resource_dict.get("description", ""),
        "format": resource_dict.get("format", file_path.suffix.lstrip(".").upper() or "WEL"),
    }
    extras = resource_dict.get("extras")
    if extras:
        data["extras"] = json.dumps(extras)
    with file_path.open("rb") as handle:
        files = {"upload": handle}
        response = requests.post(url, data=data, files=files, headers=_headers(jwt_token), timeout=300)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN resource_create failed: {payload}")
    return payload["result"]


def build_dataset_payload(
    source_dataset: Dict,
    new_name: str,
    provenance: Dict[str, str],
    mint_svo: str,
    source_url: str | None = None,
    change_summary: str | None = None,
    maintainer_username: str | None = None,
) -> Dict:
    """Create a dataset payload derived from a source dataset."""
    copy_fields = [
        "title",
        "notes",
        "owner_org",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "license_id",
        "private",
        "url",
        "version",
        "spatial",
        "temporal_coverage_start",
        "temporal_coverage_end",
        "groups",
    ]
    dataset: Dict = {field: source_dataset.get(field) for field in copy_fields if source_dataset.get(field) is not None}
    dataset["name"] = new_name
    dataset["title"] = f"{source_dataset.get('title', new_name)} (updated {_now_iso()})"
    dataset["tags"] = [{"name": t["name"]} for t in source_dataset.get("tags", []) if isinstance(t, dict) and t.get("name")]
    extras = source_dataset.get("extras", [])
    extras = _merge_extras(
        extras,
        {
            "MINT_SVO": mint_svo,
            "provenance_timestamp": provenance.get("timestamp", _now_iso()),
            "provenance_action": provenance.get("action", ""),
            "provenance_details": provenance.get("details", ""),
            "derived_from_dataset_id": provenance.get("source_dataset_id", ""),
            "derived_from_dataset_name": provenance.get("source_dataset_name", ""),
            "source_url": source_url or "",
            "change_summary": change_summary or "",
        },
    )
    dataset["extras"] = extras
    if change_summary:
        dataset["notes"] = (dataset.get("notes") or "") + f"\n\nChanges: {change_summary}"
    if maintainer_username:
        dataset["maintainer"] = maintainer_username
    return dataset


def build_resource_payload(
    source_resource: Dict,
    provenance: Dict[str, str],
    mint_svo: str,
    source_url: str | None = None,
    change_summary: str | None = None,
) -> Dict:
    """Create a resource payload derived from a source resource."""
    extras = source_resource.get("extras", [])
    extras = _merge_extras(
        extras,
        {
            "MINT_SVO": mint_svo,
            "provenance_timestamp": provenance.get("timestamp", _now_iso()),
            "provenance_action": provenance.get("action", ""),
            "provenance_details": provenance.get("details", ""),
            "derived_from_resource_id": provenance.get("source_resource_id", ""),
            "derived_from_resource_name": provenance.get("source_resource_name", ""),
            "source_url": source_url or "",
            "change_summary": change_summary or "",
        },
    )
    description = source_resource.get("description", "")
    if source_url:
        description = f"{description}\nMetadata This file comes from {source_url}".strip()
    if change_summary:
        description = f"{description}\nMetadata Description of Changes Made: {change_summary}".strip()
    return {
        "name": f"{source_resource.get('name', 'WEL')} (updated)",
        "description": description,
        "format": source_resource.get("format", "WEL"),
        "extras": extras,
    }


def publish_updated_wel(
    source_dataset_name: str,
    output_path: Path,
    provenance_details: Dict,
    jwt_token: Optional[str] = None,
    new_dataset_name: Optional[str] = None,
    source_url: Optional[str] = None,
    change_summary: Optional[str] = None,
    maintainer_username: Optional[str] = None,
) -> Dict:
    """Create or update a derived dataset and upload the updated WEL file."""
    jwt_token = jwt_token or get_jwt_token()
    source_dataset = package_show(jwt_token, source_dataset_name)
    resources = source_dataset.get("resources", [])
    wel_resource = next(
        (res for res in resources if fwm._resource_has_standard_var(res, WEL_STANDARD_VAR)),
        None,
    )
    if not wel_resource:
        raise RuntimeError("Could not find WEL resource in source dataset.")
    mint_svo = _resolve_mint_svo(wel_resource, source_dataset)
    if not mint_svo:
        raise RuntimeError("MINT_SVO missing in source metadata.")
    timestamp = _now_iso()
    if new_dataset_name:
        new_name = _slugify(new_dataset_name)
    else:
        new_name = _slugify(f"{source_dataset_name}-updated-{timestamp}")
    provenance = {
        "timestamp": timestamp,
        "action": "wel_rate_update",
        "details": json.dumps(provenance_details, sort_keys=True),
        "source_dataset_id": source_dataset.get("id", ""),
        "source_dataset_name": source_dataset_name,
        "source_resource_id": wel_resource.get("id", ""),
        "source_resource_name": wel_resource.get("name", ""),
    }
    if new_name == source_dataset_name:
        raise RuntimeError("dataset name conflict with original; choose a new name")

    existing_dataset = None
    try:
        existing_dataset = package_show(jwt_token, new_name)
    except Exception:
        existing_dataset = None

    if existing_dataset:
        if maintainer_username:
            existing_maintainer = str(existing_dataset.get("maintainer", "")).strip().lower()
            existing_email = str(existing_dataset.get("maintainer_email", "")).strip().lower()
            maintainer_key = maintainer_username.strip().lower()
            if existing_maintainer and maintainer_key != existing_maintainer and maintainer_key != existing_email:
                raise RuntimeError("dataset maintainer mismatch")
        dataset_id = existing_dataset["id"]
        dataset_payload = existing_dataset
    else:
        dataset_payload = build_dataset_payload(
            source_dataset,
            new_name,
            provenance,
            mint_svo,
            source_url=source_url,
            change_summary=change_summary,
            maintainer_username=maintainer_username,
        )
        dataset_payload["name"] = new_name
        created_dataset = create_dataset(jwt_token, dataset_payload)
        dataset_id = created_dataset["id"]
        existing_dataset = created_dataset

    resource_payload = build_resource_payload(
        wel_resource,
        provenance,
        mint_svo,
        source_url=source_url,
        change_summary=change_summary,
    )
    created_resource = create_resource_upload(
        jwt_token,
        dataset_id,
        output_path,
        resource_payload,
    )
    return {"dataset": existing_dataset, "resource": created_resource}
