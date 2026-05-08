"""Helpers for dispatching ETL work to Tapis Actors."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests


ACTORS_BASE_URL = os.environ.get("FLOPY_ACTORS_BASE_URL", "https://tacc.tapis.io/v3/actors").rstrip("/")


def _headers(token: str) -> Dict[str, str]:
    return {"X-Tapis-Token": token}


def _unwrap(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def submit_execution(actor_id: str, token: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON message to an actor and return execution metadata."""
    response = requests.post(
        f"{ACTORS_BASE_URL}/{actor_id}/messages",
        headers=_headers(token),
        data={"message": json.dumps(message)},
        timeout=60,
    )
    response.raise_for_status()
    payload = _unwrap(response.json())
    execution_id = payload.get("execution_id") or payload.get("id")
    return {
        "actorId": actor_id,
        "executionId": execution_id,
        "payload": payload,
    }


def get_execution(actor_id: str, execution_id: str, token: str) -> Dict[str, Any]:
    response = requests.get(
        f"{ACTORS_BASE_URL}/{actor_id}/executions/{execution_id}",
        headers=_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return _unwrap(response.json())


def get_logs(actor_id: str, execution_id: str, token: str) -> str:
    response = requests.get(
        f"{ACTORS_BASE_URL}/{actor_id}/executions/{execution_id}/logs",
        headers=_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    payload = _unwrap(response.json())
    logs = payload.get("logs")
    if isinstance(logs, str):
        return logs
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def get_result(actor_id: str, execution_id: str, token: str) -> Any:
    response = requests.get(
        f"{ACTORS_BASE_URL}/{actor_id}/executions/{execution_id}/results",
        headers=_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except Exception:
            return response.text
        return _unwrap(payload)
    text = response.text
    try:
        return json.loads(text)
    except Exception:
        return text
