"""Helpers for registering and running Tapis Workflows pipelines."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

from flopy_interactive.workflow_definition import build_flux_percent_pipeline


WORKFLOWS_BASE_URL = os.environ.get("FLOPY_WORKFLOWS_BASE_URL", "https://tacc.tapis.io/v3/workflows").rstrip("/")


def _headers(token: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Tapis-Token": token,
    }


def _unwrap(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


def _request(method: str, path: str, token: str, *, json_body: Dict[str, Any] | None = None, accept_not_found: bool = False) -> Dict[str, Any] | None:
    response = requests.request(
        method,
        f"{WORKFLOWS_BASE_URL}{path}",
        headers=_headers(token),
        json=json_body,
        timeout=120,
    )
    if accept_not_found and response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("message") or json.dumps(payload))
    return _unwrap(payload)


def get_pipeline(group_id: str, pipeline_id: str, token: str) -> Dict[str, Any] | None:
    return _request("GET", f"/groups/{group_id}/pipelines/{pipeline_id}", token, accept_not_found=True)


def create_pipeline(group_id: str, pipeline: Dict[str, Any], token: str) -> Dict[str, Any]:
    return _request("POST", f"/groups/{group_id}/pipelines", token, json_body=pipeline) or {}


def ensure_pipeline(group_id: str, pipeline_id: str, token: str, archive_ids: list[str] | None = None) -> Dict[str, Any]:
    existing = get_pipeline(group_id, pipeline_id, token)
    if existing is not None:
        return existing
    return create_pipeline(group_id, build_flux_percent_pipeline(pipeline_id, archive_ids=archive_ids), token)


def run_pipeline(group_id: str, pipeline_id: str, token: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return _request("POST", f"/groups/{group_id}/pipelines/{pipeline_id}/run", token, json_body={"args": args}) or {}


def _run_paths(group_id: str, pipeline_id: str, run_id: str) -> list[tuple[str, str]]:
    return [
        ("GET", f"/groups/{group_id}/pipelines/{pipeline_id}/runs/{run_id}"),
        ("GET", f"/groups/{group_id}/pipeline/{pipeline_id}/runs/{run_id}"),
        ("POST", f"/groups/{group_id}/pipelines/{pipeline_id}/runs/{run_id}"),
        ("POST", f"/groups/{group_id}/pipeline/{pipeline_id}/runs/{run_id}"),
    ]


def get_pipeline_run(group_id: str, pipeline_id: str, run_id: str, token: str) -> Dict[str, Any]:
    last_error: Exception | None = None
    for method, path in _run_paths(group_id, pipeline_id, run_id):
        try:
            payload = _request(method, path, token, json_body={} if method == "POST" else None)
        except Exception as exc:  # pragma: no cover - fallback chain
            last_error = exc
            continue
        if payload is not None:
            return payload
    raise RuntimeError(f"Failed to fetch pipeline run {run_id}: {last_error}")


def extract_run_id(submission: Dict[str, Any]) -> str:
    candidates = [
        submission.get("uuid"),
        submission.get("id"),
        submission.get("run_id"),
        submission.get("runId"),
        submission.get("current_run"),
    ]
    for value in candidates:
        if value:
            return str(value)
    result = submission.get("result")
    if isinstance(result, dict):
        for key in ("uuid", "id", "run_id", "runId", "current_run"):
            value = result.get(key)
            if value:
                return str(value)
    raise RuntimeError(f"Workflow run id missing from submission payload: {submission}")


def extract_run_status(run_payload: Dict[str, Any]) -> str:
    candidates = [
        run_payload.get("status"),
        run_payload.get("state"),
        run_payload.get("phase"),
    ]
    for value in candidates:
        if value:
            return str(value)
    for key in ("tasks", "task_runs", "taskRuns"):
        tasks = run_payload.get(key)
        if isinstance(tasks, list) and tasks:
            first = tasks[0]
            if isinstance(first, dict):
                for task_key in ("status", "state", "phase"):
                    value = first.get(task_key)
                    if value:
                        return str(value)
    return "UNKNOWN"


def extract_result_payload(run_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    def walk(value: Any) -> Dict[str, Any] | None:
        if isinstance(value, dict):
            for key in ("result_json", "stdout"):
                raw = value.get(key)
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        return parsed
            for nested in value.values():
                parsed = walk(nested)
                if parsed is not None:
                    return parsed
        elif isinstance(value, list):
            for item in value:
                parsed = walk(item)
                if parsed is not None:
                    return parsed
        return None

    return walk(run_payload)
