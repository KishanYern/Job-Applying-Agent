"""
RunPod Serverless: bump workersMin before a job, restore after.

Uses the public REST API (same API key as inference in most accounts):
  PATCH https://rest.runpod.io/v1/endpoints/{endpointId}
  Body: {"workersMin": <int>}

Enable with env:
  RUNPOD_MANAGE_WORKERS=1
Optional:
  RUNPOD_ENDPOINT_ID=...     (if not set, parsed from RUNPOD_ENDPOINT_URL)
  RUNPOD_WORKERS_MIN_ACTIVE=1  (workersMin while the agent runs)
  RUNPOD_WORKERS_MIN_IDLE=0   (workersMin after the run — scale-to-zero)
  RUNPOD_POST_SCALE_UP_WAIT_SECONDS=0  (extra sleep after PATCH; GPU may still need 1–3+ min — warm-up in apply_agent handles that)

Note: updating an endpoint can trigger a rolling release (RunPod docs). Do not run
multiple concurrent applies against the same endpoint if both toggle workersMin.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from typing import Callable, Optional

import httpx

LogFn = Callable[[str, str], None]

RUNPOD_REST_BASE = os.environ.get("RUNPOD_REST_BASE", "https://rest.runpod.io/v1")


def parse_endpoint_id_from_openai_url(url: str) -> Optional[str]:
    """Extract serverless endpoint id from OpenAI-compatible base URL."""
    if not url or not url.strip():
        return None
    # https://api.runpod.ai/v2/<endpointId>/openai/v1
    m = re.search(r"/v2/([^/]+)/", url.strip())
    return m.group(1) if m else None


def resolve_endpoint_id() -> Optional[str]:
    explicit = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if explicit:
        return explicit
    return parse_endpoint_id_from_openai_url(os.environ.get("RUNPOD_ENDPOINT_URL", ""))


def manage_workers_enabled() -> bool:
    v = os.environ.get("RUNPOD_MANAGE_WORKERS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def workers_min_active() -> int:
    try:
        return max(0, int(os.environ.get("RUNPOD_WORKERS_MIN_ACTIVE", "1")))
    except ValueError:
        return 1


def workers_min_idle() -> int:
    try:
        return max(0, int(os.environ.get("RUNPOD_WORKERS_MIN_IDLE", "0")))
    except ValueError:
        return 0


def post_scale_up_wait_seconds() -> float:
    """Optional delay after PATCH to workersMin=active (GPU boot can still take minutes after)."""
    try:
        return max(0.0, float(os.environ.get("RUNPOD_POST_SCALE_UP_WAIT_SECONDS", "0") or 0))
    except ValueError:
        return 0.0


async def set_endpoint_workers_min(
    *,
    api_key: str,
    endpoint_id: str,
    workers_min: int,
    timeout: float = 45.0,
) -> None:
    """PATCH workersMin on the serverless endpoint."""
    url = f"{RUNPOD_REST_BASE.rstrip('/')}/endpoints/{endpoint_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"workersMin": max(0, int(workers_min))}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.patch(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"RunPod PATCH {url} -> HTTP {resp.status_code}: {resp.text[:500]}"
            )


@asynccontextmanager
async def managed_runpod_workers(log: LogFn):
    """
    If RUNPOD_MANAGE_WORKERS is set, PATCH workersMin to RUNPOD_WORKERS_MIN_ACTIVE
    before yield, then restore RUNPOD_WORKERS_MIN_IDLE in finally.
    """
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not manage_workers_enabled():
        yield
        return
    if not api_key:
        log("RUNPOD_MANAGE_WORKERS is set but RUNPOD_API_KEY is empty — skipping worker control", "warning")
        yield
        return
    endpoint_id = resolve_endpoint_id()
    if not endpoint_id:
        log(
            "RUNPOD_MANAGE_WORKERS is set but endpoint id missing — set RUNPOD_ENDPOINT_ID "
            "or use RUNPOD_ENDPOINT_URL like .../v2/<id>/openai/v1",
            "warning",
        )
        yield
        return

    scaled = False
    try:
        await set_endpoint_workers_min(
            api_key=api_key,
            endpoint_id=endpoint_id,
            workers_min=workers_min_active(),
        )
        scaled = True
        log(
            f"RunPod endpoint {endpoint_id}: workersMin={workers_min_active()} (active for this run)",
            "info",
        )
        log(
            "Model load after increasing workers can take 1–3+ minutes; the agent warm-up ping "
            "and LLM retries allow that startup time.",
            "info",
        )
        wait_s = post_scale_up_wait_seconds()
        if wait_s > 0:
            log(f"Waiting {wait_s:g}s (RUNPOD_POST_SCALE_UP_WAIT_SECONDS) after scale-up...", "info")
            await asyncio.sleep(wait_s)
    except Exception as e:
        log(f"RunPod scale-up failed (continuing; expect cold start): {e}", "warning")

    try:
        yield
    finally:
        if scaled:
            try:
                await set_endpoint_workers_min(
                    api_key=api_key,
                    endpoint_id=endpoint_id,
                    workers_min=workers_min_idle(),
                )
                log(
                    f"RunPod endpoint {endpoint_id}: workersMin={workers_min_idle()} (scaled down after run)",
                    "info",
                )
            except Exception as e:
                log(f"RunPod scale-down failed — check workers in console: {e}", "warning")
