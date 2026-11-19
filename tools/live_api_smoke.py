"""Run one explicit, billable FastAPI-to-provider smoke test.

This module is deliberately excluded from pytest. It sends synthetic 1x1 PNGs
for the required case, test, and imaging inputs and prints only safe run metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import zlib

from httpx import ASGITransport, AsyncClient

from app import create_app
from config import Settings
from exceptions import ConfigurationError


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _synthetic_png() -> bytes:
    """Create a standards-compliant 1x1 RGB PNG without an image dependency."""

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\x80\x80\x80")
    return signature + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", pixels) + _png_chunk(b"IEND", b"")


SYNTHETIC_PNG = _synthetic_png()


async def _wait_for_terminal(
    client: AsyncClient,
    thread_id: str,
    *,
    timeout_seconds: float,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/runs/{thread_id}")
        if response.status_code != 200:
            return {"status": "API_STATUS_ERROR", "http_status": response.status_code}
        payload = response.json()
        if payload["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return payload
        await asyncio.sleep(1)
    return {"status": "TIMEOUT"}


async def main() -> int:
    settings = Settings()
    try:
        settings.require_openai()
    except ConfigurationError:
        print("LIVE_SMOKE_CONFIGURATION_ERROR", file=sys.stderr)
        return 2

    timeout_seconds = float(os.getenv("LIVE_SMOKE_TIMEOUT_SECONDS", "300"))
    app = create_app(settings=settings)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    resume_thread_id = os.getenv("LIVE_SMOKE_THREAD_ID")
    metadata = {
        "case_id": "live-smoke-synthetic",
        "confirmed_preoperative_only": True,
        "images": [
            {
                "image_id": "synthetic-png-1",
                "modality": "MRI",
                "body_region": "spine",
                "view_or_plane": "unknown",
                "sequence": None,
                "level": None,
                "description": "Synthetic API smoke-test image only.",
            }
        ],
    }
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://live-smoke",
            timeout=timeout_seconds + 30,
        ) as client:
            if resume_thread_id:
                resumed = await client.post(f"/v1/runs/{resume_thread_id}/resume")
                if resumed.status_code != 202:
                    print(
                        json.dumps(
                            {
                                "status": "RESUME_FAILED",
                                "http_status": resumed.status_code,
                                "error": resumed.json(),
                            }
                        )
                    )
                    return 3
                thread_id = resume_thread_id
            else:
                submitted = await client.post(
                    "/v1/cases",
                    data={"metadata": json.dumps(metadata)},
                    files=[
                        (
                            "case_files",
                            ("synthetic-case.png", SYNTHETIC_PNG, "image/png"),
                        ),
                        (
                            "test_files",
                            ("synthetic-tests.png", SYNTHETIC_PNG, "image/png"),
                        ),
                        (
                            "image_files",
                            ("synthetic-image.png", SYNTHETIC_PNG, "image/png"),
                        ),
                    ],
                )
                if submitted.status_code != 202:
                    print(
                        json.dumps(
                            {
                                "status": "SUBMISSION_FAILED",
                                "http_status": submitted.status_code,
                                "error": submitted.json(),
                            }
                        )
                    )
                    return 3
                thread_id = submitted.json()["thread_id"]
            result = await _wait_for_terminal(
                client,
                thread_id,
                timeout_seconds=timeout_seconds,
            )
            values = await app.state.run_manager._checkpoint_values(thread_id)

    run_metadata = values.get("run_metadata") if values else None
    speaker_counts = values.get("speaker_counts", {}) if values else {}
    termination_reason = values.get("termination_reason") if values else None
    final_plan = result.get("final_plan") or {}
    conditional_options = final_plan.get("conditional_surgical_options") or []
    safe_result = {
        "case_id": result.get("case_id", metadata["case_id"]),
        "thread_id": result.get("thread_id", thread_id),
        "status": result.get("status"),
        "phase": result.get("phase"),
        "message_count": len(result.get("new_messages", [])),
        "is_complete": result.get("is_complete", False),
        "error_code": result.get("error_code"),
        "termination_reason": getattr(termination_reason, "value", termination_reason),
        "speaker_counts": speaker_counts,
        "finalization_fallback_used": getattr(
            run_metadata,
            "finalization_fallback_used",
            None,
        ),
        "radiology_cache_hit": getattr(run_metadata, "radiology_cache_hit", None),
        "final_plan_confidence": final_plan.get("confidence"),
        "definitive_surgery_status": final_plan.get("definitive_surgery_status"),
        "definitive_surgery_withheld_reasons": final_plan.get(
            "definitive_surgery_withheld_reasons"
        ),
        "conditional_surgical_option_count": len(conditional_options),
        "conditional_preferred_procedure": (
            conditional_options[0].get("procedure_name")
            if conditional_options
            else None
        ),
        "conditional_preferred_uncertainty": (
            conditional_options[0].get("uncertainty")
            if conditional_options
            else None
        ),
    }
    print(json.dumps(safe_result, indent=2, sort_keys=True))
    full_coverage = bool(speaker_counts) and all(count >= 1 for count in speaker_counts.values())
    normal_finalization = safe_result["finalization_fallback_used"] is False
    return 0 if result.get("status") == "COMPLETED" and full_coverage and normal_finalization else 4


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
