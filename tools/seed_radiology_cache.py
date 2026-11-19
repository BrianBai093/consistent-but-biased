"""Seed a safe v4 radiology cache entry from a completed local run."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from config import Settings
from graph import thread_config
from schemas import RadiologyReport, RunPhase
from services.checkpoint import open_sqlite_checkpointer
from services.context_policy import normalize_radiology_planning
from services.radiology_cache import RadiologyCache, migrate_pre_v4_report


async def seed_cache(
    *,
    thread_id: str,
    checkpoint_db_path: Path,
    cache_dir: Path,
) -> dict:
    async with open_sqlite_checkpointer(checkpoint_db_path) as checkpointer:
        checkpoint = await checkpointer.aget_tuple(thread_config(thread_id))
    if checkpoint is None:
        raise RuntimeError("No checkpoint exists for the requested run.")
    values = checkpoint.checkpoint.get("channel_values", {})
    phase = values.get("phase")
    if phase not in {RunPhase.COMPLETED, RunPhase.COMPLETED.value}:
        raise RuntimeError("Only a completed run can seed a radiology cache entry.")
    if values.get("final_plan") is None:
        raise RuntimeError("The completed run has no reviewed final plan.")
    if "<!-- source: addendum-" in values.get("case_preop_md", ""):
        raise RuntimeError("Runs containing a case addendum cannot seed this cache.")

    report = RadiologyReport.model_validate(values.get("radiology_report"))
    images = list(values.get("images") or [])
    case_pages = list(values.get("case_pages") or [])
    test_pages = list(values.get("test_pages") or [])
    if not images or not case_pages or not test_pages:
        raise RuntimeError("The completed run does not contain the required source assets.")

    migrated = migrate_pre_v4_report(report)
    migrated = normalize_radiology_planning(
        migrated,
        images=images,
        allowed_ids={image.image_id for image in images},
    )
    cache = RadiologyCache(cache_dir)
    cache_key = await cache.store(
        case_pages=case_pages,
        test_pages=test_pages,
        images=images,
        report=migrated,
    )
    if cache_key is None:
        raise RuntimeError("The radiology report did not satisfy safe cache requirements.")
    verified = await cache.load(
        case_pages=case_pages,
        test_pages=test_pages,
        images=images,
    )
    if verified is None:
        raise RuntimeError("The new cache entry could not be verified after writing.")

    return {
        "status": "CACHE_CREATED",
        "cache_key": cache_key,
        "format_version": "v4",
        "image_count": len(images),
        "image_quality": verified.image_quality.value,
        "operative_planning": verified.operative_planning.model_dump(mode="json"),
        "addenda_cached": len(verified.addenda),
    }


def _parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=settings.checkpoint_db_path,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.radiology_cache_dir,
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = asyncio.run(
            seed_cache(
                thread_id=arguments.thread_id,
                checkpoint_db_path=arguments.checkpoint_db,
                cache_dir=arguments.cache_dir,
            )
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "CACHE_NOT_CREATED", "reason": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
