"""Persistence for the frozen preparation of one case.

The decision-style experiment runs the same case three times, once per director
style, and attributes any difference in the outcome to the style. That argument
only holds if the three runs argue over byte-identical evidence, so preparation
runs once and its result is stored here for every later run to load.

Case-record transcription is already deterministic OCR, but case summarisation
and radiology review are model calls and would drift on a rerun. This store is
what removes that drift.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from schemas import (
    MISSING_RECORDS,
    CaseBaseline,
    CaseImage,
    DocumentPage,
    RadiologyReport,
)
from services.case_report_filter import FilterRecord
from services.record_redaction import RedactionRecord
from services.source_fingerprint import (
    fingerprint_inputs,
    sha256_local_file,
    write_record_atomically,
)

logger = logging.getLogger(__name__)
# Salts the storage key. It identifies the *question* the key answers -- "is
# this the same prepared case?" -- and must stay fixed as the record format
# evolves, or every format change would orphan the baselines on disk.
BASELINE_KEY_NAMESPACE = "baseline"


class BaselineStore:
    """Content-addressed storage of prepared evidence, keyed by source bytes."""

    def __init__(self, root: Path | str, *, enabled: bool = True) -> None:
        self.root = Path(root).resolve()
        self.enabled = enabled

    async def load(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
    ) -> CaseBaseline | None:
        """Return the stored preparation for these exact inputs, or None.

        Every mismatch is a miss rather than an error: a baseline is an
        optimisation, and regenerating it is always correct.
        """

        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        path = self.root / f"{key}.json"
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            baseline = CaseBaseline.model_validate_json(raw)
        except (OSError, ValueError):
            return None
        if baseline.baseline_key != key or baseline.case_id != case_id:
            return None
        if not _identifiers_match(baseline, case_pages, test_pages, images):
            # The report cites page and image IDs directly, so a baseline built
            # under different identifiers would carry dangling citations.
            return None
        return baseline

    async def store(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
        case_preop_md: str,
        tests_preop_md: str,
        case_summary_md: str,
        radiology_preop_md: str,
        preparation_warnings: Sequence[str],
        radiology_report: RadiologyReport,
    ) -> str | None:
        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        # A department payload already stored for this case is preparation that
        # was paid for and frozen; re-running v5 preparation must not drop it.
        frozen_department = await self._stored_department(key)
        baseline = CaseBaseline(
            baseline_key=key,
            case_id=case_id,
            case_page_ids=[page.page_id for page in case_pages],
            test_page_ids=[page.page_id for page in test_pages],
            image_ids=[image.image_id for image in images],
            case_preop_md=case_preop_md,
            tests_preop_md=tests_preop_md,
            case_summary_md=case_summary_md,
            radiology_preop_md=radiology_preop_md,
            preparation_warnings=list(preparation_warnings),
            radiology_report=radiology_report,
            format_version="v2" if frozen_department else "v1",
            department=frozen_department,
            created_at=datetime.now(UTC),
        )
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                key,
                baseline.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist the prepared case baseline")
            return None
        return key

    async def load_redaction(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
    ) -> RedactionRecord | None:
        """The frozen redaction of this case's record, signed or not.

        Kept beside the baseline under the same key: it belongs to the same
        prepared case, and a signature given once has to survive to the next run
        or a person would be asked to review the same diff every time.
        """

        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        try:
            raw = await asyncio.to_thread(
                (self.root / f"{key}.redaction.json").read_text,
                encoding="utf-8",
            )
            return RedactionRecord.model_validate_json(raw)
        except (OSError, ValueError):
            return None

    async def store_redaction(
        self,
        record: RedactionRecord,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
    ) -> str | None:
        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                f"{key}.redaction",
                record.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist the redaction record")
            return None
        return key

    async def store_preparation(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
        case_preop_md: str,
        tests_preop_md: str,
        case_summary_md: str,
        preparation_warnings: Sequence[str],
        redaction_sha: str | None = None,
    ) -> str | None:
        """Freeze the transcription and summary before any discussion happens.

        Transcribing a case is the expensive half and it has nothing to do with
        the arms, so it is worth keeping on its own: a run that stops to let
        someone read the transcription should not have to pay for it again, and
        neither should the arms that follow.
        """

        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        frozen_department = await self._stored_department(key)
        if await self._stored_redaction_sha(key) != redaction_sha:
            # Stage I and II were built by reading the previous cleaned record.
            # A different redaction means a different record, so what the five
            # roles said about it no longer describes what the arms will read.
            frozen_department = None
        baseline = CaseBaseline(
            baseline_key=key,
            case_id=case_id,
            case_page_ids=[page.page_id for page in case_pages],
            test_page_ids=[page.page_id for page in test_pages],
            image_ids=[image.image_id for image in images],
            case_preop_md=case_preop_md,
            tests_preop_md=tests_preop_md,
            case_summary_md=case_summary_md,
            radiology_preop_md=MISSING_RECORDS,
            preparation_warnings=list(preparation_warnings),
            radiology_report=None,
            format_version="v2" if frozen_department else "v1",
            department=frozen_department,
            redaction_sha=redaction_sha,
            created_at=datetime.now(UTC),
        )
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                key,
                baseline.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist the prepared case")
            return None
        return key

    async def store_department(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
        case_preop_md: str,
        tests_preop_md: str,
        case_summary_md: str,
        preparation_warnings: Sequence[str],
        payload: BaseModel,
    ) -> str | None:
        """Attach the department's frozen stage I-II to this case's baseline.

        An existing record is upgraded in place rather than replaced, so the two
        workflows keep one record per prepared case instead of quietly diverging
        into two. A record that cannot be validated counts as absent and is
        replaced: unlike the department stages, the v5 half it would carry is
        regenerated by the preparation run that calls ``store``.
        """

        if not self.enabled:
            return None
        key = await self._key(case_pages, test_pages, images, case_id)
        if key is None:
            return None
        existing = await self.load(
            case_pages=case_pages,
            test_pages=test_pages,
            images=images,
            case_id=case_id,
        )
        if existing is not None:
            baseline = existing.model_copy(
                update={
                    "format_version": "v2",
                    "department": payload.model_dump(mode="json"),
                    "preparation_warnings": list(existing.preparation_warnings)
                    or list(preparation_warnings),
                }
            )
        else:
            # No v5 record exists for these sources. The radiology report the v5
            # schema requires is not available here and must not be invented, so
            # the department payload is stored on its own.
            baseline = CaseBaseline(
                baseline_key=key,
                case_id=case_id,
                case_page_ids=[page.page_id for page in case_pages],
                test_page_ids=[page.page_id for page in test_pages],
                image_ids=[image.image_id for image in images],
                case_preop_md=case_preop_md,
                tests_preop_md=tests_preop_md,
                case_summary_md=case_summary_md,
                radiology_preop_md=MISSING_RECORDS,
                preparation_warnings=list(preparation_warnings),
                radiology_report=None,
                department=payload.model_dump(mode="json"),
                created_at=datetime.now(UTC),
            )
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                key,
                baseline.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist the department baseline")
            return None
        return key

    async def _stored_redaction_sha(self, key: str) -> str | None:
        record = await self._stored_raw(key)
        value = record.get("redaction_sha") if record else None
        return value if isinstance(value, str) else None

    async def _stored_raw(self, key: str) -> dict | None:
        try:
            raw = await asyncio.to_thread(
                (self.root / f"{key}.json").read_text,
                encoding="utf-8",
            )
            record = json.loads(raw)
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    async def _stored_department(self, key: str) -> dict | None:
        """The frozen department stages at this key, whatever else is stored.

        Read as raw JSON rather than through the record model. A record this
        build cannot validate -- an older layout, a field since renamed -- would
        otherwise look like an empty slot, and the caller would overwrite stage
        I and II that were paid for once and are meant to outlive every arm.
        Because the key covers the case and its identifiers, whatever sits here
        belongs to this case.
        """

        path = self.root / f"{key}.json"
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            record = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict):
            return None
        payload = record.get("department")
        return payload if isinstance(payload, dict) else None

    async def _key(
        self,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
    ) -> str | None:
        """Address a record by everything that a hit requires.

        The source bytes alone are not enough. ``load`` also requires the case
        ID and the page and image identifiers to match, because the stored text
        cites those identifiers directly. If they were left out of the key, the
        same files prepared under a second case label would hash to the slot the
        first one occupies: the load would reject it and the store would then
        overwrite it, destroying one case's preparation and quietly sending the
        other back to the model. Folding them in turns that collision into two
        separate records.

        The identifiers are inputs to a hash, so nothing about the case appears
        in the stored filename.
        """

        fingerprint = await fingerprint_inputs(case_pages, test_pages, images)
        if fingerprint is None:
            return None
        identity = json.dumps(
            {
                "namespace": BASELINE_KEY_NAMESPACE,
                "case_id": case_id,
                "case_page_ids": [page.page_id for page in case_pages],
                "test_page_ids": [page.page_id for page in test_pages],
                "image_ids": [image.image_id for image in images],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return fingerprint.key(identity)


def _identifiers_match(
    baseline: CaseBaseline,
    case_pages: Sequence[DocumentPage],
    test_pages: Sequence[DocumentPage],
    images: Sequence[CaseImage],
) -> bool:
    return (
        baseline.case_page_ids == [page.page_id for page in case_pages]
        and baseline.test_page_ids == [page.page_id for page in test_pages]
        and baseline.image_ids == [image.image_id for image in images]
    )


#: A different question from the image store's, so a different salt: "is this
#: the same published case text?" can never land on a hospital case's slot.
TEXT_BASELINE_KEY_NAMESPACE = "baseline-text"


class TextCaseBaselineStore(BaselineStore):
    """The same frozen preparation, for a case that is a single text record.

    A published case report has no test sheets and no films, and the shared
    fingerprint refuses a case without both -- correctly, for the upload path.
    Here the key covers the bytes of the record and the identifiers, and
    nothing else; every other behaviour of the store is unchanged.
    """

    async def _key(
        self,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        case_id: str,
    ) -> str | None:
        if not case_pages or test_pages or images:
            return None
        digests = await asyncio.gather(
            *(sha256_local_file(page.path_or_url) for page in case_pages)
        )
        if any(digest is None for digest in digests):
            return None
        identity = json.dumps(
            {
                "namespace": TEXT_BASELINE_KEY_NAMESPACE,
                "case_id": case_id,
                "case_page_ids": [page.page_id for page in case_pages],
                "case_page_sha256": list(digests),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    async def load_filter(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        case_id: str,
    ) -> FilterRecord | None:
        """The frozen filter of this case's record, beside its baseline."""

        key = await self._key(case_pages, [], [], case_id)
        if key is None:
            return None
        try:
            raw = await asyncio.to_thread(
                (self.root / f"{key}.filter.json").read_text, encoding="utf-8"
            )
            return FilterRecord.model_validate_json(raw)
        except (OSError, ValueError):
            return None

    async def store_filter(
        self,
        record: FilterRecord,
        *,
        case_pages: Sequence[DocumentPage],
        case_id: str,
    ) -> str | None:
        key = await self._key(case_pages, [], [], case_id)
        if key is None:
            return None
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                f"{key}.filter",
                record.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist the case report filter")
            return None
        return key
