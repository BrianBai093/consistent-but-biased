"""Content-addressed cache for validated initial RadiologyReport outputs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field

from schemas import (
    CaseImage,
    CompressionDistribution,
    DocumentPage,
    DynamicStability,
    FixationImagingSupport,
    ImageLinkedFinding,
    OperativePlanningAssessment,
    PosteriorCordDriftFeasibility,
    RadiologyReport,
    SagittalAlignment,
    StrictModel,
)


from services.source_fingerprint import (
    CachedImageDescriptor,
    SourceFingerprint,
    fingerprint_inputs,
    write_record_atomically,
)

logger = logging.getLogger(__name__)
RADIOLOGY_CACHE_FORMAT_VERSION = "v4"


class RadiologyCacheRecord(StrictModel):
    format_version: Literal["v4"] = RADIOLOGY_CACHE_FORMAT_VERSION
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_page_sha256: list[str] = Field(min_length=1)
    test_page_sha256: list[str] = Field(min_length=1)
    images: list[CachedImageDescriptor] = Field(min_length=1)
    report: RadiologyReport


class RadiologyCache:
    """Persist only exact-input, path-free, initial radiology reports."""

    def __init__(self, root: Path | str, *, enabled: bool = True) -> None:
        self.root = Path(root).resolve()
        self.enabled = enabled

    async def load(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
    ) -> RadiologyReport | None:
        if not self.enabled:
            return None
        fingerprint = await fingerprint_inputs(case_pages, test_pages, images)
        if fingerprint is None:
            return None
        key = fingerprint.key(RADIOLOGY_CACHE_FORMAT_VERSION)
        cache_path = self.root / f"{key}.json"
        try:
            raw = await asyncio.to_thread(cache_path.read_text, encoding="utf-8")
            record = RadiologyCacheRecord.model_validate_json(raw)
            if record.cache_key != key or not _record_matches(record, fingerprint):
                return None
            slot_to_current = {
                descriptor.slot_id: image.image_id
                for descriptor, image in zip(record.images, images, strict=True)
            }
            return _remap_report(record.report, slot_to_current)
        except (OSError, ValueError):
            return None

    async def store(
        self,
        *,
        case_pages: Sequence[DocumentPage],
        test_pages: Sequence[DocumentPage],
        images: Sequence[CaseImage],
        report: RadiologyReport,
    ) -> str | None:
        if not self.enabled or report.addenda:
            return None
        fingerprint = await fingerprint_inputs(case_pages, test_pages, images)
        if fingerprint is None:
            return None
        current_to_slot = {
            image.image_id: descriptor.slot_id
            for descriptor, image in zip(fingerprint.images, images, strict=True)
        }
        try:
            normalized_report = _remap_report(report, current_to_slot)
        except ValueError:
            return None
        key = fingerprint.key(RADIOLOGY_CACHE_FORMAT_VERSION)
        record = RadiologyCacheRecord(
            cache_key=key,
            case_page_sha256=fingerprint.case_page_sha256,
            test_page_sha256=fingerprint.test_page_sha256,
            images=fingerprint.images,
            report=normalized_report,
        )
        try:
            await asyncio.to_thread(
                write_record_atomically,
                self.root,
                key,
                record.model_dump_json(indent=2),
            )
        except OSError:
            logger.warning("Could not persist validated radiology cache entry")
            return None
        return key


def migrate_pre_v4_report(report: RadiologyReport) -> RadiologyReport:
    """Conservatively add v4 planning fields without reinterpreting image pixels."""

    base = report.model_copy(update={"addenda": []})
    alignment_text = base.alignment.casefold()
    if any(term in alignment_text for term in ("straighten", "变直")):
        alignment = SagittalAlignment.STRAIGHTENED
    elif any(term in alignment_text for term in ("kypho", "后凸")):
        alignment = SagittalAlignment.KYPHOTIC
    elif any(
        term in alignment_text
        for term in ("preserved lordosis", "lordosis is preserved", "前凸正常")
    ):
        alignment = SagittalAlignment.LORDOTIC
    elif any(term in alignment_text for term in ("neutral", "中立")):
        alignment = SagittalAlignment.NEUTRAL
    else:
        alignment = SagittalAlignment.INDETERMINATE

    compression_text = " ".join(
        (
            base.canal_and_foramina,
            base.neural_compression,
            base.cross_modality_synthesis,
            *(item.finding for item in _all_findings(base)),
        )
    ).casefold()
    has_ventral = any(term in compression_text for term in ("ventral", "腹侧", "前方"))
    has_dorsal = any(term in compression_text for term in ("dorsal", "背侧", "后方压迫"))
    if "circumferential" in compression_text or "环周" in compression_text:
        distribution = CompressionDistribution.CIRCUMFERENTIAL
    elif has_ventral and has_dorsal:
        distribution = CompressionDistribution.MIXED
    elif has_ventral:
        distribution = CompressionDistribution.VENTRAL
    elif has_dorsal:
        distribution = CompressionDistribution.DORSAL
    else:
        distribution = CompressionDistribution.INDETERMINATE

    planning_evidence = _planning_evidence_ids(base)
    planning = OperativePlanningAssessment(
        sagittal_alignment=alignment,
        dynamic_stability=DynamicStability.NOT_ASSESSED,
        compression_distribution=distribution,
        posterior_cord_drift_feasibility=(
            PosteriorCordDriftFeasibility.INDETERMINATE
        ),
        fixation_imaging_support=FixationImagingSupport.NOT_ASSESSED,
        planning_uncertainties=[
            "Migrated from a completed pre-v4 structured report; no new image review "
            "was performed.",
            "Dynamic stability, posterior cord drift feasibility, and fixation imaging "
            "support were not established by the pre-v4 report.",
        ],
        evidence_ids=planning_evidence,
    )
    return base.model_copy(update={"operative_planning": planning})


def _record_matches(
    record: RadiologyCacheRecord,
    fingerprint: SourceFingerprint,
) -> bool:
    return (
        record.case_page_sha256 == fingerprint.case_page_sha256
        and record.test_page_sha256 == fingerprint.test_page_sha256
        and record.images == fingerprint.images
    )


def _remap_report(report: RadiologyReport, mapping: dict[str, str]) -> RadiologyReport:
    if report.addenda:
        raise ValueError("Initial radiology cache entries cannot contain addenda")

    def findings(items: Sequence[ImageLinkedFinding]) -> list[ImageLinkedFinding]:
        return [
            item.model_copy(update={"image_ids": _remap_ids(item.image_ids, mapping)})
            for item in items
        ]

    questions = [
        item.model_copy(update={"evidence_ids": _remap_ids(item.evidence_ids, mapping)})
        for item in report.questions_for_other_agents
    ]
    claims = [
        item.model_copy(update={"evidence_ids": _remap_ids(item.evidence_ids, mapping)})
        for item in report.claim_evidence
    ]
    planning = report.operative_planning.model_copy(
        update={
            "evidence_ids": _remap_ids(
                report.operative_planning.evidence_ids,
                mapping,
            )
        }
    )
    return report.model_copy(
        update={
            "xray_findings": findings(report.xray_findings),
            "ct_findings": findings(report.ct_findings),
            "mri_findings": findings(report.mri_findings),
            "questions_for_other_agents": questions,
            "claim_evidence": claims,
            "operative_planning": planning,
            "addenda": [],
        }
    )


def _remap_ids(identifiers: Sequence[str], mapping: dict[str, str]) -> list[str]:
    remapped = []
    for identifier in identifiers:
        if identifier not in mapping:
            raise ValueError("Radiology cache evidence must reference supplied images only")
        remapped.append(mapping[identifier])
    return remapped


def _all_findings(report: RadiologyReport) -> list[ImageLinkedFinding]:
    return [*report.xray_findings, *report.ct_findings, *report.mri_findings]


def _planning_evidence_ids(report: RadiologyReport) -> list[str]:
    targeted_terms = (
        "align",
        "lordo",
        "kypho",
        "straight",
        "ventral",
        "dorsal",
        "canal",
        "cord",
        "排列",
        "前凸",
        "后凸",
        "变直",
        "压迫",
        "椎管",
        "脊髓",
    )
    selected = [
        image_id
        for finding in _all_findings(report)
        if any(term in finding.finding.casefold() for term in targeted_terms)
        for image_id in finding.image_ids
    ]
    if not selected:
        selected = [
            image_id
            for finding in _all_findings(report)
            for image_id in finding.image_ids
        ]
    return list(dict.fromkeys(selected))
