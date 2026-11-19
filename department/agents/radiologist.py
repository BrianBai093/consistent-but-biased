"""Radiology: the only role that sees the images (protocol Agent 2).

A case carries roughly a dozen photographed films, which is too much for one
request, so images are read in ordered bounded batches and then consolidated by
a text-only pass. The batches run concurrently; the consolidation sees only the
per-batch findings, never the images again.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from department.agents.base import DepartmentAgent
from department.schemas import ImagingFindings
from exceptions import ModelInvocationError
from schemas import AgentRole, CaseImage
from services.image_batching import (
    DEFAULT_BATCH_MAX_BYTES,
    DEFAULT_BATCH_MAX_IMAGES,
    partition_images,
)
from services.llm_client import ModelResult, StructuredModelClient, aggregate_usage


def _sanitized(image: CaseImage) -> dict:
    """Image metadata with the local path and original filename removed."""

    return {
        "image_id": image.image_id,
        "modality": image.modality.value,
        "body_region": image.body_region,
        "view_or_plane": image.view_or_plane,
        "sequence": image.sequence,
        "level": image.level,
        "description": image.description,
    }


class RadiologyAgent(DepartmentAgent[ImagingFindings]):
    role = AgentRole.RADIOLOGIST
    display_name = "Radiologist"
    prompt_filename = "radiologist.md"
    output_model = ImagingFindings
    knowledge_base = ("structured-imaging-report-template",)

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        knowledge_source=None,
        batch_max_images: int = DEFAULT_BATCH_MAX_IMAGES,
        batch_max_bytes: int = DEFAULT_BATCH_MAX_BYTES,
        batch_concurrency: int = 2,
    ) -> None:
        super().__init__(client, knowledge_source=knowledge_source)
        #: Appended to the role prompt for a case that has a record and no
        #: films. A separate file so its text is fingerprinted run by run.
        self.record_only_prompt = self.load_department_prompt(
            "radiologist_record_only.md"
        )
        if batch_max_images < 1:
            raise ValueError("batch_max_images must be positive")
        if batch_max_bytes < 1:
            raise ValueError("batch_max_bytes must be positive")
        if batch_concurrency < 1:
            raise ValueError("batch_concurrency must be positive")
        self.batch_max_images = batch_max_images
        self.batch_max_bytes = batch_max_bytes
        self.batch_concurrency = batch_concurrency

    async def run(
        self,
        *,
        case_summary_md: str,
        images: Sequence[CaseImage],
    ) -> ModelResult[ImagingFindings]:
        if not images:
            raise ModelInvocationError("Imaging review requires at least one image.")

        batches = partition_images(
            images,
            max_images=self.batch_max_images,
            max_bytes=self.batch_max_bytes,
        )
        if len(batches) == 1:
            return await self._read_batch(case_summary_md, batches[0], 1, 1)

        semaphore = asyncio.Semaphore(self.batch_concurrency)

        async def guarded(batch: Sequence[CaseImage], number: int):
            async with semaphore:
                return await self._read_batch(
                    case_summary_md, batch, number, len(batches)
                )

        tasks = [
            asyncio.create_task(guarded(batch, number), name=f"imaging-batch-{number}")
            for number, batch in enumerate(batches, start=1)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        usages = [usage for result in results for usage in result.all_usages]
        consolidated = await self._invoke(
            user_prompt=self.payload(
                task=(
                    "Consolidate the per-batch readings into one report for the case. "
                    "Do not invent a finding that no batch reported."
                ),
                case_summary_md=case_summary_md,
                batch_findings=[
                    result.output.model_dump(mode="json") for result in results
                ],
                images=[_sanitized(image) for image in images],
            ),
            images=(),
        )
        usages.extend(consolidated.all_usages)
        return ModelResult(
            output=consolidated.output,
            usage=aggregate_usage(usages),
            usages=tuple(usages),
            intermediates=tuple(result.output for result in results),
        )

    async def read_record(self, *, case_preop_md: str) -> ModelResult[ImagingFindings]:
        """Report the imaging of a case that comes with no films.

        A published case report carries its imaging only as prose. The record it
        reads is the redacted one every other role already reads, so this role
        learns nothing the others do not; what it loses is its independent look
        at the films, and the prompt makes it say so.
        """

        return await self._invoke(
            user_prompt=self.payload(
                task=(
                    "There are no images. Report the imaging as the case record "
                    "describes it."
                ),
                case_preop_md=case_preop_md,
            ),
            images=(),
            system_prompt=f"{self.system_prompt}\n\n{self.record_only_prompt}",
        )

    async def _read_batch(
        self,
        case_summary_md: str,
        batch: Sequence[CaseImage],
        number: int,
        total: int,
    ) -> ModelResult[ImagingFindings]:
        return await self._invoke(
            user_prompt=self.payload(
                task="Read this batch of images and report what they show.",
                case_summary_md=case_summary_md,
                batch_number=number,
                total_batches=total,
                images=[_sanitized(image) for image in batch],
            ),
            images=batch,
        )
