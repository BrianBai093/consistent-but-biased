"""Multimodal spine radiology agent."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Sequence, cast

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from exceptions import ModelInvocationError, StructuredOutputError
from schemas import (
    AgentRole,
    CaseImage,
    ChatMessage,
    RadiologyAddendum,
    RadiologyProgress,
    RadiologyReport,
)
from services.image_batching import partition_images
from services.llm_client import ModelResult, aggregate_usage


RadiologyProgressCallback = Callable[[RadiologyProgress], Awaitable[None]]


class RadiologistAgent(BaseAgent[RadiologyReport]):
    role = AgentRole.RADIOLOGIST
    display_name = "Radiologist"
    prompt_filename = "radiologist.md"
    output_model = RadiologyReport

    def __init__(
        self,
        client,
        *,
        batch_max_images: int = 4,
        batch_max_bytes: int = 12 * 1024 * 1024,
        batch_concurrency: int = 2,
    ) -> None:
        super().__init__(client)
        if batch_max_images < 1:
            raise ValueError("batch_max_images must be positive")
        if batch_max_bytes < 1:
            raise ValueError("batch_max_bytes must be positive")
        if batch_concurrency < 1:
            raise ValueError("batch_concurrency must be positive")
        self.batch_max_images = batch_max_images
        self.batch_max_bytes = batch_max_bytes
        self.batch_concurrency = batch_concurrency
        self.addendum_prompt = self._load_prompt_file("radiologist_addendum.md")

    def build_user_prompt(
        self,
        *,
        case_summary_md: str,
        radiology_report: RadiologyReport | None,
        chat_log: Sequence[ChatMessage],
        question: str,
    ) -> str:
        return json.dumps(
            {
                "task": "Interpret the supplied spine images and answer the moderator question.",
                "case_summary_md": case_summary_md,
                "prior_radiology_report": (
                    radiology_report.model_dump(mode="json") if radiology_report else None
                ),
                "moderator_question": question,
            },
            ensure_ascii=False,
        )

    async def run(
        self,
        *,
        case_summary_md: str,
        images: Sequence[CaseImage],
        chat_log: Sequence[ChatMessage],
        question: str,
        radiology_report: RadiologyReport | None = None,
        progress_callback: RadiologyProgressCallback | None = None,
        output_processor: Callable[[BaseModel], BaseModel] | None = None,
    ) -> ModelResult[RadiologyReport]:
        if not images:
            raise ModelInvocationError("Radiology review requires at least one image.")

        base_prompt = self.build_user_prompt(
            case_summary_md=case_summary_md,
            radiology_report=radiology_report,
            chat_log=chat_log,
            question=question,
        )
        batches = self._partition_images(images)
        if len(batches) == 1:
            await _report_progress(
                progress_callback,
                RadiologyProgress(
                    stage="reviewing",
                    completed_images=0,
                    total_images=len(images),
                    batch_number=1,
                    total_batches=1,
                    current_image_start=1,
                    current_image_end=len(images),
                ),
            )
            prompt = self._batch_prompt(
                base_prompt=base_prompt,
                batch=batches[0],
                batch_number=1,
                batch_count=1,
                global_start=1,
            )
            result = await self._invoke(
                user_prompt=prompt,
                images=batches[0],
                output_processor=output_processor,
            )
            await _report_progress(
                progress_callback,
                RadiologyProgress(
                    stage="completed",
                    completed_images=len(images),
                    total_images=len(images),
                    batch_number=1,
                    total_batches=1,
                ),
            )
            return result

        semaphore = asyncio.Semaphore(self.batch_concurrency)
        progress_lock = asyncio.Lock()
        completed_images = [0]
        global_start = 1
        review_tasks: list[asyncio.Task[ModelResult[RadiologyReport]]] = []
        batch_positions: list[tuple[int, int]] = []
        for batch_number, batch in enumerate(batches, start=1):
            start = global_start
            end = start + len(batch) - 1
            batch_positions.append((start, end))
            prompt = self._batch_prompt(
                base_prompt=base_prompt,
                batch=batch,
                batch_number=batch_number,
                batch_count=len(batches),
                global_start=start,
            )
            review_tasks.append(
                asyncio.create_task(
                    self._review_batch(
                        semaphore,
                        prompt,
                        batch,
                        progress_callback=progress_callback,
                        progress_lock=progress_lock,
                        completed_images=completed_images,
                        batch_number=batch_number,
                        total_batches=len(batches),
                        global_start=start,
                        total_images=len(images),
                        output_processor=output_processor,
                    ),
                    name=f"radiology-batch-{batch_number}",
                )
            )
            global_start = end + 1

        try:
            batch_results = await asyncio.gather(*review_tasks)
        except BaseException:
            for task in review_tasks:
                task.cancel()
            await asyncio.gather(*review_tasks, return_exceptions=True)
            raise
        synthesis_prompt = self._synthesis_prompt(
            base_prompt=base_prompt,
            images=images,
            batch_results=batch_results,
            batch_positions=batch_positions,
        )
        await _report_progress(
            progress_callback,
            RadiologyProgress(
                stage="synthesizing",
                completed_images=len(images),
                total_images=len(images),
                total_batches=len(batches),
            ),
        )
        synthesis = await self._invoke(
            user_prompt=synthesis_prompt,
            images=(),
            output_processor=output_processor,
        )
        await _report_progress(
            progress_callback,
            RadiologyProgress(
                stage="completed",
                completed_images=len(images),
                total_images=len(images),
                total_batches=len(batches),
            ),
        )
        usages = tuple(
            usage
            for result in [*batch_results, synthesis]
            for usage in result.all_usages
        )
        return ModelResult(
            output=synthesis.output,
            usage=aggregate_usage(usages),
            usages=usages,
        )

    async def run_targeted(
        self,
        *,
        case_summary_md: str,
        images: Sequence[CaseImage],
        radiology_report: RadiologyReport,
        question: str,
        target_image_ids: Sequence[str],
        progress_callback: RadiologyProgressCallback | None = None,
    ) -> ModelResult[RadiologyAddendum]:
        targets = list(target_image_ids)
        supplied = [image.image_id for image in images]
        if not 1 <= len(targets) <= 4 or len(targets) != len(set(targets)):
            raise ModelInvocationError("Targeted radiology review requires 1-4 unique image ids.")
        if supplied != targets:
            raise ModelInvocationError(
                "Targeted radiology images must exactly match the requested image ids."
            )
        await _report_progress(
            progress_callback,
            RadiologyProgress(
                stage="reviewing",
                completed_images=0,
                total_images=len(images),
                batch_number=1,
                total_batches=1,
                current_image_start=1,
                current_image_end=len(images),
            ),
        )
        prompt = json.dumps(
            {
                "task": "Recheck only the explicitly targeted key images.",
                "case_summary_md": case_summary_md,
                "existing_radiology_report": radiology_report.model_dump(mode="json"),
                "moderator_question": question,
                "target_image_ids": targets,
                "image_metadata": [
                    _sanitized_metadata(image) for image in images
                ],
            },
            ensure_ascii=False,
        )

        def validate_addendum(output):
            addendum = RadiologyAddendum.model_validate(output)
            if addendum.target_image_ids != targets:
                raise StructuredOutputError(
                    "Radiology addendum did not preserve the targeted image order."
                )
            return addendum

        result = await self._invoke(
            user_prompt=prompt,
            images=images,
            output_model=RadiologyAddendum,
            system_prompt=self.addendum_prompt,
            output_processor=validate_addendum,
        )
        await _report_progress(
            progress_callback,
            RadiologyProgress(
                stage="completed",
                completed_images=len(images),
                total_images=len(images),
                batch_number=1,
                total_batches=1,
            ),
        )
        return cast(ModelResult[RadiologyAddendum], result)

    async def _review_batch(
        self,
        semaphore: asyncio.Semaphore,
        prompt: str,
        images: Sequence[CaseImage],
        *,
        progress_callback: RadiologyProgressCallback | None = None,
        progress_lock: asyncio.Lock | None = None,
        completed_images: list[int] | None = None,
        batch_number: int = 1,
        total_batches: int = 1,
        global_start: int = 1,
        total_images: int | None = None,
        output_processor: Callable[[BaseModel], BaseModel] | None = None,
    ) -> ModelResult[RadiologyReport]:
        async with semaphore:
            image_total = total_images or len(images)
            await _report_progress(
                progress_callback,
                RadiologyProgress(
                    stage="reviewing",
                    completed_images=(completed_images or [0])[0],
                    total_images=image_total,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    current_image_start=global_start,
                    current_image_end=global_start + len(images) - 1,
                ),
            )
            result = await self._invoke(
                user_prompt=prompt,
                images=images,
                output_processor=output_processor,
            )
            if progress_lock is not None and completed_images is not None:
                async with progress_lock:
                    completed_images[0] += len(images)
                    completed_count = completed_images[0]
            else:
                completed_count = len(images)
            await _report_progress(
                progress_callback,
                RadiologyProgress(
                    stage="batch_completed",
                    completed_images=completed_count,
                    total_images=image_total,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    current_image_start=global_start,
                    current_image_end=global_start + len(images) - 1,
                ),
            )
            return result

    def _partition_images(
        self,
        images: Sequence[CaseImage],
    ) -> list[list[CaseImage]]:
        return partition_images(
            images,
            max_images=self.batch_max_images,
            max_bytes=self.batch_max_bytes,
        )

    def _batch_prompt(
        self,
        *,
        base_prompt: str,
        batch: Sequence[CaseImage],
        batch_number: int,
        batch_count: int,
        global_start: int,
    ) -> str:
        metadata = [
            {
                "input_position": global_start + offset,
                **_sanitized_metadata(image),
            }
            for offset, image in enumerate(batch)
        ]
        instructions = {
            "review_stage": "image_batch_review",
            "batch_number": batch_number,
            "batch_count": batch_count,
            "instructions": (
                "Review only this ordered image batch. Link findings only to image_id values "
                "present in this batch. Do not infer findings from unseen batches. A later "
                "Radiologist synthesis pass will combine all batch reports."
            ),
            "image_metadata": metadata,
        }
        return f"{base_prompt}\n\nBatch review context:\n{json.dumps(instructions, ensure_ascii=False)}"

    def _synthesis_prompt(
        self,
        *,
        base_prompt: str,
        images: Sequence[CaseImage],
        batch_results: Sequence[ModelResult[RadiologyReport]],
        batch_positions: Sequence[tuple[int, int]],
    ) -> str:
        batch_reports = []
        for index, (result, positions) in enumerate(
            zip(batch_results, batch_positions, strict=True),
            start=1,
        ):
            batch_reports.append(
                {
                    "batch_number": index,
                    "input_positions": {"start": positions[0], "end": positions[1]},
                    "report": result.output.model_dump(mode="json"),
                }
            )
        context = {
            "review_stage": "cross_batch_synthesis",
            "instructions": (
                "Consolidate the Radiologist's own direct observations from every batch into one "
                "structured report. Preserve image_id links, disagreements, uncertainty, and "
                "missing views. Do not add a finding that is absent from the batch reports."
            ),
            "all_image_metadata": [
                {"input_position": index, **_sanitized_metadata(image)}
                for index, image in enumerate(images, start=1)
            ],
            "batch_reports": batch_reports,
        }
        return f"{base_prompt}\n\nCross-batch synthesis context:\n{json.dumps(context, ensure_ascii=False)}"


def _sanitized_metadata(image: CaseImage) -> dict:
    return image.model_dump(
        mode="json",
        exclude={"path_or_url", "original_filename"},
    )


async def _report_progress(
    callback: RadiologyProgressCallback | None,
    progress: RadiologyProgress,
) -> None:
    if callback is not None:
        await callback(progress)
