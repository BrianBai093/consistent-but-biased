"""Deterministic Markdown rendering for structured radiology reports."""

from schemas import ImageLinkedFinding, RadiologyAddendum, RadiologyReport


def render_radiology_markdown(report: RadiologyReport) -> str:
    sections = [
        "# 术前影像报告",
        "",
        f"- 图像质量：{report.image_quality.value}",
        f"- 可用模态：{', '.join(item.value for item in report.available_modalities) or '无'}",
        f"- 置信度：{report.confidence:.2f}",
        "",
        "## X-ray 所见",
        *_finding_lines(report.xray_findings),
        "",
        "## CT 所见",
        *_finding_lines(report.ct_findings),
        "",
        "## MRI 所见",
        *_finding_lines(report.mri_findings),
        "",
        "## 综合评估",
        f"- 节段：{', '.join(report.spine_levels) or '未确定'}",
        f"- 排列：{report.alignment}",
        f"- 骨性结构：{report.bony_structures}",
        f"- 椎管及椎间孔：{report.canal_and_foramina}",
        f"- 神经压迫：{report.neural_compression}",
        f"- 跨模态结论：{report.cross_modality_synthesis}",
        "",
        "## 手术规划相关影像因素",
        f"- 矢状位曲度：{report.operative_planning.sagittal_alignment.value}",
        f"- 动态稳定性：{report.operative_planning.dynamic_stability.value}",
        f"- 压迫前后方分布：{report.operative_planning.compression_distribution.value}",
        (
            "- 脊髓后漂移条件："
            f"{report.operative_planning.posterior_cord_drift_feasibility.value}"
        ),
        f"- 固定相关影像支持：{report.operative_planning.fixation_imaging_support.value}",
        (
            "- 规划证据："
            f"{', '.join(report.operative_planning.evidence_ids) or '未提供'}"
        ),
        "- 规划不确定性：",
        *_nested_text_lines(report.operative_planning.planning_uncertainties),
        "",
        "## 可能诊断",
        *_text_lines(report.possible_diagnoses),
        "",
        "## 不确定性",
        *_text_lines(report.uncertain_findings),
        "",
        "## 缺失视图或序列",
        *_text_lines(report.missing_views_or_sequences),
        "",
        "## 公开摘要",
        report.public_summary,
        *_addendum_sections(report.addenda),
    ]
    return "\n".join(sections).strip() + "\n"


def _finding_lines(findings: list[ImageLinkedFinding]) -> list[str]:
    if not findings:
        return ["- 无可报告所见"]
    return [
        f"- {item.finding}（图像：{', '.join(item.image_ids)}）"
        for item in findings
    ]


def _text_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- 无"]


def _nested_text_lines(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items] if items else ["  - 无"]


def merge_radiology_addendum(
    report: RadiologyReport,
    addendum: RadiologyAddendum,
) -> RadiologyReport:
    """Append a targeted recheck without allowing it to erase the base report."""

    return report.model_copy(update={"addenda": [*report.addenda, addendum]})


def _addendum_sections(addenda: list[RadiologyAddendum]) -> list[str]:
    lines: list[str] = []
    for index, addendum in enumerate(addenda, start=1):
        lines.extend(
            [
                "",
                f"## Targeted key-image recheck {index}",
                f"- Images: {', '.join(addendum.target_image_ids)}",
                f"- Confidence: {addendum.confidence:.2f}",
                *_finding_lines(addendum.clarifications),
                *_text_lines(addendum.uncertain_findings),
                *_text_lines(addendum.missing_views_or_sequences),
                addendum.public_summary,
            ]
        )
    return lines
