from schemas import (
    AdditionalTests,
    AgentRole,
    AlternativePlan,
    ApproachAssessment,
    ApproachComparison,
    ApproachSuitability,
    AnesthesiologistOutput,
    CaseImage,
    CaseInput,
    CaseSummaryMarkdown,
    ClaimEvidence,
    ConsistencyAssessment,
    DiagnosisEvidence,
    DefinitiveSurgeryStatus,
    DocumentPage,
    FinalConsensus,
    FinalPlan,
    FinalPlanReview,
    ImageLinkedFinding,
    ImageQuality,
    CompressionDistribution,
    DynamicStability,
    FixationImagingSupport,
    Modality,
    ModeratorAction,
    ModeratorDecision,
    NeurologistOutput,
    PerioperativePhysicianOutput,
    PerioperativeRiskSummary,
    PlanningCertainty,
    PlanPosition,
    PlanPositionAction,
    PosteriorCordDriftFeasibility,
    RadiologyAddendum,
    RadiologyReport,
    RiskLevel,
    SagittalAlignment,
    RadiologySummary,
    RecommendedSurgery,
    SpineSurgeonOutput,
    SurgicalApproach,
    SurgicalIndicationPlan,
    SurgicalOption,
    SurgicalOptionUncertainty,
    TestDocumentMarkdown,
)
from services.ocr_engine import FakeOcrEngine, OcrLine


def radiology_report() -> RadiologyReport:
    return RadiologyReport(
        image_quality=ImageQuality.DIAGNOSTIC,
        available_modalities=[Modality.MRI],
        xray_findings=[],
        ct_findings=[],
        mri_findings=[
            ImageLinkedFinding(finding="Right L4-L5 disc extrusion", image_ids=["mri-1"])
        ],
        spine_levels=["L4-L5"],
        alignment="Preserved on supplied key image",
        bony_structures="No acute osseous abnormality on supplied key image",
        canal_and_foramina="Right lateral recess narrowing at L4-L5",
        neural_compression="Right L5 nerve root compression",
        cross_modality_synthesis="Only MRI key images are available",
        possible_diagnoses=["L4-L5 disc extrusion"],
        uncertain_findings=[],
        missing_views_or_sequences=["No radiographs", "No CT"],
        public_summary="MRI key images show a right L4-L5 disc extrusion compressing the L5 root.",
        confidence=0.84,
        operative_planning={
            "sagittal_alignment": SagittalAlignment.LORDOTIC,
            "dynamic_stability": DynamicStability.NOT_ASSESSED,
            "compression_distribution": CompressionDistribution.VENTRAL,
            "posterior_cord_drift_feasibility": (
                PosteriorCordDriftFeasibility.INDETERMINATE
            ),
            "fixation_imaging_support": FixationImagingSupport.DOES_NOT_SUPPORT,
            "planning_uncertainties": [
                "Dynamic stability was not assessed because no flexion-extension "
                "or other dynamic view was supplied."
            ],
            "evidence_ids": ["mri-1"],
        },
    )


def moderator_decision() -> ModeratorDecision:
    return ModeratorDecision(
        public_message="Please provide the initial cross-modality radiology report.",
        action=ModeratorAction.ASK_SPECIALIST,
        next_speaker=AgentRole.RADIOLOGIST,
        question_for_next_speaker="What are the principal imaging findings?",
        unresolved_issues=["Imaging interpretation pending"],
        disagreements=[],
        consensus_reached=False,
        routing_reason="Radiology must report first.",
        continue_discussion=True,
        enter_final_summary=False,
        target_image_ids=["mri-1"],
    )


def spine_output() -> SpineSurgeonOutput:
    option = surgical_option()
    return SpineSurgeonOutput(
        public_message="The supplied data support an L4-L5 decompression indication.",
        surgical_indication=True,
        indication_rationale=["Concordant L5 radiculopathy"],
        functional_impact="Right leg pain limits walking and prolonged sitting.",
        conservative_treatment_review=["Nonoperative care did not provide durable relief."],
        preoperative_prerequisites=["Complete the missing standardized motor examination."],
        recommended_procedure="L4-L5 decompression and discectomy",
        surgical_levels=["L4-L5"],
        approach="Posterior",
        decompression_required=True,
        fusion_required=False,
        fixation_required=False,
        alternative_plans=["Continued nonoperative care if deficits are stable"],
        operative_options=[option],
        preferred_option_id=option.option_id,
        approach_comparison=approach_comparison(),
        major_surgical_risks=["Dural tear", "Nerve injury"],
        questions_for_other_agents=[],
        need_retrieval=False,
        retrieval_query=None,
        confidence=0.78,
        claim_evidence=[
            ClaimEvidence(
                field_path="public_message",
                claim="L4-L5 decompression is supported.",
                support_type="inference",
                evidence_ids=["case-page-001", "mri-1"],
            )
        ],
    )


def surgical_option(
    *,
    option_id: str = "option-1",
    rank: int = 1,
    procedure_name: str = "L4-L5 decompression and discectomy",
    approach: SurgicalApproach = SurgicalApproach.POSTERIOR,
) -> SurgicalOption:
    return SurgicalOption(
        option_id=option_id,
        rank=rank,
        procedure_name=procedure_name,
        approach=approach,
        levels=["L4-L5"],
        levels_certainty=PlanningCertainty.ESTABLISHED,
        decompression_extent="L4-L5 right lateral recess and affected root",
        fusion_or_fixation="No fusion or fixation proposed",
        fixation_certainty=PlanningCertainty.ESTABLISHED,
        rationale=["Concordant right L5 root compression and symptoms"],
        selection_conditions=["Confirm the documented neurologic localization"],
        reasons_against=["Incomplete standardized motor examination"],
        required_confirmation=["Complete the standardized motor examination"],
        uncertainty=SurgicalOptionUncertainty.MODERATE,
        confidence=0.78,
        evidence_ids=["case-page-001", "mri-1"],
    )


def approach_comparison() -> ApproachComparison:
    return ApproachComparison(
        assessments=[
            ApproachAssessment(
                approach=SurgicalApproach.ANTERIOR,
                suitability=ApproachSuitability.DISFAVORED,
                supporting_reasons=[],
                limiting_factors=["No anterior indication is established"],
                required_confirmation=["Review full imaging if anterior disease is suspected"],
                confidence=0.65,
                evidence_ids=["mri-1"],
            ),
            ApproachAssessment(
                approach=SurgicalApproach.POSTERIOR,
                suitability=ApproachSuitability.FAVORED,
                supporting_reasons=["Direct access to the affected lateral recess"],
                limiting_factors=["Motor examination is incomplete"],
                required_confirmation=["Confirm neurologic localization"],
                confidence=0.78,
                evidence_ids=["case-page-001", "mri-1"],
            ),
            ApproachAssessment(
                approach=SurgicalApproach.COMBINED,
                suitability=ApproachSuitability.DISFAVORED,
                supporting_reasons=[],
                limiting_factors=["No combined-approach indication is established"],
                required_confirmation=["Reassess only if instability is identified"],
                confidence=0.7,
                evidence_ids=["mri-1"],
            ),
        ],
        preferred_approach=SurgicalApproach.POSTERIOR,
        preference_rationale=["The proposed decompression is posterior"],
    )


def position(
    *,
    role: AgentRole,
    option_id: str = "option-1",
    action: PlanPositionAction = PlanPositionAction.SUPPORT,
    target_fields: list[str] | None = None,
    reservations: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    confidence: float = 0.6,
    draft_version: int | None = None,
    message_sequence: int | None = None,
) -> PlanPosition:
    return PlanPosition(
        role=role,
        option_id=option_id,
        action=action,
        target_fields=target_fields or [],
        reservations=reservations or [],
        evidence_ids=evidence_ids or ["case-page-001"],
        confidence=confidence,
        draft_version=draft_version,
        message_sequence=message_sequence,
    )


def anesthesia_output() -> AnesthesiologistOutput:
    return AnesthesiologistOutput(
        public_message="Anesthesia risk cannot be finalized without airway and cardiopulmonary data.",
        anesthesia_risk=RiskLevel.UNKNOWN,
        asa_classification="Not established from available records",
        proposed_anesthesia_plan="General anesthesia after airway and cardiopulmonary assessment.",
        airway_risk="Not documented",
        positioning_risk="Standard prone-position precautions are required",
        cardiopulmonary_risk="Not documented",
        blood_loss_risk="Expected to be limited for the proposed procedure",
        monitoring_recommendations=["Standard ASA monitoring"],
        transfusion_preparation=[],
        preoperative_optimization=["Complete airway and cardiopulmonary assessment"],
        recommend_delay=False,
        delay_reasons=[],
        questions_for_other_agents=[],
        position_on_plan=position(role=AgentRole.ANESTHESIOLOGIST),
        need_retrieval=False,
        retrieval_query=None,
        confidence=0.55,
        claim_evidence=[
            ClaimEvidence(
                field_path="airway_risk",
                claim="Airway risk is not documented.",
                support_type="inference",
                evidence_ids=["case-page-001"],
            )
        ],
    )


def neurology_output() -> NeurologistOutput:
    return NeurologistOutput(
        public_message="The documented pattern is compatible with right L5 radiculopathy.",
        neurologic_localization="Right L5 nerve root",
        motor_function="Not fully documented",
        sensory_function="Reported in an L5 distribution",
        reflexes="Not documented",
        sphincter_function="No dysfunction reported",
        symptom_imaging_consistency=ConsistencyAssessment.CONSISTENT,
        neurologic_deterioration_risk=RiskLevel.MODERATE,
        additional_tests=["Complete standardized motor examination"],
        preoperative_neurologic_baseline="Limited baseline consistent with right L5 radiculopathy",
        questions_for_other_agents=[],
        position_on_plan=position(role=AgentRole.NEUROLOGIST),
        need_retrieval=False,
        retrieval_query=None,
        confidence=0.7,
        claim_evidence=[
            ClaimEvidence(
                field_path="neurologic_localization",
                claim="The pattern supports right L5 localization.",
                support_type="inference",
                evidence_ids=["case-page-001", "mri-1"],
            )
        ],
    )


def perioperative_output() -> PerioperativePhysicianOutput:
    return PerioperativePhysicianOutput(
        public_message="General medical risk remains incompletely characterized.",
        major_comorbidities=[],
        cardiopulmonary_risk=RiskLevel.UNKNOWN,
        infection_risk=RiskLevel.UNKNOWN,
        thrombosis_risk=RiskLevel.MODERATE,
        nutritional_risk=RiskLevel.UNKNOWN,
        hepatic_renal_risk=RiskLevel.UNKNOWN,
        medication_adjustments=[],
        preoperative_optimization=["Complete medication reconciliation"],
        additional_tests=["CBC", "Metabolic panel"],
        recommend_delay=False,
        delay_reasons=[],
        questions_for_other_agents=[],
        position_on_plan=position(role=AgentRole.PERIOPERATIVE_PHYSICIAN),
        need_retrieval=False,
        retrieval_query=None,
        confidence=0.5,
        claim_evidence=[
            ClaimEvidence(
                field_path="public_message",
                claim="Medical risk is incompletely characterized.",
                support_type="inference",
                evidence_ids=["case-page-001", "test-page-001"],
            )
        ],
    )


def final_plan(*, force_finalized: bool = False) -> FinalPlan:
    return FinalPlan(
        radiology_summary=RadiologySummary(
            xray_findings=[],
            ct_findings=[],
            mri_findings=["Right L4-L5 disc extrusion"],
            cross_modality_conclusion="Only MRI key images are available",
            uncertainties=[],
        ),
        diagnoses=[
            DiagnosisEvidence(
                diagnosis="L4-L5 disc extrusion",
                case_evidence=["Right L5 radiculopathy"],
                neurologic_evidence=["Right L5 nerve root localization"],
                imaging_evidence=["Right L5 nerve root compression"],
                guideline_evidence_ids=[],
            )
        ],
        surgical_indication=SurgicalIndicationPlan(
            surgery_recommended=True,
            indications=["Concordant L5 radiculopathy"],
            functional_impact="Right leg pain limits walking and prolonged sitting.",
            conservative_treatment_summary=[
                "Nonoperative care did not provide durable relief."
            ],
            reasons_not_to_operate=[],
            prerequisites=[],
        ),
        recommended_surgery=RecommendedSurgery(
            procedure_name="L4-L5 decompression and discectomy",
            levels=["L4-L5"],
            approach="Posterior",
            decompression_extent="L4-L5",
            fusion_or_fixation="Not required",
            intraoperative_monitoring=["Standard monitoring"],
            key_intraoperative_considerations=["Protect the right L5 root"],
        ),
        alternatives=AlternativePlan(
            alternative_surgery=[],
            conservative_treatment=[],
            applicability_conditions=[],
            key_differences=[],
        ),
        definitive_surgery_status=DefinitiveSurgeryStatus.ESTABLISHED,
        definitive_surgery_withheld_reasons=[],
        conditional_surgical_options=[],
        approach_comparison=approach_comparison(),
        perioperative_risks=PerioperativeRiskSummary(
            anesthesia_risk="unknown",
            asa_classification="Not established from available records",
            anesthesia_plan="General anesthesia after required assessment",
            cardiopulmonary_risk="unknown",
            bleeding_risk="limited",
            positioning_risk="prone positioning precautions",
            infection_risk="unknown",
            thrombosis_risk="moderate",
            comorbidity_risk="unknown",
            mitigation_measures=[],
        ),
        additional_tests=AdditionalTests(
            laboratory=[],
            imaging=[],
            cardiopulmonary=[],
            neurophysiology=[],
            specialty_consultations=[],
            other=[],
        ),
        disagreements=[],
        final_consensus=FinalConsensus(
            recommendation="Proceed after completion of required preoperative evaluation.",
            implementation_prerequisites=[],
            remaining_concerns=[],
            has_unresolved_disagreement=False,
        ),
        confidence=0.9,
        force_finalized=force_finalized,
        claim_evidence=[
            ClaimEvidence(
                field_path=path,
                claim="Supported operative proposal.",
                support_type="proposal",
                evidence_ids=["case-page-001", "mri-1"],
            )
            for path in (
                "/recommended_surgery/procedure_name",
                "/recommended_surgery/levels",
                "/recommended_surgery/approach",
                "/recommended_surgery/decompression_extent",
                "/recommended_surgery/fusion_or_fixation",
            )
        ],
    )


def fake_ocr_engine(text: str = "Right L5 radiculopathy.") -> FakeOcrEngine:
    """Offline stand-in for the pinned OCR backend, keyed on nothing in particular.

    Case-record transcription is deterministic OCR rather than a model call, so
    it is scripted here instead of in ``all_fake_responses``.
    """

    return FakeOcrEngine(
        {},
        default=[
            OcrLine(
                text=text,
                confidence=1.0,
                top=0.1,
                left=0.1,
                width=0.7,
                height=0.02,
            )
        ],
    )


def all_fake_responses() -> dict:
    return {
        TestDocumentMarkdown: TestDocumentMarkdown(
            markdown="<!-- source: test-page-001 -->\n\nHemoglobin 128 g/L.",
        ),
        CaseSummaryMarkdown: CaseSummaryMarkdown(
            markdown="Right L5 radiculopathy with sensory change.",
        ),
        ModeratorDecision: moderator_decision(),
        RadiologyReport: radiology_report(),
        RadiologyAddendum: RadiologyAddendum(
            target_image_ids=["mri-1"],
            clarifications=[
                ImageLinkedFinding(
                    finding="Targeted recheck confirms the L4-L5 finding.",
                    image_ids=["mri-1"],
                )
            ],
            public_summary="Targeted key-image recheck confirms the prior finding.",
            confidence=0.8,
        ),
        SpineSurgeonOutput: spine_output(),
        AnesthesiologistOutput: anesthesia_output(),
        NeurologistOutput: neurology_output(),
        PerioperativePhysicianOutput: perioperative_output(),
        FinalPlan: final_plan(),
        FinalPlanReview: FinalPlanReview(
            verdict="pass",
            definitive_surgery_allowed=True,
            established_field_paths=[
                "/recommended_surgery/procedure_name",
                "/recommended_surgery/levels",
                "/recommended_surgery/approach",
                "/recommended_surgery/decompression_extent",
                "/recommended_surgery/fusion_or_fixation",
            ],
        ),
    }


def case_input(
    *,
    case_id: str,
    images: list[CaseImage],
    artifact_dir: str | None = None,
) -> CaseInput:
    return CaseInput(
        case_id=case_id,
        case_pages=[
            DocumentPage(
                page_id="case-page-001",
                page_number=1,
                path_or_url="https://example.test/case-page-001.jpg",
                original_filename="case-page-001.jpg",
            )
        ],
        test_pages=[
            DocumentPage(
                page_id="test-page-001",
                page_number=1,
                path_or_url="https://example.test/test-page-001.jpg",
                original_filename="test-page-001.jpg",
            )
        ],
        images=images,
        artifact_dir=artifact_dir or f".data/test-artifacts/{case_id}",
    )
