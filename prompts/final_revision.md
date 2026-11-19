Revise an evidence-bound preoperative MDT draft exactly once.

- Apply every independent review issue; do not debate or ignore it.
- Use only the supplied original documents and structured radiology report.
- Public chat and absent specialist history are not evidence.
- Remove unsupported completed events and use "Not established from available records".
- A retained completed examination, clearance, or optimization requires documented claim_evidence
  for that exact field.
- Fill recommended_surgery value fields from the Spine Surgeon's preferred operative option; never
  invent operative content. Set the per-field *_status fields and definitive_surgery_status to
  withheld: the workflow assigns field-level statuses deterministically after review.
- Preserve valid concrete procedure and approach content in conditional_surgical_options, with
  provisional/not_established levels or fixation where appropriate.
- Never create a surgical option when the Spine Surgeon contribution is absent or states no indication.
- Correct invalid evidence ids, uncertainty, confidence caps, certainty labels, and approach-comparison
  conflicts. If no legal candidate remains after this single revision, fail closed rather than inventing one.
- Preserve auditable claim_evidence with valid source ids.
- Return only FinalPlan for research and teaching use.
