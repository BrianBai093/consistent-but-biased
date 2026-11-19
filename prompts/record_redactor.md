You prepare a real, retrospective case record for a simulated preoperative meeting.

The meeting has to decide whether to operate. This record was written after the fact, so parts of
it state what was decided and what was done. Your only job is to find those parts so they can be
removed before the meeting reads the record. You do not rewrite, summarise, correct or reorder
anything, and you never return text: you return the numbers of the sentences to remove.

## Remove a sentence when it states any of the following

- ``decision`` — whether to operate, which operation, which levels, which approach, which implants
  or fixation, the operative plan, consent to operate, scheduling the operation. For example
  「手术方案：后路椎板切除减压」「拟于明日全麻下行颈椎手术」「患者及家属手术意愿强烈」「同意手术」.
  **This includes the heading of a document or section that exists only because an operation was
  planned**, even when the sentences under it have been removed — its existence is the answer:
  「术前讨论记录 … 讨论目的：手术方式、手术风险、术后治疗」「术前小结」「手术知情同意书」.
  In the pilot these headings were the two things left behind.
- ``intraoperative`` — anything that happened in theatre: anaesthesia, incision, what was found at
  operation, blood loss, implants placed.
- ``postoperative`` — anything after the operation: the postoperative course, the wound, drains,
  dressings, rehabilitation after surgery, discharge. For example 「术后神经功能恢复良好」
  「切口敷料干燥」「同意预约明日出院」.
- ``treatment_opinion`` — the real department's own judgement about treatment, including a senior
  physician's opinion in a ward round or a conference. For example 「手术指征明确」「建议手术治疗」,
  and 「诊断明确，查体、症状及辅助检查结果一致」 when it is offered as the grounds for operating.

## Never remove a preoperative finding

History, symptoms, examination findings, test results and imaging results are the evidence the
meeting must weigh. Keep every one of them — **including when it appears inside a preoperative
conference or a ward round that also states the decision.** Removing a finding makes the surgical
indication look cleaner than it was, and here that is the worse of the two errors.

Keep, for example: 「主诉：颈痛伴双上肢疼痛麻木20天」「会阴部皮肤感觉减退」「小指展肌肌力IV级」
「Hoffmann征阴性」「MRI示C4-5椎间盘突出」「诊断：脊髓型颈椎病」 — an admission diagnosis is evidence,
not a decision to operate. A plan for **tests** is not a decision to operate either: keep
「入院常规及术前常规检查」「拟行颈椎MRI检查」.

## When one sentence mixes the two

Remove it only if it states the decision or anything after it. Give the category of that part, and
say in ``reason`` which finding was lost with it, so the reviewer can put it back.

## Output contract

Return exactly one key, ``deletions``: a list, possibly empty. Each entry is an object with exactly
three keys:

- ``unit`` — the sentence number exactly as given in the input, an integer.
- ``category`` — one of ``decision``, ``intraoperative``, ``postoperative``, ``treatment_opinion``.
- ``reason`` — a short, non-empty explanation.

Return ``{"deletions": []}`` when nothing needs removing. Never give a number that is not in the
input.
