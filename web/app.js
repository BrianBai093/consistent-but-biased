"use strict";

const elements = {
  serviceState: document.querySelector("#serviceState"),
  caseForm: document.querySelector("#caseForm"),
  caseIdLabel: document.querySelector("#caseIdLabel"),
  casePageInput: document.querySelector("#casePageInput"),
  casePageDropzone: document.querySelector("#casePageDropzone"),
  casePageList: document.querySelector("#casePageList"),
  casePageCount: document.querySelector("#casePageCount"),
  testPageInput: document.querySelector("#testPageInput"),
  testPageDropzone: document.querySelector("#testPageDropzone"),
  testPageList: document.querySelector("#testPageList"),
  testPageCount: document.querySelector("#testPageCount"),
  imageInput: document.querySelector("#imageInput"),
  imageDropzone: document.querySelector("#imageDropzone"),
  imageList: document.querySelector("#imageList"),
  imageCount: document.querySelector("#imageCount"),
  pageTemplate: document.querySelector("#pageItemTemplate"),
  imageTemplate: document.querySelector("#imageItemTemplate"),
  preopConfirmation: document.querySelector("#preopConfirmation"),
  formError: document.querySelector("#formError"),
  submitButton: document.querySelector("#submitButton"),
  runBadge: document.querySelector("#runBadge"),
  runContext: document.querySelector("#runContext"),
  threadId: document.querySelector("#threadId"),
  runPhase: document.querySelector("#runPhase"),
  copyThreadButton: document.querySelector("#copyThreadButton"),
  chatTab: document.querySelector("#chatTab"),
  resultTab: document.querySelector("#resultTab"),
  chatView: document.querySelector("#chatView"),
  resultView: document.querySelector("#resultView"),
  emptyChat: document.querySelector("#emptyChat"),
  messageList: document.querySelector("#messageList"),
  messageCount: document.querySelector("#messageCount"),
  thinkingIndicator: document.querySelector("#thinkingIndicator"),
  resultPlaceholder: document.querySelector("#resultPlaceholder"),
  finalPlan: document.querySelector("#finalPlan"),
  preparedDocuments: document.querySelector("#preparedDocuments"),
  preparationWarnings: document.querySelector("#preparationWarnings"),
  casePreopDocument: document.querySelector("#casePreopDocument"),
  testsPreopDocument: document.querySelector("#testsPreopDocument"),
  caseSummaryDocument: document.querySelector("#caseSummaryDocument"),
  radiologyPreopDocument: document.querySelector("#radiologyPreopDocument"),
};

const state = {
  caseId: createCaseId(),
  casePages: [],
  testPages: [],
  images: [],
  messages: new Map(),
  threadId: null,
  status: "IDLE",
  phase: null,
  eventSource: null,
  pollTimer: null,
};

const SNAPSHOT_POLL_INTERVAL_MS = 10000;
const MISSING_RECORDS = "Not established from available records";

const phaseLabels = {
  created: "等待整理资料",
  input_normalized: "病历逐页转写中",
  case_document_preparation: "检验逐页转写中",
  document_preparation: "病例摘要生成中",
  case_summarization: "影像分批阅片准备中",
  moderator_opening: "影像分批阅片中",
  radiology_review: "等待进入 MDT 讨论",
  independent_specialist_round: "专科独立首轮采样中",
  awaiting_dynamic_routing: "正在安排下一位 Agent",
  dynamic_discussion: "MDT 术前讨论中",
  awaiting_finalization: "正在整理最终术前方案",
  completed: "已完成",
  failed: "运行失败",
  cancelled: "已取消",
};

const roleDetails = {
  moderator: { label: "MDT 主持人", initials: "M" },
  radiologist: { label: "影像科", initials: "R" },
  spine_surgeon: { label: "主诊脊柱外科医生", initials: "S" },
  anesthesiologist: { label: "麻醉科", initials: "A" },
  neurologist: { label: "神经功能专科", initials: "N" },
  perioperative_physician: { label: "围术期内科医生", initials: "P" },
};

const statusLabels = {
  IDLE: "等待病例",
  QUEUED: "已进入队列",
  RUNNING: "讨论进行中",
  PAUSE_REQUESTED: "正在暂停",
  PAUSED: "已暂停",
  INTERRUPTED: "运行已中断",
  CANCEL_REQUESTED: "正在取消",
  CANCELLED: "已取消",
  COMPLETED: "讨论完成",
  FAILED: "运行失败",
};

const modelFailureCategoryLabels = {
  MODEL_TIMEOUT: "请求超时",
  MODEL_RATE_LIMITED: "请求被限流",
  MODEL_CONNECTION_FAILED: "连接失败",
  MODEL_SERVICE_FAILED: "模型服务端失败",
  MODEL_REQUEST_REJECTED: "请求被拒绝",
  MODEL_INVALID_OUTPUT: "输出格式错误",
  FINAL_PLAN_STATE_MISMATCH: "最终方案收尾状态不一致",
  FINAL_PLAN_UNKNOWN_EVIDENCE: "最终方案引用了未知证据",
  FINAL_PLAN_OMITTED_ISSUES: "最终方案遗漏未解决问题",
  FINAL_PLAN_OMITTED_DISAGREEMENT: "最终方案遗漏分歧",
  FINAL_PLAN_DISAGREEMENT_STATUS: "最终方案分歧状态不一致",
  MODEL_INVOCATION_FAILED: "模型调用失败",
};

const modelFailureStageLabels = {
  moderator_routing: "主持人安排发言",
  specialist_contribution: "专科意见生成",
  final_plan_generation: "最终方案生成",
  final_plan_review: "独立盲审",
  final_plan_revision: "盲审后修订",
};

const fieldLabels = {
  radiology_summary: "影像总结",
  diagnoses: "诊断与依据",
  surgical_indication: "手术指征",
  recommended_surgery: "建议术式",
  definitive_surgery_status: "确定方案状态",
  definitive_surgery_withheld_reasons: "暂缓原因",
  conditional_surgical_options: "条件性候选方案",
  approach_comparison: "前路/后路/联合入路比较",
  alternatives: "替代方案",
  perioperative_risks: "围术期风险",
  additional_tests: "补充检查",
  disagreements: "尚存分歧",
  final_consensus: "最终共识",
  xray_findings: "X-ray 所见",
  ct_findings: "CT 所见",
  mri_findings: "MRI 所见",
  cross_modality_conclusion: "跨模态结论",
  uncertainties: "影像不确定性",
  diagnosis: "诊断",
  case_evidence: "病例依据",
  neurologic_evidence: "神经功能依据",
  imaging_evidence: "影像依据",
  guideline_evidence_ids: "指南证据",
  claim_evidence: "术式主张来源",
  field_path: "对应字段",
  claim: "主张",
  support_type: "支持类型",
  evidence_ids: "来源 ID",
  surgery_recommended: "是否建议手术",
  indications: "支持指征",
  functional_impact: "功能影响",
  conservative_treatment_summary: "既往保守治疗",
  reasons_not_to_operate: "暂不手术原因",
  prerequisites: "前置条件",
  procedure_name: "术式",
  levels: "节段",
  approach: "入路",
  decompression_extent: "减压范围",
  fusion_or_fixation: "融合/固定",
  procedure_status: "术式状态",
  levels_status: "节段状态",
  approach_status: "入路状态",
  decompression_status: "减压范围状态",
  fusion_fixation_status: "融合/固定状态",
  option_id: "方案 ID",
  rank: "顺位",
  levels_certainty: "节段确定性",
  fixation_certainty: "固定确定性",
  rationale: "选择理由",
  selection_conditions: "选择条件",
  reasons_against: "反对因素",
  required_confirmation: "必须补充确认",
  uncertainty: "不确定性",
  confidence: "候选置信度",
  assessments: "入路评估",
  preferred_approach: "首选入路",
  preference_rationale: "首选依据",
  suitability: "适用性",
  supporting_reasons: "支持理由",
  limiting_factors: "限制因素",
  intraoperative_monitoring: "术中监测",
  key_intraoperative_considerations: "术中要点",
  alternative_surgery: "其他手术路径",
  conservative_treatment: "保守治疗",
  applicability_conditions: "适用条件",
  key_differences: "关键差异",
  anesthesia_risk: "麻醉风险",
  asa_classification: "ASA 分级",
  anesthesia_plan: "拟定麻醉方案",
  cardiopulmonary_risk: "心肺风险",
  bleeding_risk: "出血风险",
  positioning_risk: "体位风险",
  infection_risk: "感染风险",
  thrombosis_risk: "血栓风险",
  comorbidity_risk: "合并症风险",
  mitigation_measures: "风险缓解措施",
  laboratory: "实验室检查",
  imaging: "影像检查",
  cardiopulmonary: "心肺评估",
  neurophysiology: "神经功能检查",
  specialty_consultations: "专科会诊",
  other: "其他",
  recommendation: "综合建议",
  implementation_prerequisites: "实施前提",
  remaining_concerns: "待解决问题",
  has_unresolved_disagreement: "是否仍有分歧",
  agents: "涉及专业",
  issue: "议题",
  positions: "各方观点",
  moderator_resolution: "主持人结论",
  resolved: "是否解决",
  unresolved_reason: "未解决原因",
};

initialize();

function initialize() {
  elements.caseIdLabel.textContent = state.caseId;
  bindEvents();
  checkHealth();
  restoreLatestRun();
}

function bindEvents() {
  bindUploadZone("casePages", elements.casePageInput, elements.casePageDropzone);
  bindUploadZone("testPages", elements.testPageInput, elements.testPageDropzone);
  bindUploadZone("images", elements.imageInput, elements.imageDropzone);
  elements.caseForm.addEventListener("submit", submitCase);
  elements.copyThreadButton.addEventListener("click", copyThreadId);
  elements.chatTab.addEventListener("click", () => switchView("chat"));
  elements.resultTab.addEventListener("click", () => switchView("result"));
}

function bindUploadZone(category, input, dropzone) {
  input.addEventListener("change", (event) => {
    addFiles(category, event.target.files);
    input.value = "";
  });
  for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  }
  dropzone.addEventListener("drop", (event) => addFiles(category, event.dataTransfer.files));
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("health check failed");
    elements.serviceState.className = "service-state online";
    elements.serviceState.lastElementChild.textContent = "服务运行正常";
  } catch {
    elements.serviceState.className = "service-state offline";
    elements.serviceState.lastElementChild.textContent = "服务不可用";
  }
}

function createCaseId() {
  const now = new Date();
  const date = [now.getFullYear(), now.getMonth() + 1, now.getDate()]
    .map((value, index) => (index === 0 ? String(value) : String(value).padStart(2, "0")))
    .join("");
  const suffix = crypto.randomUUID().replaceAll("-", "").slice(0, 8);
  return `case-${date}-${suffix}`;
}

function addFiles(category, fileList) {
  clearFormError();
  if (category === "images") syncImageMetadataFromDom();
  const acceptedTypes = new Set(["image/png", "image/jpeg"]);
  const incoming = Array.from(fileList || []);
  if (incoming.some((file) => !acceptedTypes.has(file.type))) {
    showFormError("仅支持 PNG、JPG 或 JPEG 图片；其他文件已忽略。");
  }
  const target = state[category];
  for (const file of incoming.filter((item) => acceptedTypes.has(item.type))) {
    const duplicate = target.some(
      (item) => item.file.name === file.name && item.file.size === file.size
        && item.file.lastModified === file.lastModified,
    );
    if (duplicate) continue;
    target.push({
      id: crypto.randomUUID(),
      file,
      previewUrl: URL.createObjectURL(file),
      metadata: category === "images" ? { modality: inferModality(file.name) } : {},
    });
  }
  if (category === "images") renderImages();
  else renderPageFiles(category);
}

function inferModality(filename) {
  const normalized = filename.toLowerCase();
  if (normalized.includes("mri") || normalized.includes("磁共振")) return "MRI";
  if (normalized.includes("ct")) return "CT";
  if (normalized.includes("x-ray") || normalized.includes("xray") || normalized.includes("x光")) return "XRAY";
  return "";
}

function renderPageFiles(category) {
  const files = state[category];
  const list = category === "casePages" ? elements.casePageList : elements.testPageList;
  const count = category === "casePages" ? elements.casePageCount : elements.testPageCount;
  list.replaceChildren();
  count.textContent = `${files.length} 张`;
  files.forEach((entry, index) => {
    const fragment = elements.pageTemplate.content.cloneNode(true);
    fragment.querySelector(".image-preview").src = entry.previewUrl;
    fragment.querySelector(".image-preview").alt = `第 ${index + 1} 页：${entry.file.name}`;
    fragment.querySelector(".image-name").textContent = `${index + 1}. ${entry.file.name}`;
    fragment.querySelector(".image-size").textContent = formatFileSize(entry.file.size);
    fragment.querySelector(".remove-image").addEventListener("click", () => {
      URL.revokeObjectURL(entry.previewUrl);
      state[category] = state[category].filter((item) => item.id !== entry.id);
      renderPageFiles(category);
    });
    list.append(fragment);
  });
}

function syncImageMetadataFromDom() {
  for (const item of elements.imageList.querySelectorAll(".image-item")) {
    const image = state.images.find((candidate) => candidate.id === item.dataset.imageId);
    if (!image) continue;
    image.metadata = {
      ...(image.metadata || {}),
      modality: item.querySelector(".modality-select").value || null,
      body_region: valueOrNull(item.querySelector(".body-region-input").value),
      view_or_plane: valueOrNull(item.querySelector(".plane-input").value),
      sequence: valueOrNull(item.querySelector(".sequence-input").value),
      level: valueOrNull(item.querySelector(".level-input").value),
      description: valueOrNull(item.querySelector(".description-input").value),
    };
  }
}

function renderImages() {
  elements.imageList.replaceChildren();
  elements.imageCount.textContent = `${state.images.length} 张`;
  state.images.forEach((image, index) => {
    const fragment = elements.imageTemplate.content.cloneNode(true);
    const item = fragment.querySelector(".image-item");
    item.dataset.imageId = image.id;
    fragment.querySelector(".image-preview").src = image.previewUrl;
    fragment.querySelector(".image-preview").alt = `术前影像 ${index + 1}：${image.file.name}`;
    fragment.querySelector(".image-name").textContent = image.file.name;
    fragment.querySelector(".image-size").textContent = formatFileSize(image.file.size);
    fragment.querySelector(".modality-select").value = image.metadata?.modality || "";
    fragment.querySelector(".body-region-input").value = displayMetadataValue(image.metadata?.body_region);
    fragment.querySelector(".plane-input").value = displayMetadataValue(image.metadata?.view_or_plane);
    fragment.querySelector(".sequence-input").value = displayMetadataValue(image.metadata?.sequence);
    fragment.querySelector(".level-input").value = displayMetadataValue(image.metadata?.level);
    fragment.querySelector(".description-input").value = displayMetadataValue(image.metadata?.description);
    fragment.querySelector(".remove-image").addEventListener("click", () => removeImage(image.id));
    elements.imageList.append(fragment);
  });
}

function displayMetadataValue(value) {
  return value && value !== "Not provided" ? String(value) : "";
}

function removeImage(imageId) {
  syncImageMetadataFromDom();
  const image = state.images.find((item) => item.id === imageId);
  if (image) URL.revokeObjectURL(image.previewUrl);
  state.images = state.images.filter((item) => item.id !== imageId);
  renderImages();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function submitCase(event) {
  event.preventDefault();
  clearFormError();
  syncImageMetadataFromDom();
  if (!state.casePages.length || !state.testPages.length || !state.images.length) {
    showFormError("病历页、检验页和术前影像均至少需要一张。");
    return;
  }
  if (!elements.preopConfirmation.checked) {
    showFormError("请确认已经人工排除手术、术中、术后和随访资料。");
    elements.preopConfirmation.focus();
    return;
  }

  const imageItems = Array.from(elements.imageList.querySelectorAll(".image-item"));
  const metadataImages = [];
  for (let index = 0; index < imageItems.length; index += 1) {
    const item = imageItems[index];
    const modality = item.querySelector(".modality-select").value;
    if (!modality) {
      showFormError(`请为第 ${index + 1} 张术前影像选择 MRI、CT 或 X-ray。`);
      item.querySelector(".modality-select").focus();
      return;
    }
    metadataImages.push({
      image_id: `${state.caseId}-image-${String(index + 1).padStart(3, "0")}`,
      modality,
      body_region: valueOrNull(item.querySelector(".body-region-input").value),
      view_or_plane: valueOrNull(item.querySelector(".plane-input").value),
      sequence: valueOrNull(item.querySelector(".sequence-input").value),
      level: valueOrNull(item.querySelector(".level-input").value),
      description: valueOrNull(item.querySelector(".description-input").value),
    });
  }

  const metadata = {
    case_id: state.caseId,
    confirmed_preoperative_only: true,
    images: metadataImages,
  };
  const formData = new FormData();
  formData.append("metadata", JSON.stringify(metadata));
  for (const entry of state.casePages) formData.append("case_files", entry.file, entry.file.name);
  for (const entry of state.testPages) formData.append("test_files", entry.file, entry.file.name);
  for (const entry of state.images) formData.append("image_files", entry.file, entry.file.name);

  setSubmitting(true);
  resetRunDisplay();
  try {
    const response = await fetch("/v1/cases", { method: "POST", body: formData });
    const body = await readJsonSafely(response);
    if (!response.ok) throw new Error(body?.message || "病例资料包提交失败。");
    state.threadId = body.thread_id;
    localStorage.setItem("spine-mdt-thread-id", state.threadId);
    applySnapshot(body);
    connectEvents();
  } catch (error) {
    showFormError(error instanceof Error ? error.message : "病例资料包提交失败。");
    setStatus("FAILED");
  } finally {
    setSubmitting(false);
  }
}

function valueOrNull(value) {
  return value.trim() || null;
}

async function readJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function setSubmitting(active) {
  elements.submitButton.disabled = active;
  elements.submitButton.firstElementChild.textContent = active
    ? "正在上传资料…"
    : "整理资料并发起 MDT";
}

function showFormError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function clearFormError() {
  elements.formError.hidden = true;
  elements.formError.textContent = "";
}

function resetRunDisplay() {
  if (state.eventSource) state.eventSource.close();
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.messages.clear();
  state.threadId = null;
  state.phase = null;
  elements.messageList.replaceChildren();
  elements.messageCount.textContent = "0";
  elements.emptyChat.hidden = false;
  elements.finalPlan.replaceChildren();
  elements.resultPlaceholder.hidden = false;
  elements.resultTab.disabled = true;
  elements.runContext.hidden = true;
  elements.runPhase.textContent = "等待开始";
  elements.preparedDocuments.hidden = true;
  switchView("chat");
  setStatus("QUEUED");
}

function applySnapshot(snapshot) {
  if (!snapshot) return;
  state.threadId = snapshot.thread_id || state.threadId;
  if (state.threadId) {
    elements.threadId.textContent = state.threadId;
    elements.runContext.hidden = false;
  }
  setStatus(snapshot.status || state.status);
  state.phase = snapshot.phase || state.phase;
  elements.runPhase.textContent = phaseLabels[state.phase] || "正在处理";
  if (!isTerminal(snapshot.status)) applyRadiologyProgress(snapshot.radiology_progress);
  renderPreparedDocuments(snapshot.prepared_documents);
  for (const message of snapshot.new_messages || []) appendMessage(message);
  if (snapshot.final_plan) renderFinalPlan(snapshot.final_plan, snapshot.run_metadata);
  if (snapshot.status === "FAILED" && snapshot.error_message) {
    showFormError(formatRunError(snapshot));
  }
}

function formatRunError(snapshot) {
  const code = snapshot.error_code || "";
  const pageId = snapshot.error_message?.match(/\b(?:case|test)-page-\d{3}\b/)?.[0];
  const isDocumentStage = code.startsWith("CASE_DOCUMENT_TRANSCRIPTION_")
    || code.startsWith("TEST_DOCUMENT_TRANSCRIPTION_");
  const stage = code.startsWith("CASE_DOCUMENT_TRANSCRIPTION_")
    ? "病历转写"
    : code.startsWith("TEST_DOCUMENT_TRANSCRIPTION_")
      ? "检验转写"
      : code.startsWith("CASE_SUMMARIZATION_")
        ? "病例摘要"
        : code.startsWith("RADIOLOGY_REVIEW_")
          ? "影像阅片"
          : null;
  if (!stage) return snapshot.error_message;
  const reason = code.endsWith("MODEL_TIMEOUT")
    ? isDocumentStage ? "模型请求超时" : "模型请求重试后仍超时"
    : code.endsWith("MODEL_RATE_LIMITED")
      ? isDocumentStage ? "模型服务限流" : "模型服务持续限流"
      : code.endsWith("MODEL_CONNECTION_FAILED")
        ? "无法连接模型服务"
        : code.endsWith("MODEL_SERVICE_FAILED")
          ? isDocumentStage ? "模型服务返回错误" : "模型服务持续返回错误"
          : code.endsWith("MODEL_REQUEST_REJECTED")
            ? "模型服务拒绝了请求，请检查权限或请求配置"
            : code.endsWith("MODEL_INVALID_OUTPUT")
              ? isDocumentStage ? "模型返回无效结果" : "模型多次返回无效结果"
              : "模型调用失败";
  const pageLabel = pageId ? `（${pageId}）` : "";
  return `${stage}失败${pageLabel}：${reason}。本次运行未进入 MDT 讨论。`;
}

function renderPreparedDocuments(documents) {
  if (!documents) return;
  const hasContent = [
    documents.case_preop_md,
    documents.tests_preop_md,
    documents.case_summary_md,
    documents.radiology_preop_md,
  ].some(Boolean);
  elements.preparedDocuments.hidden = !hasContent;
  elements.casePreopDocument.textContent = documents.case_preop_md || "尚未生成";
  elements.testsPreopDocument.textContent = documents.tests_preop_md || "尚未生成";
  elements.caseSummaryDocument.textContent = documents.case_summary_md || "尚未生成";
  elements.radiologyPreopDocument.textContent = documents.radiology_preop_md || "尚未生成";
  const warnings = documents.warnings || [];
  elements.preparationWarnings.textContent = warnings.length
    ? `转写警告：${warnings.join("；")}`
    : "";
}

function applyRadiologyProgress(progress) {
  if (!progress || isTerminal(state.status)) return;
  const completed = progress.completed_images ?? 0;
  const total = progress.total_images ?? 0;
  if (progress.stage === "reviewing") {
    const start = progress.current_image_start;
    const end = progress.current_image_end;
    const range = start === end ? `第 ${start}/${total} 张` : `第 ${start}–${end}/${total} 张`;
    elements.runPhase.textContent = `影像分批阅片中：正在处理${range}`;
    return;
  }
  if (progress.stage === "batch_completed") {
    elements.runPhase.textContent = `影像分批阅片中：已完成 ${completed}/${total} 张`;
    return;
  }
  if (progress.stage === "synthesizing") {
    elements.runPhase.textContent = `已完成 ${completed}/${total} 张影像，正在汇总影像报告`;
    return;
  }
  if (progress.stage === "completed") {
    elements.runPhase.textContent = `影像阅片完成（${completed}/${total}），准备进入 MDT`;
  }
}

function connectEvents() {
  if (!state.threadId || isTerminal(state.status)) return;
  if (state.eventSource) state.eventSource.close();
  const eventSource = new EventSource(`/v1/runs/${encodeURIComponent(state.threadId)}/events`);
  state.eventSource = eventSource;
  scheduleSnapshotRefresh();

  eventSource.addEventListener("chat.message", (event) => {
    appendMessage(JSON.parse(event.data));
    elements.thinkingIndicator.hidden = false;
    scrollChatToBottom();
  });

  eventSource.addEventListener("radiology.progress", (event) => {
    applyRadiologyProgress(JSON.parse(event.data));
  });

  for (const eventType of ["run.started", "run.resumed"]) {
    eventSource.addEventListener(eventType, () => setStatus("RUNNING"));
  }
  eventSource.addEventListener("run.pause_requested", () => setStatus("PAUSE_REQUESTED"));
  eventSource.addEventListener("run.paused", () => setStatus("PAUSED"));
  eventSource.addEventListener("run.interrupted", () => setStatus("INTERRUPTED"));
  eventSource.addEventListener("run.cancel_requested", () => setStatus("CANCEL_REQUESTED"));
  eventSource.addEventListener("run.cancelled", () => {
    setStatus("CANCELLED");
    eventSource.close();
  });
  eventSource.addEventListener("run.completed", async () => {
    setStatus("COMPLETED");
    eventSource.close();
    await fetchResult();
  });
  eventSource.addEventListener("run.failed", async () => {
    setStatus("FAILED");
    eventSource.close();
    await refreshSnapshot();
  });
  eventSource.onerror = () => {
    if (!isTerminal(state.status)) scheduleSnapshotRefresh();
  };
}

function scheduleSnapshotRefresh() {
  if (state.pollTimer) return;
  state.pollTimer = window.setTimeout(async () => {
    state.pollTimer = null;
    await refreshSnapshot();
  }, SNAPSHOT_POLL_INTERVAL_MS);
}

async function refreshSnapshot() {
  if (!state.threadId) return;
  try {
    const response = await fetch(`/v1/runs/${encodeURIComponent(state.threadId)}?after_sequence=0`);
    if (!response.ok) return;
    const snapshot = await response.json();
    applySnapshot(snapshot);
    if (snapshot.status === "COMPLETED" && snapshot.final_plan) switchView("result");
  } catch {
    // EventSource will keep trying; a transient network failure needs no UI alarm.
  } finally {
    if (state.threadId && !isTerminal(state.status)) scheduleSnapshotRefresh();
  }
}

async function fetchResult() {
  if (!state.threadId) return;
  try {
    const response = await fetch(`/v1/runs/${encodeURIComponent(state.threadId)}/result`);
    const body = await readJsonSafely(response);
    if (!response.ok) throw new Error(body?.message || "最终方案读取失败。 ");
    applySnapshot(body);
    switchView("result");
  } catch (error) {
    showFormError(error instanceof Error ? error.message : "最终方案读取失败。 ");
  }
}

async function restoreLatestRun() {
  const threadId = localStorage.getItem("spine-mdt-thread-id");
  if (!threadId) return;
  state.threadId = threadId;
  try {
    const response = await fetch(`/v1/runs/${encodeURIComponent(threadId)}?after_sequence=0`);
    if (response.status === 404) {
      localStorage.removeItem("spine-mdt-thread-id");
      state.threadId = null;
      return;
    }
    if (!response.ok) return;
    applySnapshot(await response.json());
    if (state.status === "COMPLETED") switchView("result");
    else connectEvents();
  } catch {
    // The health indicator communicates that the service cannot currently be reached.
  }
}

function setStatus(status) {
  state.status = status;
  elements.runBadge.textContent = statusLabels[status] || status;
  elements.runBadge.className = `run-badge ${String(status).toLowerCase()}`;
  elements.thinkingIndicator.hidden = status !== "RUNNING";
}

function isTerminal(status) {
  return ["COMPLETED", "FAILED", "CANCELLED"].includes(status);
}

function appendMessage(message) {
  if (!message || state.messages.has(message.sequence)) return;
  state.messages.set(message.sequence, message);
  renderMessages();
}

function renderMessages() {
  elements.messageList.replaceChildren();
  const messages = Array.from(state.messages.values()).sort((a, b) => a.sequence - b.sequence);
  elements.emptyChat.hidden = messages.length > 0;
  elements.messageCount.textContent = String(messages.length);

  for (const message of messages) {
    const role = roleDetails[message.speaker_role] || {
      label: message.speaker_name || message.speaker_role,
      initials: "A",
    };
    const article = document.createElement("article");
    article.className = "message";
    article.dataset.role = message.speaker_role;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role.initials;
    avatar.setAttribute("aria-hidden", "true");

    const body = document.createElement("div");
    body.className = "message-body";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const name = document.createElement("strong");
    name.textContent = role.label;
    const time = document.createElement("span");
    time.textContent = formatTime(message.timestamp, message.sequence);
    meta.append(name, time);

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;
    body.append(meta, content);

    if (message.confidence !== null && message.confidence !== undefined) {
      const details = document.createElement("div");
      details.className = "message-details";
      details.textContent = `置信度 ${Math.round(message.confidence * 100)}%`;
      body.append(details);
    }
    article.append(avatar, body);
    elements.messageList.append(article);
  }
}

function formatTime(timestamp, sequence) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return `#${sequence}`;
  return `#${sequence} · ${new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date)}`;
}

function scrollChatToBottom() {
  requestAnimationFrame(() => {
    elements.chatView.scrollTo({ top: elements.chatView.scrollHeight, behavior: "smooth" });
  });
}

function switchView(view) {
  const showResult = view === "result" && !elements.resultTab.disabled;
  elements.chatView.hidden = showResult;
  elements.resultView.hidden = !showResult;
  elements.chatTab.classList.toggle("active", !showResult);
  elements.chatTab.setAttribute("aria-selected", String(!showResult));
  elements.resultTab.classList.toggle("active", showResult);
  elements.resultTab.setAttribute("aria-selected", String(showResult));
}

function renderFinalPlan(plan, runMetadata) {
  elements.resultPlaceholder.hidden = true;
  elements.finalPlan.replaceChildren();
  elements.resultTab.disabled = false;

  const hero = document.createElement("section");
  hero.className = "plan-hero";
  const heroTop = document.createElement("div");
  heroTop.className = "plan-hero-top";
  const copy = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = plan.definitive_surgery_status === "established"
    ? "已形成确定术前方案"
    : plan.definitive_surgery_status === "partial"
      ? "已形成部分确定手术方案"
      : (plan.conditional_surgical_options || []).length
        ? "已形成条件性手术方案"
        : "当前未形成具体手术方案";
  const recommendation = document.createElement("p");
  recommendation.textContent = plan.final_consensus?.recommendation || "讨论已完成。";
  copy.append(title, recommendation);
  const confidence = document.createElement("div");
  confidence.className = "confidence-ring";
  confidence.textContent = `${Math.round((plan.confidence || 0) * 100)}%`;
  confidence.title = "最终方案置信度";
  heroTop.append(copy, confidence);

  const flags = document.createElement("div");
  flags.className = "plan-flags";
  flags.append(createFlag(plan.force_finalized ? "强制收尾" : "正常收尾"));
  if (runMetadata?.final_review_outcome === "passed") {
    flags.append(createFlag("盲审通过"));
  } else if (runMetadata?.final_review_outcome === "revised") {
    flags.append(createFlag("盲审后已修订"));
  }
  if (runMetadata?.finalization_fallback_used) flags.append(createFlag("保守降级方案"));
  if (runMetadata?.radiology_cache_hit) flags.append(createFlag("已复用影像缓存"));
  if (runMetadata?.specialist_message_count !== undefined) {
    flags.append(createFlag(`${runMetadata.specialist_message_count} 次专业发言`));
  }
  if (runMetadata?.model_failures?.length) {
    flags.append(createFlag(`${runMetadata.model_failures.length} 条模型异常`));
  }
  hero.append(heroTop, flags);
  elements.finalPlan.append(hero);

  if (runMetadata?.model_failures?.length) {
    elements.finalPlan.append(renderModelFailures(runMetadata.model_failures));
  }

  const safeEvidenceIds = Array.from(
    new Set([
      ...(plan.claim_evidence || []).flatMap((item) => item.evidence_ids || []),
      ...(plan.conditional_surgical_options || []).flatMap((item) => item.evidence_ids || []),
      ...(plan.approach_comparison?.assessments || []).flatMap(
        (item) => item.evidence_ids || [],
      ),
    ]),
  );
  if (safeEvidenceIds.length) {
    const evidenceSection = document.createElement("section");
    evidenceSection.className = "plan-section";
    const heading = document.createElement("h4");
    heading.textContent = "术式安全来源 ID";
    evidenceSection.append(heading);
    appendStructuredValue(evidenceSection, safeEvidenceIds, "evidence_ids");
    elements.finalPlan.append(evidenceSection);
  }

  renderDefinitiveSurgery(plan);
  renderConditionalOptions(
    plan.conditional_surgical_options || [],
    !["established", "partial"].includes(plan.definitive_surgery_status),
  );
  renderApproachComparison(plan.approach_comparison);

  const sections = [
    ["final_consensus", plan.final_consensus],
    ["radiology_summary", plan.radiology_summary],
    ["diagnoses", plan.diagnoses],
    ["surgical_indication", plan.surgical_indication],
    ["perioperative_risks", plan.perioperative_risks],
    ["additional_tests", plan.additional_tests],
    ["disagreements", plan.disagreements],
  ];

  for (const [key, value] of sections) {
    if (isEmptyValue(value)) continue;
    const section = document.createElement("section");
    section.className = "plan-section";
    const heading = document.createElement("h4");
    heading.textContent = fieldLabels[key] || humanizeKey(key);
    section.append(heading);
    appendStructuredValue(section, value, key);
    elements.finalPlan.append(section);
  }
}

function renderDefinitiveSurgery(plan) {
  const legacyEstablished =
    plan.definitive_surgery_status === "withheld" &&
    !(plan.definitive_surgery_withheld_reasons || []).length &&
    !(plan.conditional_surgical_options || []).length &&
    !plan.approach_comparison &&
    plan.recommended_surgery?.procedure_name &&
    plan.recommended_surgery.procedure_name !== MISSING_RECORDS;
  const status = legacyEstablished ? "established" : plan.definitive_surgery_status;
  const established = status === "established";
  const partial = status === "partial";
  const section = document.createElement("section");
  section.className = `plan-section definitive-section ${established || partial ? "established" : "withheld"}`;
  const heading = document.createElement("h4");
  heading.textContent = established ? "确定方案" : partial ? "部分确定方案（逐字段状态）" : "确定方案暂缓";
  section.append(heading);
  if (established || partial) {
    appendStructuredValue(section, plan.recommended_surgery, "recommended_surgery");
    if ((plan.definitive_surgery_withheld_reasons || []).length) {
      appendStructuredValue(
        section,
        plan.definitive_surgery_withheld_reasons,
        "definitive_surgery_withheld_reasons",
      );
    }
  } else {
    const reasons = plan.definitive_surgery_withheld_reasons || [];
    appendStructuredValue(
      section,
      reasons.length ? reasons : ["当前证据不足以发布确定术式。"],
      "definitive_surgery_withheld_reasons",
    );
  }
  elements.finalPlan.append(section);
}

function renderConditionalOptions(options, hasConditionalPreferred) {
  if (!options.length) return;
  const section = document.createElement("section");
  section.className = "plan-section";
  const heading = document.createElement("h4");
  heading.textContent = "条件性手术方案";
  section.append(heading);
  [...options]
    .sort((left, right) => left.rank - right.rank)
    .slice(0, 3)
    .forEach((option, index) => section.append(
      renderSurgicalOption(option, hasConditionalPreferred && index === 0),
    ));
  elements.finalPlan.append(section);
}

function renderSurgicalOption(option, preferred) {
  const card = document.createElement("article");
  card.className = `surgical-option ${option.uncertainty || "high"}`;
  const header = document.createElement("div");
  header.className = "option-header";
  const title = document.createElement("strong");
  title.textContent = `${preferred ? "条件性首选" : `备选 ${option.rank}`} · ${option.procedure_name}`;
  const badges = document.createElement("div");
  badges.className = "option-badges";
  badges.append(createOptionBadge(uncertaintyLabel(option.uncertainty), option.uncertainty));
  if (option.levels_certainty !== "established") {
    badges.append(createOptionBadge(`节段 ${certaintyLabel(option.levels_certainty)}`, "provisional"));
  }
  if (option.fixation_certainty !== "established") {
    badges.append(createOptionBadge(`固定 ${certaintyLabel(option.fixation_certainty)}`, "provisional"));
  }
  header.append(title, badges);
  card.append(header);
  if (option.uncertainty === "very_high") {
    const warning = document.createElement("p");
    warning.className = "option-warning";
    warning.textContent = "极高不确定性，仅供进一步评估；不得作为临床手术指令。";
    card.append(warning);
  }
  const details = { ...option };
  delete details.option_id;
  delete details.rank;
  delete details.procedure_name;
  delete details.uncertainty;
  appendStructuredValue(card, details, "conditional_surgical_options");
  return card;
}

function renderApproachComparison(comparison) {
  if (!comparison?.assessments?.length) return;
  const section = document.createElement("section");
  section.className = "plan-section";
  const heading = document.createElement("h4");
  heading.textContent = "前路 / 后路 / 联合入路对照";
  section.append(heading);
  const table = document.createElement("div");
  table.className = "approach-table";
  comparison.assessments.forEach((assessment) => {
    const row = document.createElement("article");
    row.className = assessment.approach === comparison.preferred_approach ? "preferred" : "";
    const title = document.createElement("strong");
    title.textContent = `${approachLabel(assessment.approach)} · ${suitabilityLabel(assessment.suitability)}`;
    row.append(title);
    appendStructuredValue(row, {
      supporting_reasons: assessment.supporting_reasons,
      limiting_factors: assessment.limiting_factors,
      required_confirmation: assessment.required_confirmation,
      confidence: assessment.confidence,
      evidence_ids: assessment.evidence_ids,
    }, "approach_comparison");
    table.append(row);
  });
  section.append(table);
  const rationale = document.createElement("div");
  rationale.className = "plan-field";
  const label = document.createElement("span");
  label.className = "plan-field-label";
  label.textContent = `首选 ${approachLabel(comparison.preferred_approach)} 的依据`;
  rationale.append(label);
  appendStructuredValue(rationale, comparison.preference_rationale || [], "preference_rationale");
  section.append(rationale);
  elements.finalPlan.append(section);
}

function createOptionBadge(text, kind) {
  const badge = document.createElement("span");
  badge.className = `option-badge ${kind || ""}`;
  badge.textContent = text;
  return badge;
}

function uncertaintyLabel(value) {
  return { moderate: "中度不确定性", high: "高不确定性", very_high: "极高不确定性" }[value] || value;
}

function certaintyLabel(value) {
  return { established: "已确定", provisional: "暂定", not_established: "未确定" }[value] || value;
}

function approachLabel(value) {
  return { anterior: "前路", posterior: "后路", combined: "联合入路" }[value] || value;
}

function suitabilityLabel(value) {
  return { favored: "优先", reasonable: "可考虑", disfavored: "不优先", indeterminate: "待判断" }[value] || value;
}

function renderModelFailures(failures) {
  const section = document.createElement("section");
  section.className = "plan-section";
  const heading = document.createElement("h4");
  heading.textContent = "模型调用异常";
  section.append(heading);

  for (const failure of failures) {
    const card = document.createElement("div");
    card.className = "nested-card";
    const title = document.createElement("strong");
    const role = roleDetails[failure.agent_role]?.label || failure.agent_role;
    const outcome = failure.outcome === "recovered" ? "重试后恢复" : "最终失败";
    title.textContent = `${role} · ${outcome}`;

    const summary = document.createElement("p");
    const stage = modelFailureStageLabels[failure.stage] || failure.stage;
    const category = modelFailureCategoryLabels[failure.category] || failure.category;
    summary.textContent = `${stage}：${category}；共尝试 ${failure.attempt_count}/${failure.max_attempts} 次。`;

    const attempts = document.createElement("p");
    attempts.textContent = `失败轨迹：${(failure.attempt_categories || [])
      .map((item) => modelFailureCategoryLabels[item] || item)
      .join(" → ")}`;
    card.append(title, summary, attempts);
    section.append(card);
  }
  return section;
}

function createFlag(text) {
  const flag = document.createElement("span");
  flag.className = "plan-flag";
  flag.textContent = text;
  return flag;
}

function appendStructuredValue(container, value, parentKey = "") {
  if (Array.isArray(value)) {
    const cleaned = cleanArray(value, parentKey);
    if (!cleaned.length) return;
    if (cleaned.every((item) => typeof item !== "object" || item === null)) {
      container.append(createList(cleaned));
      return;
    }
    cleaned.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "nested-card";
      const cardTitle = document.createElement("strong");
      cardTitle.textContent = item?.diagnosis || item?.issue || `条目 ${index + 1}`;
      card.append(cardTitle);
      appendStructuredValue(card, item, parentKey);
      container.append(card);
    });
    return;
  }

  if (value && typeof value === "object") {
    for (const [key, nestedValue] of Object.entries(value)) {
      if (isEmptyValue(nestedValue) || (parentKey === "diagnoses" && key === "diagnosis")) continue;
      const field = document.createElement("div");
      field.className = "plan-field";
      const label = document.createElement("span");
      label.className = "plan-field-label";
      label.textContent = fieldLabels[key] || humanizeKey(key);
      field.append(label);
      appendStructuredValue(field, nestedValue, key);
      container.append(field);
    }
    return;
  }

  const paragraph = document.createElement("p");
  paragraph.textContent = typeof value === "boolean" ? (value ? "是" : "否") : String(value);
  container.append(paragraph);
}

function createList(items) {
  const list = document.createElement("ul");
  list.className = "plan-list";
  const visible = items.slice(0, 12);
  for (const item of visible) {
    const listItem = document.createElement("li");
    listItem.textContent = String(item);
    list.append(listItem);
  }
  if (items.length > visible.length) {
    const detailsItem = document.createElement("li");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `查看其余 ${items.length - visible.length} 项`;
    const extra = document.createElement("ul");
    extra.className = "plan-list";
    for (const item of items.slice(visible.length)) {
      const extraItem = document.createElement("li");
      extraItem.textContent = String(item);
      extra.append(extraItem);
    }
    details.append(summary, extra);
    detailsItem.append(details);
    list.append(detailsItem);
  }
  return list;
}

function cleanArray(items, parentKey) {
  if (!items.every((item) => typeof item !== "object" || item === null)) return items;
  const seen = new Set();
  return items.filter((item) => {
    const text = String(item).trim();
    if (!text) return false;
    if (["implementation_prerequisites", "prerequisites"].includes(parentKey)) {
      if (/^Request .+ could not be fulfilled/i.test(text) || /^Termination reason:/i.test(text)) return false;
    }
    const normalized = text.toLocaleLowerCase().replace(/[\s.,;:()\[\]-]+/g, " ").trim();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

function isEmptyValue(value) {
  return value === null || value === undefined || value === "" ||
    (Array.isArray(value) && value.length === 0) ||
    (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0);
}

function humanizeKey(key) {
  return key.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

async function copyThreadId() {
  if (!state.threadId) return;
  try {
    await navigator.clipboard.writeText(state.threadId);
    elements.copyThreadButton.textContent = "已复制";
    window.setTimeout(() => { elements.copyThreadButton.textContent = "复制"; }, 1200);
  } catch {
    elements.copyThreadButton.textContent = "复制失败";
  }
}
