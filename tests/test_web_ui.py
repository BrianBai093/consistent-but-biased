from httpx import ASGITransport, AsyncClient

from app import app


async def test_web_ui_is_served_without_provider_configuration() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        page = await client.get("/")
        styles = await client.get("/static/styles.css")
        script = await client.get("/static/app.js")

    assert page.status_code == 200
    assert "Spine MDT" in page.text
    assert "提交术前资料包" in page.text
    assert 'id="casePageInput"' in page.text
    assert 'id="testPageInput"' in page.text
    assert 'id="imageInput"' in page.text
    assert 'id="preopConfirmation"' in page.text
    assert 'id="preparedDocuments"' in page.text
    assert 'id="runPhase"' in page.text
    assert 'id="caseFileInput"' not in page.text
    assert styles.status_code == 200
    assert "discussion-panel" in styles.text
    assert script.status_code == 200
    assert 'formData.append("metadata"' in script.text
    assert 'formData.append("case_files"' in script.text
    assert 'formData.append("test_files"' in script.text
    assert 'formData.append("image_files"' in script.text
    assert "confirmed_preoperative_only" in script.text
    assert "主诊脊柱外科医生" in script.text
    assert "围术期内科医生" in script.text
    assert 'functional_impact: "功能影响"' in script.text
    assert 'asa_classification: "ASA 分级"' in script.text
    assert 'formData.append("files"' not in script.text
    assert "new EventSource" in script.text
    assert "SNAPSHOT_POLL_INTERVAL_MS = 10000" in script.text
    assert 'input_normalized: "病历逐页转写中"' in script.text
    assert 'case_document_preparation: "检验逐页转写中"' in script.text
    assert "CASE_DOCUMENT_TRANSCRIPTION_" in script.text
    assert "TEST_DOCUMENT_TRANSCRIPTION_" in script.text
    assert "independent_specialist_round" in script.text
    assert "pageId" in script.text
    assert 'isDocumentStage ? "模型请求超时"' in script.text
    assert "modelFailureCategoryLabels" in script.text
    assert "runMetadata.model_failures" in script.text
    assert "final_review_outcome" in script.text
    assert 'heading.textContent = "术式安全来源 ID"' in script.text
    assert 'addEventListener("radiology.progress"' in script.text
    assert "applyRadiologyProgress(snapshot.radiology_progress)" in script.text
    assert "正在处理${range}" in script.text
    assert '["implementation_prerequisites", "prerequisites"]' in script.text
    assert "parseImportedJson" not in script.text
    assert "convertStructuredCase" not in script.text


def test_web_ui_does_not_add_non_api_routes_to_openapi() -> None:
    paths = app.openapi()["paths"]
    assert "/" not in paths
    assert "/v1/cases" in paths
    assert not any("websocket" in path.lower() for path in paths)
