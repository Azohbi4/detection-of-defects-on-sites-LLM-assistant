import pytest
from fastapi.testclient import TestClient

from chantier_defects.api.main import create_app
from chantier_defects.config import Settings
from chantier_defects.llm.report import ReportService, TemplateReportWriter
from tests.conftest import concrete_image, png_bytes


@pytest.fixture
def client(fake_predictor, fake_retriever):
    service = ReportService(fake_retriever, TemplateReportWriter())
    app = create_app(
        predictor=fake_predictor, report_service=service, settings=Settings(max_images_per_request=3)
    )
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["classes"] == ["crack", "spallation"]


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/docs"


def test_analyze_then_report(client):
    files = [
        ("files", ("fissure.png", png_bytes(concrete_image()), "image/png")),
        ("files", ("sain.png", png_bytes(concrete_image(crack=False)), "image/png")),
    ]
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert [len(i["detections"]) for i in images] == [1, 0]
    assert images[0]["detections"][0]["defect_type"] == "crack"

    report = client.post("/report", json={"analyses": images, "context": "Pile P2, face nord"})
    assert report.status_code == 200
    body = report.json()
    assert body["severity"]["level"] == "faible"
    assert body["recommendations"] and body["sources"][0]["id"] == "S1"
    assert "Validation humaine requise" in body["disclaimer"]

    metrics = client.get("/metrics").json()
    assert metrics["requests"] == 2 and metrics["detection_score_histogram"]["0.9-1.0"] == 1


def test_rejects_non_image(client):
    resp = client.post("/analyze", files=[("files", ("note.txt", b"pas une image", "text/plain"))])
    assert resp.status_code == 400


def test_rejects_too_many_images(client):
    files = [("files", (f"{i}.png", png_bytes(concrete_image(64, 64)), "image/png")) for i in range(4)]
    assert client.post("/analyze", files=files).status_code == 413


def test_report_requires_analyses(client):
    assert client.post("/report", json={"analyses": []}).status_code == 422


def test_503_without_model(fake_retriever, tmp_path):
    settings = Settings(model_path=tmp_path / "absent.pt")
    app = create_app(report_service=ReportService(None, None), settings=settings)
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        files = [("files", ("a.png", png_bytes(concrete_image(64, 64)), "image/png"))]
        assert c.post("/analyze", files=files).status_code == 503
