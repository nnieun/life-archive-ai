"""FastAPI application skeleton tests."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Life Archive AI",
        "version": "0.0.0",
    }


def test_not_found_uses_standard_error_response() -> None:
    response = client.get("/api/v1/missing")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_error"
    assert error["message"] == "Not Found"
    assert error["request_id"] == response.headers["X-Request-ID"]
