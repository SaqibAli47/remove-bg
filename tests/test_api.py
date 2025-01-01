import pytest
from fastapi.testclient import TestClient
from api.index import app
import base64

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_invalid_base64():
    response = client.post(
        "/remove-bg",
        json={"imageBase64": "invalid_base64"}
    )
    assert response.status_code == 500
