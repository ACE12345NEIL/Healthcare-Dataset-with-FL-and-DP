import json
import threading
import time
import urllib.request
import pytest
from app import HTTPServer, MediFedRequestHandler, initialize_default_model

TEST_PORT = 8089


@pytest.fixture(scope="module")
def server():
    initialize_default_model()
    httpd = HTTPServer(("", TEST_PORT), MediFedRequestHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    yield f"http://localhost:{TEST_PORT}"
    httpd.shutdown()


def test_get_index(server):
    req = urllib.request.Request(f"{server}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "MediFed-DP" in content
        assert "Healthcare Federated Learning" in content


def test_get_status(server):
    req = urllib.request.Request(f"{server}/api/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ready"


def test_train_api_laplace_dp(server):
    payload = {
        "model": "deep_resnet",
        "mechanism": "laplace",
        "epsilon": 1.0,
        "clip_norm": 0.2,
        "n_clients": 3,
        "rounds": 5,
        "local_epochs": 2,
        "lr": 0.05,
    }
    req = urllib.request.Request(
        f"{server}/api/train",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert data["dp_guarantee"]["provable"] is True
        assert len(data["history"]) == 5


def test_predict_api(server):
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 168,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    req = urllib.request.Request(
        f"{server}/api/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in (0, 1)
        assert "dp_guarantee" in data
