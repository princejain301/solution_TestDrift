import json
from fastapi.testclient import TestClient
import pytest
from drift_detector import app, psi, chi2_score

client = TestClient(app)

def test_psi_perfect():
    a = [1,2,3,4,5]
    assert psi(np.array(a), np.array(a), buckets=5) == pytest.approx(0.0, abs=1e-6)

def test_chi2_no_drift():
    base = {"a":10, "b":10}
    actual = {"a":1, "b":1}
    # identical proportions → chi2 near zero
    assert chi2_score(base, actual) < 1.0

def test_api_ok():
    payload = {
      "features": {"age": 30, "income": 50000, "tenure_months": 24, "product_category": "basic"},
      "model_version": "v1.0",
      "timestamp": "2025-05-29T10:30:00Z"
    }
    resp = client.post("/monitor/predict", json=payload)
    data = resp.json()
    assert resp.status_code == 200
    assert "status" in data
    assert "drift_scores" in data

def test_health():
    resp = client.get("/monitor/health")
    assert resp.json() == {"status":"healthy"}
