#!/usr/bin/env python3
import json
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from scipy.stats import chi2_contingency
import numpy as np

# ————— Logging Setup —————————
logger = logging.getLogger("drift_detector")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    json.dumps({
        "time": "%(asctime)s",
        "level": "%(levelname)s",
        "message": "%(message)s"
    })
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ————— Prometheus Metrics —————————
REQUEST_COUNT = Counter("drift_requests_total", "Total drift requests", ["status"])
DRIFT_SCORE = Gauge("feature_drift_score", "Latest drift score", ["feature"])
DRIFT_ALERT = Counter("drift_alerts_total", "Drift alerts triggered", ["severity"])

# ————— Configurable Thresholds —————————
WARNING_THRESHOLD = 0.2
CRITICAL_THRESHOLD = 0.5

# ————— Load Baseline —————————
with open("baseline_data.json") as f:
    baseline = json.load(f)

numerical_features = baseline["numerical"].keys()
categorical_features = baseline["categorical"].keys()

# ————— Pydantic Models —————————
class FeaturePayload(BaseModel):
    features: Dict[str, Any] = Field(..., example={
        "age": 35, "income": 50000, "tenure_months": 24, "product_category": "premium"
    })
    model_version: str
    timestamp: str

# ————— Drift Computations —————————
def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index computation."""
    def _bucket(arr):
        return np.histogram(arr, bins=buckets)[0] / len(arr)
    exp_pct = _bucket(expected)
    act_pct = _bucket(actual)
    # avoid zeros
    exp_pct = np.where(exp_pct==0, 1e-8, exp_pct)
    act_pct = np.where(act_pct==0, 1e-8, act_pct)
    return np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct))

def chi2_score(expected: Dict[str,int], actual: Dict[str,int]) -> float:
    """Chi-square drift for categorical."""
    # align categories
    cats = set(expected) | set(actual)
    exp = np.array([expected.get(c,0) for c in cats])
    act = np.array([actual.get(c,0) for c in cats])
    chi2, p, _, _ = chi2_contingency([exp, act])
    return chi2

# ————— FastAPI App —————————
app = FastAPI(title="Drift Detector")

@app.post("/monitor/predict")
async def detect_drift(payload: FeaturePayload, request: Request):
    logger.info(f"Received payload: {payload.json()}")
    scores = {}
    # numeric drift
    for feat in numerical_features:
        baseline_arr = np.array(baseline["numerical"][feat])
        val = payload.features.get(feat)
        if val is None:
            continue
        # treat single value as distribution
        scores[feat] = psi(baseline_arr, np.array([val]))
        DRIFT_SCORE.labels(feature=feat).set(scores[feat])
    # categorical drift
    for feat in categorical_features:
        base_counts = baseline["categorical"][feat]
        obs = payload.features.get(feat)
        if obs is None:
            continue
        actual_counts = {obs: 1}
        scores[feat] = chi2_score(base_counts, actual_counts)
        DRIFT_SCORE.labels(feature=feat).set(scores[feat])
    # determine status
    max_score = max(scores.values()) if scores else 0.0
    status = "ok"
    if max_score > CRITICAL_THRESHOLD:
        status = "critical"
        DRIFT_ALERT.labels(severity="critical").inc()
    elif max_score > WARNING_THRESHOLD:
        status = "warning"
        DRIFT_ALERT.labels(severity="warning").inc()
    REQUEST_COUNT.labels(status=status).inc()
    return {"drift_scores": scores, "status": status}

@app.get("/monitor/health")
async def health():
    return {"status": "healthy"}

@app.get("/monitor/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
