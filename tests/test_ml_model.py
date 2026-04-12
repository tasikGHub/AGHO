"""
Tests for src/ml_model.py — MLForecast
Covers: determinism, output correctness, fallback behaviour.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.ml_model import MLForecast, run_ml_forecast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tasks_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    task_types = rng.choice(["deicing", "fueling", "catering"], size=n)
    aircraft_types = rng.choice(["narrow", "wide"], size=n)
    stand_ids = [f"S{i % 10 + 1:02d}" for i in range(n)]
    hour = rng.integers(6, 14, size=n).astype(float)
    svc_actual = rng.uniform(10, 40, size=n)

    return pd.DataFrame({
        "task_id": [f"T{i:04d}" for i in range(1, n + 1)],
        "flight_id": [f"FL{i:03d}" for i in range(1, n + 1)],
        "task_type": task_types,
        "aircraft_type": aircraft_types,
        "stand_id": stand_ids,
        "hour_of_day": hour,
        "service_time_actual": svc_actual,
    })


# ---------------------------------------------------------------------------
# Test 1: Determinism — two runs with seed=42 produce identical predictions
# ---------------------------------------------------------------------------

def test_determinism():
    tasks = _make_tasks_df(seed=0)

    result1, mae1 = run_ml_forecast(tasks, seed=42)
    result2, mae2 = run_ml_forecast(tasks, seed=42)

    pd.testing.assert_series_equal(
        result1["service_time_pred"].reset_index(drop=True),
        result2["service_time_pred"].reset_index(drop=True),
        check_names=False,
    )
    assert mae1 == mae2


# ---------------------------------------------------------------------------
# Test 2: Output correctness — shape, dtypes, positivity, MAE >= 0
# ---------------------------------------------------------------------------

def test_output_correctness():
    tasks = _make_tasks_df(n=40, seed=1)

    result, mae = run_ml_forecast(tasks, seed=42)

    # Same number of rows
    assert len(result) == len(tasks)

    # Column present
    assert "service_time_pred" in result.columns

    # All predictions positive
    assert (result["service_time_pred"] > 0).all()

    # MAE is non-negative float
    assert isinstance(mae, float)
    assert mae >= 0.0

    # Original columns preserved
    for col in tasks.columns:
        assert col in result.columns


# ---------------------------------------------------------------------------
# Test 3: Fallback — broken model → predictions = mean, no crash
# ---------------------------------------------------------------------------

def test_fallback_on_training_failure():
    tasks = _make_tasks_df(n=20, seed=2)

    model = MLForecast(seed=42)

    # Monkey-patch: replace model with one that raises on fit
    class _BrokenRF:
        def fit(self, X, y):
            raise RuntimeError("injected training failure")

    model.model = _BrokenRF()

    result, mae = model.fit_predict(tasks)

    expected_mean = tasks["service_time_actual"].mean()

    # All predictions equal fallback mean
    assert np.allclose(result["service_time_pred"].values, expected_mean)

    # MAE is computable and non-negative
    assert isinstance(mae, float)
    assert mae >= 0.0

    # No rows dropped
    assert len(result) == len(tasks)


# ---------------------------------------------------------------------------
# Test 4: ValueError on empty input
# ---------------------------------------------------------------------------

def test_empty_tasks_raises():
    empty = pd.DataFrame(columns=[
        "task_id", "flight_id", "task_type", "aircraft_type",
        "stand_id", "hour_of_day", "service_time_actual",
    ])

    with pytest.raises(ValueError, match="must not be empty"):
        run_ml_forecast(empty, seed=42)
