"""
Tests for src/metrics.py — compute_and_report

Covers:
1. Aviation constraint validation — priority_on_time groups present,
   violation_count and cascade_count correct
2. KPI correctness — all 7 keys present, on_time_rate / assigned_rate / avg_delay_min accurate
3. Determinism — two identical calls return identical kpi_dict
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.metrics import compute_and_report


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE = datetime(2026, 4, 13, 6, 0, 0)
EARLIEST = BASE + timedelta(minutes=15)
STD = BASE + timedelta(minutes=90)

CONFIG = {
    "metrics": {
        "charts_dir": "charts_test_metrics",
        "reports_dir": "reports_test_metrics",
        "save_routes_gantt": True,
        "save_load_chart": True,
        "save_results_csv": True,
    },
    "data_generator": {"time_window_hours": 8},
}


def _make_tasks_df():
    return pd.DataFrame([
        {
            "task_id": "T1", "priority_group": 1,
            "earliest_start": EARLIEST, "STD": STD,
            "vehicle_type_req": "deicing_truck",
        },
        {
            "task_id": "T2", "priority_group": 2,
            "earliest_start": EARLIEST, "STD": STD,
            "vehicle_type_req": "fuel_truck",
        },
        {
            "task_id": "T3", "priority_group": 3,
            "earliest_start": EARLIEST, "STD": STD,
            "vehicle_type_req": "catering_truck",
        },
    ])


def _make_executed_routes():
    return [
        {
            "task_id": "T1", "vehicle_id": "V01",
            "planned_start": EARLIEST,
            "actual_start": EARLIEST,
            "actual_end": EARLIEST + timedelta(minutes=20),
            "delay_min": 0.0,
            "status": "on_time",
        },
        {
            "task_id": "T2", "vehicle_id": "V02",
            "planned_start": EARLIEST,
            "actual_start": EARLIEST + timedelta(minutes=10),
            "actual_end": EARLIEST + timedelta(minutes=30),
            "delay_min": 10.0,
            "status": "delayed",
        },
        {
            "task_id": "T3", "vehicle_id": "V03",
            "planned_start": EARLIEST,
            "actual_start": EARLIEST + timedelta(minutes=80),
            "actual_end": EARLIEST + timedelta(minutes=100),
            "delay_min": 80.0,
            "status": "missed_window",
        },
    ]


def _make_sim_violations():
    return [{"task_id": "T3", "reason": "missed_window"}]


def _make_sim_stats():
    return {
        "total_tasks": 3,
        "on_time": 1,
        "delayed": 1,
        "missed_window": 1,
        "overrun": 0,
        "violation_count": 1,
        "cascade_count": 2,
    }


# ---------------------------------------------------------------------------
# Test 1: Aviation constraint validation
# Priority groups (deicing=1, fueling=2, catering=3) must appear in
# priority_on_time; violation_count and cascade_count must be exact.
# ---------------------------------------------------------------------------

class TestAviationConstraints:
    def test_priority_groups_present(self, tmp_path):
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        # All 3 priority groups (deicing=1, fueling=2, catering=3) must be present
        po = kpi["priority_on_time"]
        assert 1 in po, "priority_group 1 (deicing) missing from priority_on_time"
        assert 2 in po, "priority_group 2 (fueling) missing from priority_on_time"
        assert 3 in po, "priority_group 3 (catering) missing from priority_on_time"

    def test_deicing_on_time_rate(self, tmp_path):
        """Deicing task (T1) was on_time — its group rate should be 1.0."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert kpi["priority_on_time"][1] == pytest.approx(1.0), "deicing on_time rate should be 1.0"
        assert kpi["priority_on_time"][3] == pytest.approx(0.0), "catering on_time rate should be 0.0"

    def test_violation_and_cascade_count(self, tmp_path):
        """violation_count == len(sim_violations); cascade_count from sim_stats."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert kpi["violation_count"] == 1
        assert kpi["cascade_count"] == 2


# ---------------------------------------------------------------------------
# Test 2: KPI correctness
# All 7 keys present; specific formulas produce expected values.
# ---------------------------------------------------------------------------

class TestKPICorrectness:
    REQUIRED_KEYS = {
        "assigned_rate", "on_time_rate", "avg_delay_min",
        "violation_count", "cascade_count",
        "vehicle_utilization", "priority_on_time",
    }

    def test_all_seven_kpi_keys(self, tmp_path):
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        missing = self.REQUIRED_KEYS - set(kpi.keys())
        assert not missing, f"Missing KPI keys: {missing}"

    def test_assigned_rate(self, tmp_path):
        """3 executed / 3 total tasks → assigned_rate == 1.0."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert kpi["assigned_rate"] == pytest.approx(1.0)

    def test_on_time_rate(self, tmp_path):
        """1 on_time out of 3 executed → on_time_rate == 1/3."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert kpi["on_time_rate"] == pytest.approx(1 / 3, rel=1e-3)

    def test_assigned_rate_partial(self, tmp_path):
        """Only 2 tasks executed out of 3 → assigned_rate == 2/3."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        routes_2 = _make_executed_routes()[:2]  # drop T3
        kpi = compute_and_report(
            routes_2,
            [],
            {"total_tasks": 2, "on_time": 1, "delayed": 1, "missed_window": 0, "overrun": 0, "violation_count": 0},
            _make_tasks_df(),
            cfg,
        )

        assert kpi["assigned_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_vehicle_utilization_dict(self, tmp_path):
        """vehicle_utilization is a dict keyed by vehicle_id with float values."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        vu = kpi["vehicle_utilization"]
        assert isinstance(vu, dict)
        for v, pct in vu.items():
            assert isinstance(v, str)
            assert 0.0 <= pct <= 100.0, f"Utilization for {v} out of range: {pct}"

    def test_output_files_created(self, tmp_path):
        """All three output files should be created when flags are True."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert (tmp_path / "routes_gantt.png").exists()
        assert (tmp_path / "load_chart.png").exists()
        assert (tmp_path / "results.csv").exists()

    def test_save_flags_respected(self, tmp_path):
        """When save flags are False, files should NOT be created."""
        cfg = {
            "metrics": {
                "charts_dir": str(tmp_path),
                "reports_dir": str(tmp_path),
                "save_routes_gantt": False,
                "save_load_chart": False,
                "save_results_csv": False,
            },
            "data_generator": {"time_window_hours": 8},
        }

        compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        assert not (tmp_path / "routes_gantt.png").exists()
        assert not (tmp_path / "load_chart.png").exists()
        assert not (tmp_path / "results.csv").exists()


# ---------------------------------------------------------------------------
# Test 3: Determinism — two identical calls produce identical kpi_dict
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_results(self, tmp_path):
        """Running compute_and_report twice with the same inputs gives identical KPI."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        kpi1 = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )
        kpi2 = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            _make_tasks_df(),
            cfg,
        )

        # Scalar fields must be bit-identical
        scalar_keys = ["assigned_rate", "on_time_rate", "avg_delay_min",
                       "violation_count", "cascade_count"]
        for key in scalar_keys:
            assert kpi1[key] == kpi2[key], f"Mismatch on key '{key}': {kpi1[key]} vs {kpi2[key]}"

        # Dict fields must match
        assert kpi1["vehicle_utilization"] == kpi2["vehicle_utilization"]
        assert kpi1["priority_on_time"] == kpi2["priority_on_time"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_raises_on_empty_executed_routes(self, tmp_path):
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        with pytest.raises(ValueError, match="empty"):
            compute_and_report([], [], {}, _make_tasks_df(), cfg)

    def test_no_priority_group_column(self, tmp_path):
        """tasks_df without priority_group should not crash — priority_on_time is empty."""
        cfg = dict(CONFIG)
        cfg["metrics"] = dict(CONFIG["metrics"], charts_dir=str(tmp_path), reports_dir=str(tmp_path))

        tasks_minimal = pd.DataFrame({"task_id": ["T1", "T2", "T3"]})
        kpi = compute_and_report(
            _make_executed_routes(),
            _make_sim_violations(),
            _make_sim_stats(),
            tasks_minimal,
            cfg,
        )

        assert isinstance(kpi["priority_on_time"], dict)
