"""
Tests for src/optimizer.py — Optimizer
Covers: output format, aviation priority/urgency ordering, time window violations,
        safe interval enforcement, no-vehicle violations, determinism, edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import pytest

from src.optimizer import Optimizer, run_optimizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 4, 13, 6, 0, 0)


def _make_graph() -> nx.Graph:
    """Minimal connected apron graph: DEPOT — S01 — S02."""
    G = nx.Graph()
    G.add_node("DEPOT", node_type="depot")
    G.add_node("S01", node_type="stand")
    G.add_node("S02", node_type="stand")
    G.add_edge("DEPOT", "S01", distance_m=50.0)
    G.add_edge("S01", "S02", distance_m=80.0)
    return G


def _make_config() -> dict:
    return {
        "optimizer": {
            "priority_groups": {"deicing": 1, "fueling": 2, "catering": 3},
            "safe_interval_min": 2,
            "max_speed_kmh": 30,
        }
    }


def _make_vehicles_df() -> pd.DataFrame:
    """One vehicle of each type, all starting at DEPOT."""
    return pd.DataFrame([
        {
            "vehicle_id": "V01", "vehicle_type": "deicing_truck",
            "speed_kmh": 20.0, "capacity": 5000.0,
            "start_stand": "DEPOT", "free_at": BASE_TIME,
        },
        {
            "vehicle_id": "V02", "vehicle_type": "fuel_truck",
            "speed_kmh": 25.0, "capacity": 20000.0,
            "start_stand": "DEPOT", "free_at": BASE_TIME,
        },
        {
            "vehicle_id": "V03", "vehicle_type": "catering_truck",
            "speed_kmh": 30.0, "capacity": 100.0,
            "start_stand": "DEPOT", "free_at": BASE_TIME,
        },
    ])


def _make_tasks_df() -> pd.DataFrame:
    """One task of each type for FL001 at S01. All comfortably fit in 2-hour window."""
    std = BASE_TIME + timedelta(hours=2)
    earliest = BASE_TIME + timedelta(minutes=15)
    rows = [
        {
            "task_id": "T0001", "flight_id": "FL001",
            "task_type": "deicing",  "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
        {
            "task_id": "T0002", "flight_id": "FL001",
            "task_type": "fueling",  "priority_group": 2,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
        {
            "task_id": "T0003", "flight_id": "FL001",
            "task_type": "catering", "priority_group": 3,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "catering_truck",
            "service_time_pred": 10.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Output format — all required keys present, all tasks assigned
# ---------------------------------------------------------------------------

def test_output_format():
    routes, violations = run_optimizer(
        _make_tasks_df(), _make_vehicles_df(), _make_graph(), _make_config()
    )

    assert isinstance(routes, list)
    assert isinstance(violations, list)

    for r in routes:
        assert "task_id"    in r
        assert "vehicle_id" in r
        assert "start_time" in r
        assert "end_time"   in r
        assert "route"      in r
        assert isinstance(r["route"], list)
        assert len(r["route"]) >= 1
        assert r["start_time"] < r["end_time"]

    for v in violations:
        assert "task_id" in v
        assert "reason"  in v

    # All 3 tasks assigned (one vehicle per type, window is 2 h)
    assigned_ids = {r["task_id"] for r in routes}
    assert assigned_ids == {"T0001", "T0002", "T0003"}
    assert violations == []


# ---------------------------------------------------------------------------
# Test 2: Aviation priority — urgency ordering within the same vehicle type
#
# Two fueling tasks share one fuel_truck.
# Task A: STD far away → high urgency score → sorted second
# Task B: STD close    → low urgency score  → sorted first
# Expected: Task B starts before Task A.
# ---------------------------------------------------------------------------

def test_urgency_ordering():
    std_far   = BASE_TIME + timedelta(hours=3)   # urgency = 180 - 15 - 15 = 150
    std_close = BASE_TIME + timedelta(minutes=60) # urgency =  60 - 15 - 15 = 30
    earliest  = BASE_TIME + timedelta(minutes=15)

    tasks = pd.DataFrame([
        {
            "task_id": "T0001", "flight_id": "FL001",
            "task_type": "fueling", "priority_group": 2,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std_far,
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
        {
            "task_id": "T0002", "flight_id": "FL002",
            "task_type": "fueling", "priority_group": 2,
            "stand_id": "S02", "STA": BASE_TIME, "STD": std_close,
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
    ])

    # Only one fuel_truck so tasks are serialised
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "DEPOT", "free_at": BASE_TIME,
    }])

    routes, _ = run_optimizer(tasks, vehicles, _make_graph(), _make_config())
    assert len(routes) == 2

    by_task = {r["task_id"]: r for r in routes}
    # T0002 (tight STD) must start before T0001 (relaxed STD)
    assert by_task["T0002"]["start_time"] <= by_task["T0001"]["start_time"]


# ---------------------------------------------------------------------------
# Test 3: Priority group takes precedence over urgency
#
# T0001: priority_group=1 (deicing), urgency=100 (relaxed)
# T0002: priority_group=2 (fueling), urgency=10  (tight)
#
# deicing_truck handles T0001; fuel_truck handles T0002 independently.
# Both can be assigned. Verify both appear in assigned_routes.
# (priority_group=1 task must not be skipped in favour of priority_group=2)
# ---------------------------------------------------------------------------

def test_priority_group_respected():
    std = BASE_TIME + timedelta(hours=2)
    earliest = BASE_TIME + timedelta(minutes=15)

    tasks = pd.DataFrame([
        {
            "task_id": "T0001", "flight_id": "FL001",
            "task_type": "deicing",  "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
        {
            "task_id": "T0002", "flight_id": "FL002",
            "task_type": "fueling",  "priority_group": 2,
            "stand_id": "S01", "STA": BASE_TIME,
            "STD": BASE_TIME + timedelta(minutes=40),  # tighter window
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
    ])

    routes, violations = run_optimizer(tasks, _make_vehicles_df(), _make_graph(), _make_config())

    assigned_ids = {r["task_id"] for r in routes}
    # Both tasks must be assigned (different vehicle types, no conflict)
    assert "T0001" in assigned_ids
    assert "T0002" in assigned_ids


# ---------------------------------------------------------------------------
# Test 4: Time window violation
#
# Task's end_time will exceed STD → appears in violations with
# "time_window_violated" but also remains in assigned_routes.
# ---------------------------------------------------------------------------

def test_time_window_violation():
    std_tight = BASE_TIME + timedelta(minutes=10)   # only 10 min window
    earliest  = BASE_TIME + timedelta(minutes=0)

    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S01", "STA": BASE_TIME, "STD": std_tight,
        "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
        "service_time_pred": 20.0,   # 20 min > 10 min window → violation
    }])

    routes, violations = run_optimizer(tasks, _make_vehicles_df(), _make_graph(), _make_config())

    violation_reasons = {v["task_id"]: v["reason"] for v in violations}
    assert "T0001" in violation_reasons
    assert violation_reasons["T0001"] == "time_window_violated"

    # Task still assigned despite window violation
    assigned_ids = {r["task_id"] for r in routes}
    assert "T0001" in assigned_ids


# ---------------------------------------------------------------------------
# Test 5: No vehicle of required type → violation, task not assigned
# ---------------------------------------------------------------------------

def test_no_vehicle_of_type():
    # Remove all deicing trucks from fleet
    vehicles = _make_vehicles_df()
    vehicles = vehicles[vehicles["vehicle_type"] != "deicing_truck"].reset_index(drop=True)

    tasks = _make_tasks_df()
    routes, violations = run_optimizer(tasks, vehicles, _make_graph(), _make_config())

    violation_reasons = {v["task_id"]: v["reason"] for v in violations}
    assert "T0001" in violation_reasons
    assert violation_reasons["T0001"] == "no_vehicle_of_type"

    # Deicing task must NOT appear in assigned_routes
    assigned_ids = {r["task_id"] for r in routes}
    assert "T0001" not in assigned_ids

    # Other tasks (fueling, catering) still assigned
    assert "T0002" in assigned_ids
    assert "T0003" in assigned_ids


# ---------------------------------------------------------------------------
# Test 6: Safe interval enforcement
#
# Two deicing tasks share one deicing_truck at the same stand (S01).
# Task B's start must be >= Task A's end + safe_interval_min (2 min).
# ---------------------------------------------------------------------------

def test_safe_interval():
    std = BASE_TIME + timedelta(hours=3)
    earliest = BASE_TIME + timedelta(minutes=15)

    tasks = pd.DataFrame([
        {
            "task_id": "T0001", "flight_id": "FL001",
            "task_type": "deicing", "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
        {
            "task_id": "T0002", "flight_id": "FL002",
            "task_type": "deicing", "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
    ])

    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "deicing_truck",
        "speed_kmh": 20.0, "capacity": 5000.0,
        "start_stand": "DEPOT", "free_at": BASE_TIME,
    }])

    cfg = _make_config()
    safe_min = cfg["optimizer"]["safe_interval_min"]

    routes, _ = run_optimizer(tasks, vehicles, _make_graph(), cfg)
    assert len(routes) == 2

    by_task = {r["task_id"]: r for r in routes}
    first_end  = min(r["end_time"]   for r in routes)
    second_start = max(r["start_time"] for r in routes)

    gap = (second_start - first_end).total_seconds() / 60.0
    assert gap >= safe_min, f"Safe interval violated: gap={gap:.2f} min < {safe_min} min"


# ---------------------------------------------------------------------------
# Test 7: Determinism — two identical calls return identical results
# ---------------------------------------------------------------------------

def test_determinism():
    tasks = _make_tasks_df()
    vehicles = _make_vehicles_df()
    graph = _make_graph()
    cfg = _make_config()

    routes1, violations1 = run_optimizer(tasks, vehicles, graph, cfg)
    routes2, violations2 = run_optimizer(tasks, vehicles, graph, cfg)

    assert routes1 == routes2
    assert violations1 == violations2


# ---------------------------------------------------------------------------
# Test 8: Edge cases — empty input and disconnected graph raise correctly
# ---------------------------------------------------------------------------

def test_empty_tasks_raises():
    empty = pd.DataFrame(columns=[
        "task_id", "flight_id", "task_type", "priority_group",
        "stand_id", "STA", "STD", "earliest_start",
        "vehicle_type_req", "service_time_pred",
    ])
    with pytest.raises(ValueError, match="must not be empty"):
        run_optimizer(empty, _make_vehicles_df(), _make_graph(), _make_config())


def test_disconnected_graph_raises():
    G = nx.Graph()
    G.add_node("DEPOT", node_type="depot")
    G.add_node("S01",   node_type="stand")
    G.add_node("S02",   node_type="stand")
    # S02 deliberately not connected → graph is disconnected
    G.add_edge("DEPOT", "S01", distance_m=50.0)

    with pytest.raises(RuntimeError, match="not connected"):
        run_optimizer(_make_tasks_df(), _make_vehicles_df(), G, _make_config())


def test_none_vehicles_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        run_optimizer(_make_tasks_df(), None, _make_graph(), _make_config())


def test_unknown_stand_id_raises():
    tasks = _make_tasks_df().copy()
    tasks.loc[0, "stand_id"] = "S99"   # not in _make_graph()

    with pytest.raises(ValueError, match="stand_id"):
        run_optimizer(tasks, _make_vehicles_df(), _make_graph(), _make_config())
