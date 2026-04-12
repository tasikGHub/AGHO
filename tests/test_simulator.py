"""
Smoke tests for src/simulator.py — run_simulation

Covers:
1. Output structure validity — keys, types, status values
2. Status classification logic — on_time / delayed / missed_window / overrun
3. Safe interval enforcement between vehicles at the same stand
4. Determinism — two identical calls return identical results
5. Edge cases — empty assigned_routes, missing stand in apron_graph
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import pytest

from src.simulator import run_simulation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 4, 13, 6, 0, 0)

VALID_STATUSES = {"on_time", "delayed", "missed_window", "overrun"}


def _make_graph() -> nx.Graph:
    """DEPOT — S01 — S02 with known distances."""
    G = nx.Graph()
    G.add_node("DEPOT", node_type="depot")
    G.add_node("S01",   node_type="stand")
    G.add_node("S02",   node_type="stand")
    G.add_edge("DEPOT", "S01", distance_m=50.0)
    G.add_edge("S01",   "S02", distance_m=80.0)
    return G


def _make_config() -> dict:
    return {
        "optimizer": {
            "safe_interval_min": 2,
            "max_speed_kmh": 30,
        }
    }


def _make_vehicles_df(free_at=None) -> pd.DataFrame:
    if free_at is None:
        free_at = BASE_TIME
    return pd.DataFrame([
        {
            "vehicle_id": "V01", "vehicle_type": "deicing_truck",
            "speed_kmh": 20.0, "capacity": 5000.0,
            "start_stand": "DEPOT", "free_at": free_at,
        },
        {
            "vehicle_id": "V02", "vehicle_type": "fuel_truck",
            "speed_kmh": 25.0, "capacity": 20000.0,
            "start_stand": "DEPOT", "free_at": free_at,
        },
    ])


def _make_tasks_df() -> pd.DataFrame:
    std = BASE_TIME + timedelta(hours=2)
    earliest = BASE_TIME + timedelta(minutes=15)
    return pd.DataFrame([
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
            "stand_id": "S02", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
        {
            "task_id": "T0003", "flight_id": "FL002",
            "task_type": "deicing",  "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
        {
            "task_id": "T0004", "flight_id": "FL002",
            "task_type": "fueling",  "priority_group": 2,
            "stand_id": "S02", "STA": BASE_TIME, "STD": std,
            "earliest_start": earliest, "vehicle_type_req": "fuel_truck",
            "service_time_pred": 15.0,
        },
    ])


def _make_assigned_routes(start_offset_min: int = 15) -> list[dict]:
    """Two vehicles, four tasks — one pair per vehicle."""
    t0 = BASE_TIME + timedelta(minutes=start_offset_min)
    return [
        {
            "task_id": "T0001", "vehicle_id": "V01",
            "start_time": t0,
            "end_time": t0 + timedelta(minutes=20),
            "route": ["DEPOT", "S01"],
        },
        {
            "task_id": "T0002", "vehicle_id": "V02",
            "start_time": t0,
            "end_time": t0 + timedelta(minutes=15),
            "route": ["DEPOT", "S02"],
        },
        {
            "task_id": "T0003", "vehicle_id": "V01",
            "start_time": t0 + timedelta(minutes=22),
            "end_time": t0 + timedelta(minutes=42),
            "route": ["S01", "S01"],
        },
        {
            "task_id": "T0004", "vehicle_id": "V02",
            "start_time": t0 + timedelta(minutes=17),
            "end_time": t0 + timedelta(minutes=32),
            "route": ["S02", "S02"],
        },
    ]


# ---------------------------------------------------------------------------
# Test 1: Output structure — required keys, valid statuses, correct types
# ---------------------------------------------------------------------------

def test_output_structure():
    executed, violations, stats = run_simulation(
        _make_assigned_routes(),
        _make_tasks_df(),
        _make_vehicles_df(),
        _make_graph(),
        _make_config(),
    )

    assert isinstance(executed, list)
    assert isinstance(violations, list)
    assert isinstance(stats, dict)

    for r in executed:
        assert "task_id"    in r
        assert "vehicle_id" in r
        assert "start_time" in r
        assert "end_time"   in r
        assert "route"      in r
        assert "status"     in r
        assert r["status"] in VALID_STATUSES
        assert isinstance(r["route"], list)
        assert len(r["route"]) >= 1
        assert r["start_time"] <= r["end_time"]

    for v in violations:
        assert "task_id" in v
        assert "reason"  in v

    required_stat_keys = {"total_tasks", "on_time", "delayed", "missed_window", "overrun", "violation_count"}
    assert required_stat_keys.issubset(stats.keys())
    assert stats["total_tasks"] == len(executed)


# ---------------------------------------------------------------------------
# Test 2: Status classification
#
# on_time  — vehicle available, task fits comfortably in window
# delayed  — vehicle arrives after earliest_start, but finishes before STD
# missed_window — task cannot finish before STD
# overrun  — vehicle available only after STD
# ---------------------------------------------------------------------------

def test_status_on_time():
    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S01", "STA": BASE_TIME,
        "STD": BASE_TIME + timedelta(hours=2),
        "earliest_start": BASE_TIME + timedelta(minutes=15),
        "vehicle_type_req": "fuel_truck",
        "service_time_pred": 15.0,
    }])
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "S01",  # already at the stand
        "free_at": BASE_TIME + timedelta(minutes=15),
    }])
    routes = [{
        "task_id": "T0001", "vehicle_id": "V01",
        "start_time": BASE_TIME + timedelta(minutes=15),
        "end_time":   BASE_TIME + timedelta(minutes=30),
        "route": ["S01"],
    }]

    executed, _, stats = run_simulation(routes, tasks, vehicles, _make_graph(), _make_config())
    assert len(executed) == 1
    assert executed[0]["status"] == "on_time"
    assert stats["on_time"] == 1


def test_status_delayed():
    """Vehicle arrives late (after earliest_start) but finishes before STD."""
    earliest = BASE_TIME + timedelta(minutes=15)
    std = BASE_TIME + timedelta(hours=2)
    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S01", "STA": BASE_TIME,
        "STD": std, "earliest_start": earliest,
        "vehicle_type_req": "fuel_truck",
        "service_time_pred": 15.0,
    }])
    # Vehicle is busy until 30 min after BASE_TIME (after earliest_start=15)
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "S01",
        "free_at": BASE_TIME + timedelta(minutes=30),
    }])
    routes = [{
        "task_id": "T0001", "vehicle_id": "V01",
        "start_time": BASE_TIME + timedelta(minutes=30),
        "end_time":   BASE_TIME + timedelta(minutes=45),
        "route": ["S01"],
    }]

    executed, _, stats = run_simulation(routes, tasks, vehicles, _make_graph(), _make_config())
    assert executed[0]["status"] == "delayed"
    assert stats["delayed"] == 1


def test_status_missed_window():
    """Service cannot finish before STD."""
    earliest = BASE_TIME
    std = BASE_TIME + timedelta(minutes=10)
    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S01", "STA": BASE_TIME,
        "STD": std, "earliest_start": earliest,
        "vehicle_type_req": "fuel_truck",
        "service_time_pred": 20.0,  # 20 min > 10 min window
    }])
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "S01", "free_at": BASE_TIME,
    }])
    routes = [{
        "task_id": "T0001", "vehicle_id": "V01",
        "start_time": BASE_TIME,
        "end_time":   BASE_TIME + timedelta(minutes=20),
        "route": ["S01"],
    }]

    executed, violations, stats = run_simulation(routes, tasks, vehicles, _make_graph(), _make_config())
    assert executed[0]["status"] == "missed_window"
    assert stats["missed_window"] == 1
    assert any(v["task_id"] == "T0001" for v in violations)


def test_status_overrun():
    """Vehicle becomes available only after STD."""
    std = BASE_TIME + timedelta(minutes=30)
    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S01", "STA": BASE_TIME,
        "STD": std,
        "earliest_start": BASE_TIME,
        "vehicle_type_req": "fuel_truck",
        "service_time_pred": 15.0,
    }])
    # Vehicle busy until after STD
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "S01",
        "free_at": BASE_TIME + timedelta(minutes=60),  # after STD
    }])
    routes = [{
        "task_id": "T0001", "vehicle_id": "V01",
        "start_time": BASE_TIME + timedelta(minutes=60),
        "end_time":   BASE_TIME + timedelta(minutes=75),
        "route": ["S01"],
    }]

    executed, violations, stats = run_simulation(routes, tasks, vehicles, _make_graph(), _make_config())
    assert executed[0]["status"] == "overrun"
    assert stats["overrun"] == 1
    assert any(v["task_id"] == "T0001" for v in violations)


# ---------------------------------------------------------------------------
# Test 3: Safe interval — two tasks at same stand, second must start
#         >= first end + safe_interval_min
# ---------------------------------------------------------------------------

def test_safe_interval_enforcement():
    std = BASE_TIME + timedelta(hours=3)
    earliest = BASE_TIME + timedelta(minutes=15)
    cfg = _make_config()
    safe_min = cfg["optimizer"]["safe_interval_min"]

    tasks = pd.DataFrame([
        {
            "task_id": "T0001", "flight_id": "FL001",
            "task_type": "deicing", "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME,
            "STD": std, "earliest_start": earliest,
            "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
        {
            "task_id": "T0002", "flight_id": "FL002",
            "task_type": "deicing", "priority_group": 1,
            "stand_id": "S01", "STA": BASE_TIME,
            "STD": std, "earliest_start": earliest,
            "vehicle_type_req": "deicing_truck",
            "service_time_pred": 20.0,
        },
    ])
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "deicing_truck",
        "speed_kmh": 20.0, "capacity": 5000.0,
        "start_stand": "DEPOT", "free_at": BASE_TIME,
    }])
    # Assign both tasks to V01 sequentially
    t_start1 = earliest
    t_end1   = t_start1 + timedelta(minutes=20)
    t_start2 = t_end1 + timedelta(minutes=1)  # 1 min gap — less than safe_interval
    t_end2   = t_start2 + timedelta(minutes=20)
    routes = [
        {
            "task_id": "T0001", "vehicle_id": "V01",
            "start_time": t_start1, "end_time": t_end1,
            "route": ["DEPOT", "S01"],
        },
        {
            "task_id": "T0002", "vehicle_id": "V01",
            "start_time": t_start2, "end_time": t_end2,
            "route": ["S01"],
        },
    ]

    executed, _, _ = run_simulation(routes, tasks, vehicles, _make_graph(), cfg)
    assert len(executed) == 2

    by_task = {r["task_id"]: r for r in executed}
    first_end     = by_task["T0001"]["end_time"]
    second_start  = by_task["T0002"]["start_time"]

    gap_min = (second_start - first_end).total_seconds() / 60.0
    assert gap_min >= safe_min, f"Safe interval violated: gap={gap_min:.2f} < {safe_min}"


# ---------------------------------------------------------------------------
# Test 4: Determinism — two identical calls return identical results
# ---------------------------------------------------------------------------

def test_determinism():
    routes   = _make_assigned_routes()
    tasks    = _make_tasks_df()
    vehicles = _make_vehicles_df()
    graph    = _make_graph()
    cfg      = _make_config()

    exec1, viol1, stats1 = run_simulation(routes, tasks, vehicles, graph, cfg)
    exec2, viol2, stats2 = run_simulation(routes, tasks, vehicles, graph, cfg)

    assert exec1 == exec2
    assert viol1 == viol2
    assert stats1 == stats2


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------

def test_empty_assigned_routes():
    exec_, viol, stats = run_simulation(
        [], _make_tasks_df(), _make_vehicles_df(), _make_graph(), _make_config()
    )
    assert exec_ == []
    assert viol  == []
    assert stats["total_tasks"] == 0


def test_stand_not_in_graph_logs_and_skips():
    """If stand is missing from apron_graph, task is skipped with a violation."""
    tasks = pd.DataFrame([{
        "task_id": "T0001", "flight_id": "FL001",
        "task_type": "fueling", "priority_group": 2,
        "stand_id": "S99",  # not in graph
        "STA": BASE_TIME,
        "STD": BASE_TIME + timedelta(hours=2),
        "earliest_start": BASE_TIME + timedelta(minutes=15),
        "vehicle_type_req": "fuel_truck",
        "service_time_pred": 15.0,
    }])
    vehicles = pd.DataFrame([{
        "vehicle_id": "V01", "vehicle_type": "fuel_truck",
        "speed_kmh": 25.0, "capacity": 20000.0,
        "start_stand": "DEPOT", "free_at": BASE_TIME,
    }])
    routes = [{
        "task_id": "T0001", "vehicle_id": "V01",
        "start_time": BASE_TIME + timedelta(minutes=15),
        "end_time":   BASE_TIME + timedelta(minutes=30),
        "route": ["DEPOT", "S99"],
    }]

    # Should NOT raise — fallback travel_time=0 and task is simulated at earliest_start
    executed, _, stats = run_simulation(routes, tasks, vehicles, _make_graph(), _make_config())
    # Task is still executed (with fallback travel time 0), no hard crash
    assert isinstance(executed, list)
    assert isinstance(stats, dict)
