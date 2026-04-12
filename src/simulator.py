"""
Simulator — Airport Ground Handling Optimizer
Discrete-event execution simulation of assigned vehicle routes.

Re-plays the optimizer plan step by step, recalculates actual timings,
enforces safe intervals, and classifies each task as:
  on_time | delayed | missed_window | overrun
"""

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd


def _log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [Simulator]     {level} — {message}")


# ---------------------------------------------------------------------------
# Travel time helper
# ---------------------------------------------------------------------------

def _calc_travel_time_min(
    apron_graph: nx.Graph,
    source: str,
    target: str,
    speed_kmh: float,
    max_speed_kmh: float,
    task_id: str,
) -> tuple[float, list]:
    """
    Return (travel_time_min, path).

    Falls back to (0.0, [target]) and logs a warning if source or target
    is not in the graph, or no path exists.
    """
    if source == target:
        return 0.0, [target]

    nodes = set(apron_graph.nodes)
    if source not in nodes:
        _log("WARN", f"task {task_id}: source node '{source}' not in apron_graph — skipping travel")
        return 0.0, [target]
    if target not in nodes:
        _log("WARN", f"task {task_id}: target node '{target}' not in apron_graph — skipping travel")
        return 0.0, [target]

    try:
        dist_m, path = nx.single_source_dijkstra(
            apron_graph, source=source, target=target, weight="distance_m"
        )
    except nx.NetworkXNoPath:
        _log("WARN", f"task {task_id}: no path '{source}' → '{target}' in apron_graph — skipping travel")
        return 0.0, [target]

    effective_speed = min(speed_kmh, max_speed_kmh)
    travel_time_min = dist_m / (effective_speed * 1000.0 / 60.0)
    return travel_time_min, path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    assigned_routes: list[dict],
    tasks_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    apron_graph: nx.Graph,
    config: dict,
) -> tuple[list[dict], list[dict], dict]:
    """
    Simulate execution of the optimizer's plan and classify each task.

    Parameters
    ----------
    assigned_routes : list[dict]
        Output of optimizer: [{task_id, vehicle_id, start_time, end_time, route}, ...]
    tasks_df : pd.DataFrame
        Task schedule with columns: task_id, earliest_start, STD, service_time_pred.
    vehicles_df : pd.DataFrame
        Vehicle fleet with columns: vehicle_id, speed_kmh, start_stand, free_at.
    apron_graph : nx.Graph
        Apron geometry (nodes with distance_m edge weights).
    config : dict
        Full pipeline config loaded from scenario YAML.

    Returns
    -------
    executed_routes : list[dict]
        One entry per assigned task:
        {task_id, vehicle_id, start_time, end_time, route, status}
        status ∈ {on_time, delayed, missed_window, overrun}
    sim_violations : list[dict]
        Constraint violations found during simulation:
        {task_id, reason}
    sim_stats : dict
        Aggregated KPIs:
        {total_tasks, on_time, delayed, missed_window, overrun, violation_count}
    """
    if not assigned_routes:
        raise ValueError("assigned_routes is empty — nothing to simulate")

    opt_cfg = config["optimizer"]
    safe_interval_min: float = float(opt_cfg["safe_interval_min"])
    max_speed_kmh: float = float(opt_cfg["max_speed_kmh"])

    # Index tasks by task_id for O(1) lookups
    tasks_index: dict[str, dict] = {
        row["task_id"]: row.to_dict()
        for _, row in tasks_df.iterrows()
    }

    # Initialise vehicle simulation state from vehicles_df
    v_state: dict[str, dict] = {
        row["vehicle_id"]: {
            "speed_kmh": float(row["speed_kmh"]),
            "current_pos": row["start_stand"],
            "free_at": row["free_at"],
        }
        for _, row in vehicles_df.iterrows()
    }

    # Stand safe-interval tracker: stand_id → last end_time at that stand
    stand_last_end: dict[str, datetime] = {}

    # Process routes in ascending start_time order (deterministic)
    sorted_routes = sorted(assigned_routes, key=lambda r: r["start_time"])

    executed_routes: list[dict] = []
    sim_violations: list[dict] = []
    cascade_count: int = 0

    for entry in sorted_routes:
        task_id: str = entry["task_id"]
        vehicle_id: str = entry["vehicle_id"]
        planned_start: datetime = entry["start_time"]   # optimizer's planned start
        planned_route: list = entry.get("route", [])

        # --- Lookup task info ---
        task = tasks_index.get(task_id)
        if task is None:
            _log("WARN", f"task {task_id}: not found in tasks_df — skipping")
            sim_violations.append({"task_id": task_id, "reason": "task_not_in_tasks_df"})
            continue

        earliest_start: datetime = task["earliest_start"]
        std: datetime = task["STD"]
        svc_time: float = float(task.get("service_time_pred", 0.0))

        # stand_id: prefer from tasks_df; fallback to last node in planned route
        stand_id: str = task.get("stand_id") or (planned_route[-1] if planned_route else "")
        if not stand_id:
            _log("WARN", f"task {task_id}: stand_id unknown — skipping")
            sim_violations.append({"task_id": task_id, "reason": "stand_id_unknown"})
            continue

        # --- Lookup vehicle state ---
        vs = v_state.get(vehicle_id)
        if vs is None:
            _log("WARN", f"task {task_id}: vehicle '{vehicle_id}' not in vehicles_df — skipping")
            sim_violations.append({"task_id": task_id, "reason": "vehicle_not_in_vehicles_df"})
            continue

        # --- Step 1: travel time ---
        travel_time_min, path = _calc_travel_time_min(
            apron_graph,
            source=vs["current_pos"],
            target=stand_id,
            speed_kmh=vs["speed_kmh"],
            max_speed_kmh=max_speed_kmh,
            task_id=task_id,
        )

        # --- Step 2: actual start time ---
        # Vehicle must finish previous task and travel to this stand
        raw_start: datetime = vs["free_at"] + timedelta(minutes=travel_time_min)
        # Aircraft must be parked before servicing starts
        actual_start: datetime = max(raw_start, earliest_start)

        # --- Step 3: safe interval between operations at the same stand ---
        last_end = stand_last_end.get(stand_id)
        if last_end is not None:
            earliest_allowed = last_end + timedelta(minutes=safe_interval_min)
            if earliest_allowed > actual_start:
                actual_start = earliest_allowed
                cascade_count += 1

        actual_end: datetime = actual_start + timedelta(minutes=svc_time)
        delay_min: float = max(0.0, (actual_start - planned_start).total_seconds() / 60.0)

        # --- Status classification ---
        if actual_start >= std:
            # Task started at or after STD — window completely missed
            status = "missed_window"
            sim_violations.append({"task_id": task_id, "reason": "missed_window"})
        elif actual_end > std:
            # Started in time but overruns STD
            status = "overrun"
            sim_violations.append({"task_id": task_id, "reason": "overrun"})
        elif actual_start > earliest_start:
            # Late arrival but task finishes before STD
            status = "delayed"
        else:
            status = "on_time"

        executed_routes.append({
            "task_id": task_id,
            "vehicle_id": vehicle_id,
            "planned_start": planned_start,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "delay_min": delay_min,
            "route": path,
            "status": status,
        })

        # --- Step 4: update vehicle state ---
        vs["free_at"] = actual_end
        vs["current_pos"] = stand_id

        # Update stand safe-interval tracker
        prev_end = stand_last_end.get(stand_id)
        if prev_end is None or actual_end > prev_end:
            stand_last_end[stand_id] = actual_end

    # --- Aggregate stats ---
    status_counts: dict[str, int] = {"on_time": 0, "delayed": 0, "missed_window": 0, "overrun": 0}
    for r in executed_routes:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    sim_stats: dict = {
        "total_tasks": len(executed_routes),
        "on_time": status_counts["on_time"],
        "delayed": status_counts["delayed"],
        "missed_window": status_counts["missed_window"],
        "overrun": status_counts["overrun"],
        "violation_count": len(sim_violations),
        "cascade_count": cascade_count,
    }

    _log(
        "OK",
        f"{status_counts['on_time']}/{len(executed_routes)} tasks on_time, "
        f"{len(sim_violations)} violations",
    )

    return executed_routes, sim_violations, sim_stats
