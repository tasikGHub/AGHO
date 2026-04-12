"""
Metrics — Airport Ground Handling Optimizer
Aggregates simulation results into KPI metrics, generates visualisations,
and prints a formatted summary to stdout.

Only aggregates data from simulator output — no business logic.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "on_time":       "#2ca02c",   # green
    "delayed":       "#ffcc00",   # yellow
    "missed_window": "#d62728",   # red
    "overrun":       "#8B0000",   # dark red
}

_VTYPE_COLORS = {
    "deicing_truck":  "#1f77b4",  # blue
    "fuel_truck":     "#ff7f0e",  # orange
    "catering_truck": "#9467bd",  # purple
}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [Metrics]        {level} — {message}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_routes_df(
    executed_routes: list[dict],
    tasks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert executed_routes to a DataFrame and enrich it with task metadata.

    Normalises field names so both simulator variants are supported:
      • start_time / end_time   (actual simulator output)
      • actual_start / actual_end  (design-doc names)
    """
    df = pd.DataFrame(executed_routes)
    if df.empty:
        return df

    # Merge with tasks_df to get priority_group, earliest_start, vehicle_type_req
    tasks_cols = [c for c in ["task_id", "priority_group", "earliest_start", "STD", "vehicle_type_req"]
                  if c in tasks_df.columns]
    if tasks_cols:
        df = df.merge(tasks_df[tasks_cols], on="task_id", how="left")

    # Compute delay_min if absent (fallback: from earliest_start)
    if "delay_min" not in df.columns:
        if "planned_start" in df.columns:
            diff = (df["actual_start"] - df["planned_start"]).dt.total_seconds() / 60.0
            df["delay_min"] = diff.clip(lower=0)
        elif "earliest_start" in df.columns:
            diff = (df["actual_start"] - df["earliest_start"]).dt.total_seconds() / 60.0
            df["delay_min"] = diff.clip(lower=0)
        else:
            df["delay_min"] = 0.0

    return df


def _shift_duration_min(config: dict) -> float:
    """Return shift duration in minutes from config."""
    hours = config.get("data_generator", {}).get("time_window_hours", 8)
    return float(hours) * 60.0


def _vehicle_type_map(routes_df: pd.DataFrame) -> dict[str, str]:
    """
    Derive vehicle_id → vehicle_type from the task assignments.
    Each vehicle serves a single task type in practice.
    """
    if routes_df.empty or "vehicle_type_req" not in routes_df.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in routes_df[["vehicle_id", "vehicle_type_req"]].dropna().iterrows():
        mapping[str(row["vehicle_id"])] = str(row["vehicle_type_req"])
    return mapping


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------


def _compute_kpi(
    routes_df: pd.DataFrame,
    sim_violations: list[dict],
    sim_stats: dict,
    tasks_df: pd.DataFrame,
    config: dict,
) -> dict:
    n_tasks = len(tasks_df)
    n_executed = len(routes_df)

    assigned_rate = n_executed / n_tasks if n_tasks > 0 else 0.0

    on_time_count = int((routes_df["status"] == "on_time").sum()) if not routes_df.empty else 0
    on_time_rate = on_time_count / n_executed if n_executed > 0 else 0.0

    if not routes_df.empty and "delay_min" in routes_df.columns:
        raw_avg = routes_df["delay_min"].mean()
        avg_delay_min = float(raw_avg) if not pd.isna(raw_avg) else 0.0
    else:
        avg_delay_min = 0.0

    violation_count = len(sim_violations)
    cascade_count = int(sim_stats.get("cascade_count", 0))

    # Vehicle utilization: busy minutes / shift minutes per vehicle
    shift_min = _shift_duration_min(config)
    vehicle_utilization: dict[str, float] = {}
    if not routes_df.empty and "actual_start" in routes_df.columns and "actual_end" in routes_df.columns:
        df_copy = routes_df.copy()
        df_copy["service_min"] = (
            (df_copy["actual_end"] - df_copy["actual_start"]).dt.total_seconds() / 60.0
        )
        for vid, grp in df_copy.groupby("vehicle_id"):
            busy = grp["service_min"].sum()
            vehicle_utilization[str(vid)] = round(busy / shift_min * 100, 1) if shift_min > 0 else 0.0

    # Priority on-time rate per priority_group
    priority_on_time: dict[int, float] = {}
    if not routes_df.empty and "priority_group" in routes_df.columns:
        for pg, grp in routes_df.groupby("priority_group"):
            ot = int((grp["status"] == "on_time").sum())
            priority_on_time[int(pg)] = round(ot / len(grp), 4) if len(grp) > 0 else 0.0

    return {
        "assigned_rate":       round(assigned_rate, 4),
        "on_time_rate":        round(on_time_rate, 4),
        "avg_delay_min":       round(avg_delay_min, 2),
        "violation_count":     violation_count,
        "cascade_count":       cascade_count,
        "vehicle_utilization": vehicle_utilization,
        "priority_on_time":    priority_on_time,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def _save_gantt(routes_df: pd.DataFrame, reports_dir: str) -> None:
    """Save Gantt chart of vehicle routes to routes_gantt.png."""
    vehicles = sorted(routes_df["vehicle_id"].unique())
    v_index = {v: i for i, v in enumerate(vehicles)}

    all_starts = routes_df["actual_start"].dropna()
    if all_starts.empty:
        return
    t0 = all_starts.min()

    n_vehicles = len(vehicles)
    fig, ax = plt.subplots(figsize=(14, max(4, n_vehicles * 0.8 + 2)))

    for _, row in routes_df.iterrows():
        y = v_index[row["vehicle_id"]]
        start = row["actual_start"]
        end = row["actual_end"]
        if pd.isna(start) or pd.isna(end):
            continue

        start_offset = (start - t0).total_seconds() / 60.0
        duration = max((end - start).total_seconds() / 60.0, 0.5)

        color = _STATUS_COLORS.get(row.get("status", "on_time"), "#2ca02c")
        ax.barh(y, duration, left=start_offset, height=0.6,
                color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

        # Planned-start dashed marker
        ps = row.get("planned_start")
        if ps is not None and not pd.isna(ps):
            ps_offset = (ps - t0).total_seconds() / 60.0
            ax.plot([ps_offset, ps_offset], [y - 0.35, y + 0.35],
                    color="gray", linestyle="--", linewidth=1.0, alpha=0.75)

    ax.set_yticks(range(n_vehicles))
    ax.set_yticklabels(vehicles, fontsize=8)
    ax.set_xlabel("Time offset (min from first task start)")
    ax.set_title("Vehicle Routes — Gantt Chart")

    legend_handles = [
        mpatches.Patch(color=c, label=s.replace("_", " ").title())
        for s, c in _STATUS_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "routes_gantt.png"), dpi=150)
    plt.close()


def _save_load_chart(
    vehicle_utilization: dict[str, float],
    vehicle_type_map: dict[str, str],
    reports_dir: str,
) -> None:
    """Save vehicle utilization bar chart to load_chart.png."""
    if not vehicle_utilization:
        return

    vehicles = sorted(vehicle_utilization.keys())
    values = [vehicle_utilization[v] for v in vehicles]
    colors = [_VTYPE_COLORS.get(vehicle_type_map.get(v, ""), "#7f7f7f") for v in vehicles]
    avg = sum(values) / len(values)

    fig, ax = plt.subplots(figsize=(max(6, len(vehicles) * 0.9 + 2), 5))
    ax.bar(vehicles, values, color=colors, edgecolor="white", width=0.6)
    ax.axhline(avg, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Vehicle ID")
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Vehicle Fleet Utilization")
    ax.set_ylim(0, max(110.0, max(values) + 10))

    # Legend: vehicle types present + average line
    type_patches = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in _VTYPE_COLORS.items()
        if t in vehicle_type_map.values()
    ]
    avg_line = plt.Line2D([0], [0], color="black", linestyle="--",
                          label=f"Fleet avg: {avg:.1f}%")
    ax.legend(handles=type_patches + [avg_line], fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "load_chart.png"), dpi=150)
    plt.close()


def _save_results_csv(kpi_dict: dict, reports_dir: str) -> None:
    """Flatten kpi_dict and save to results.csv."""
    rows = []
    for key, val in kpi_dict.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                rows.append({"metric": f"{key}.{sub_key}", "value": sub_val})
        else:
            rows.append({"metric": key, "value": val})
    pd.DataFrame(rows).to_csv(os.path.join(reports_dir, "results.csv"), index=False)


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------


def _print_summary(
    kpi_dict: dict,
    n_tasks: int,
    n_executed: int,
    sim_violations: list[dict],
) -> None:
    n_on_time = round(kpi_dict["on_time_rate"] * n_executed)

    # Violation breakdown by reason
    reason_counts: dict[str, int] = {}
    for v in sim_violations:
        r = v.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    viol_str = (
        ", ".join(f"{cnt} {r}" for r, cnt in reason_counts.items())
        if reason_counts else "none"
    )

    util = kpi_dict.get("vehicle_utilization", {})
    util_str = "  ".join(f"{v}={pct}%" for v, pct in sorted(util.items()))

    sep = "-" * 44
    print(f"\n[Metrics] {sep}")
    print(f"  Assigned:       {n_executed} / {n_tasks} tasks  ({kpi_dict['assigned_rate'] * 100:.1f}%)")
    print(f"  On-time:        {n_on_time} / {n_executed} tasks  ({kpi_dict['on_time_rate'] * 100:.1f}%)")
    print(f"  Avg delay:      {kpi_dict['avg_delay_min']:.1f} min")
    print(f"  Violations:     {kpi_dict['violation_count']}  ({viol_str})")
    print(f"  Cascades:       {kpi_dict['cascade_count']}")
    print(f"  Vehicle util:   {util_str if util_str else 'n/a'}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_and_report(
    executed_routes: list[dict],
    sim_violations: list[dict],
    sim_stats: dict,
    tasks_df: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Compute KPI metrics from simulation results and produce reports.

    Aggregates executed_routes and sim_violations into 7 KPI metrics,
    optionally saves Gantt chart (routes_gantt.png), utilization chart
    (load_chart.png), and a flat CSV (results.csv) under reports_dir.
    Prints a formatted KPI summary to stdout.

    Parameters
    ----------
    executed_routes : list[dict]
        Per-task simulation output from simulator.run_simulation().
        Required keys per entry: task_id, vehicle_id, status, and either
        (actual_start, actual_end) or (start_time, end_time).
    sim_violations : list[dict]
        Constraint violations from simulator: [{task_id, reason}, ...].
    sim_stats : dict
        Aggregated simulator statistics — at minimum:
        {total_tasks, on_time, delayed, missed_window, overrun,
         violation_count}. Optional: cascade_count.
    tasks_df : pd.DataFrame
        Full task schedule. Required columns: task_id. Optional but used
        when present: priority_group, earliest_start, STD, vehicle_type_req.
    config : dict
        Pipeline config loaded from scenario YAML. Consumed keys:
          config['metrics']['charts_dir']         (str,  default 'charts')
          config['metrics']['reports_dir']        (str,  default 'reports')
          config['metrics']['save_routes_gantt']  (bool, default True)
          config['metrics']['save_load_chart']    (bool, default True)
          config['metrics']['save_results_csv']   (bool, default True)
          config['data_generator']['time_window_hours'] (float, default 8)

    Returns
    -------
    kpi_dict : dict
        {
          assigned_rate       : float,          # fraction of tasks executed
          on_time_rate        : float,          # fraction executed on time
          avg_delay_min       : float,          # mean delay in minutes
          violation_count     : int,            # total constraint violations
          cascade_count       : int,            # cascade delays (from sim)
          vehicle_utilization : dict[str,float],# {vehicle_id: utilization %}
          priority_on_time    : dict[int,float],# {priority_group: on_time rate}
        }

    Raises
    ------
    ValueError
        If executed_routes is empty.
    OSError
        If the reports directory cannot be created.
    """
    if not executed_routes:
        raise ValueError("executed_routes is empty — nothing to report")

    metrics_cfg = config.get("metrics", {})
    reports_dir: str = metrics_cfg.get("reports_dir", "reports")

    os.makedirs(reports_dir, exist_ok=True)

    routes_df = _build_routes_df(executed_routes, tasks_df)

    kpi_dict = _compute_kpi(routes_df, sim_violations, sim_stats, tasks_df, config)

    if metrics_cfg.get("save_routes_gantt", True) and not routes_df.empty:
        _save_gantt(routes_df, reports_dir)

    vehicle_type_map = _vehicle_type_map(routes_df)

    if metrics_cfg.get("save_load_chart", True):
        _save_load_chart(kpi_dict["vehicle_utilization"], vehicle_type_map, reports_dir)

    if metrics_cfg.get("save_results_csv", True):
        _save_results_csv(kpi_dict, reports_dir)

    _print_summary(kpi_dict, len(tasks_df), len(executed_routes), sim_violations)

    avg_util = (
        sum(kpi_dict["vehicle_utilization"].values()) / len(kpi_dict["vehicle_utilization"])
        if kpi_dict["vehicle_utilization"] else 0.0
    )
    _log(
        "OK",
        f"on_time: {kpi_dict['on_time_rate'] * 100:.1f}%, "
        f"violations: {kpi_dict['violation_count']}, "
        f"utilization: avg {avg_util:.1f}%",
    )

    return kpi_dict
