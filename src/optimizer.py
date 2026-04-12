"""
Optimizer — Airport Ground Handling Optimizer
Greedy rule-based vehicle assignment and route planning.
"""

from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd


def _log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [Optimizer]     {level} — {message}")


class Optimizer:
    """
    Greedy vehicle scheduler for airport ground handling tasks.

    Sorts tasks by priority group (deicing > fueling > catering) then
    by urgency, and assigns the earliest-available compatible vehicle
    to each task using shortest-path routing on the apron graph.
    """

    def __init__(
        self,
        tasks_df: pd.DataFrame,
        vehicles_df: pd.DataFrame,
        apron_graph: nx.Graph,
        config: dict,
    ) -> None:
        if tasks_df is None or tasks_df.empty:
            raise ValueError("tasks_df must not be empty")
        if vehicles_df is None or vehicles_df.empty:
            raise ValueError("vehicles_df must not be empty")
        if "optimizer" not in config:
            raise ValueError("config missing required section: 'optimizer'")

        self._validate_graph(apron_graph)
        self._validate_stand_ids(tasks_df, apron_graph)

        self.tasks_df = tasks_df.copy()
        self.vehicles_df = vehicles_df.copy()
        self.apron_graph = apron_graph
        self.config = config
        self.opt_cfg = config["optimizer"]

        self.safe_interval_min = float(self.opt_cfg["safe_interval_min"])
        self.max_speed_kmh = float(self.opt_cfg["max_speed_kmh"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> tuple[list[dict], list[dict]]:
        """
        Run the greedy optimizer.

        Returns
        -------
        assigned_routes : list of dicts {task_id, vehicle_id, start_time, end_time, route}
        violations      : list of dicts {task_id, reason}
        """
        df = self._compute_urgency(self.tasks_df)
        sorted_df = self._sort_tasks(df)
        v_state = self._init_vehicle_state()
        stand_last_end: dict[str, datetime] = {}

        assigned_routes, violations = self._assign_tasks(sorted_df, v_state, stand_last_end)

        n_total = len(sorted_df)
        n_assigned = len(assigned_routes)
        _log("OK", f"{n_assigned}/{n_total} tasks assigned")

        return assigned_routes, violations

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_graph(self, apron_graph: nx.Graph) -> None:
        if apron_graph is None or apron_graph.number_of_nodes() == 0:
            raise RuntimeError("apron_graph is empty")
        if not nx.is_connected(apron_graph):
            raise RuntimeError("apron_graph is not connected")

    def _validate_stand_ids(self, tasks_df: pd.DataFrame, apron_graph: nx.Graph) -> None:
        graph_nodes = set(apron_graph.nodes)
        unknown = set(tasks_df["stand_id"].unique()) - graph_nodes
        if unknown:
            raise ValueError(
                f"tasks_df references stand_id(s) not in apron_graph: {unknown}"
            )

    # ------------------------------------------------------------------
    # Step 1 — urgency
    # ------------------------------------------------------------------

    def _compute_urgency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle missing or NaN service_time_pred
        if "service_time_pred" not in df.columns or df["service_time_pred"].isna().any():
            if "service_time_actual" in df.columns:
                _log("WARN", "service_time_pred has NaN — falling back to service_time_actual")
                if "service_time_pred" not in df.columns:
                    df["service_time_pred"] = df["service_time_actual"]
                else:
                    df["service_time_pred"] = df["service_time_pred"].fillna(
                        df["service_time_actual"]
                    )
            else:
                _log("WARN", "service_time_pred missing and no fallback — using 0.0")
                if "service_time_pred" not in df.columns:
                    df["service_time_pred"] = 0.0
                else:
                    df["service_time_pred"] = df["service_time_pred"].fillna(0.0)

        df["urgency"] = (
            (df["STD"] - df["earliest_start"]).dt.total_seconds() / 60.0
            - df["service_time_pred"]
        )
        return df

    # ------------------------------------------------------------------
    # Step 2 — sort
    # ------------------------------------------------------------------

    def _sort_tasks(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            by=["priority_group", "urgency"],
            ascending=[True, True],
            ignore_index=True,
        )

    # ------------------------------------------------------------------
    # Vehicle state initialisation
    # ------------------------------------------------------------------

    def _init_vehicle_state(self) -> dict:
        state: dict[str, dict] = {}
        for _, row in self.vehicles_df.iterrows():
            state[row["vehicle_id"]] = {
                "vehicle_type": row["vehicle_type"],
                "speed_kmh": float(row["speed_kmh"]),
                "capacity": float(row["capacity"]),
                "current_pos": row["start_stand"],
                "free_at": row["free_at"],
            }
        return state

    # ------------------------------------------------------------------
    # Travel time helper
    # ------------------------------------------------------------------

    def _get_travel(
        self,
        v_state_entry: dict,
        stand_id: str,
    ) -> tuple[float, list]:
        """
        Returns (travel_time_min, route_node_list).

        Raises RuntimeError if no path exists.
        """
        current_pos = v_state_entry["current_pos"]

        if current_pos == stand_id:
            return 0.0, [stand_id]

        try:
            dist_m, path = nx.single_source_dijkstra(
                self.apron_graph,
                source=current_pos,
                target=stand_id,
                weight="distance_m",
            )
        except nx.NetworkXNoPath:
            raise RuntimeError(
                f"No path from '{current_pos}' to '{stand_id}' in apron_graph"
            )
        except nx.NodeNotFound as exc:
            raise RuntimeError(f"apron_graph node not found: {exc}")

        effective_speed_kmh = min(v_state_entry["speed_kmh"], self.max_speed_kmh)
        # distance_m / (m/min) = minutes;  speed_kmh * 1000/60 = m/min
        travel_time_min = dist_m / (effective_speed_kmh * 1000.0 / 60.0)
        return travel_time_min, path

    # ------------------------------------------------------------------
    # Safe interval enforcement
    # ------------------------------------------------------------------

    def _enforce_safe_interval(
        self,
        stand_id: str,
        candidate_start: datetime,
        stand_last_end: dict,
    ) -> datetime:
        last_end = stand_last_end.get(stand_id)
        if last_end is None:
            return candidate_start
        earliest_allowed = last_end + timedelta(minutes=self.safe_interval_min)
        return max(candidate_start, earliest_allowed)

    # ------------------------------------------------------------------
    # Step 3 — greedy assignment
    # ------------------------------------------------------------------

    def _assign_tasks(
        self,
        sorted_df: pd.DataFrame,
        v_state: dict,
        stand_last_end: dict,
    ) -> tuple[list, list]:
        assigned_routes: list[dict] = []
        violations: list[dict] = []

        for _, task in sorted_df.iterrows():
            task_id = task["task_id"]
            stand_id = task["stand_id"]
            req_type = task["vehicle_type_req"]
            earliest_start: datetime = task["earliest_start"]
            std: datetime = task["STD"]
            svc_time: float = task["service_time_pred"]

            # Filter candidates by vehicle type
            candidates = [
                (vid, vs)
                for vid, vs in v_state.items()
                if vs["vehicle_type"] == req_type
            ]

            if not candidates:
                violations.append({"task_id": task_id, "reason": "no_vehicle_of_type"})
                continue

            # Score each candidate: (infeasible_flag, actual_start) — minimise
            # infeasible_flag=0 means feasible (preferred), 1 means not
            best_score: tuple | None = None
            best_vid: str | None = None
            best_actual_start: datetime | None = None
            best_end_time: datetime | None = None
            best_route: list | None = None
            best_feasible: bool = False

            for vid, vs in candidates:
                try:
                    travel_time_min, route = self._get_travel(vs, stand_id)
                except RuntimeError:
                    continue  # unreachable path for this vehicle; try next

                raw_start = vs["free_at"] + timedelta(minutes=travel_time_min)
                actual_start = max(raw_start, earliest_start)
                actual_start = self._enforce_safe_interval(
                    stand_id, actual_start, stand_last_end
                )
                end_time = actual_start + timedelta(minutes=svc_time)
                feasible = end_time <= std

                score = (0 if feasible else 1, actual_start)
                if best_score is None or score < best_score:
                    best_score = score
                    best_vid = vid
                    best_actual_start = actual_start
                    best_end_time = end_time
                    best_route = route
                    best_feasible = feasible

            if best_vid is None:
                violations.append({"task_id": task_id, "reason": "no_reachable_vehicle"})
                continue

            # Record time-window violation (task still gets assigned)
            if not best_feasible:
                violations.append({"task_id": task_id, "reason": "time_window_violated"})

            assigned_routes.append(
                {
                    "task_id": task_id,
                    "vehicle_id": best_vid,
                    "start_time": best_actual_start,
                    "end_time": best_end_time,
                    "route": best_route,
                }
            )

            # Update vehicle state
            v_state[best_vid]["free_at"] = best_end_time
            v_state[best_vid]["current_pos"] = stand_id

            # Update stand safe-interval tracker
            prev = stand_last_end.get(stand_id)
            if prev is None or best_end_time > prev:
                stand_last_end[stand_id] = best_end_time

        return assigned_routes, violations


# ---------------------------------------------------------------------------
# Convenience function used by pipeline.py
# ---------------------------------------------------------------------------

def run_optimizer(
    tasks_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    apron_graph: nx.Graph,
    config: dict,
) -> tuple[list, list]:
    """Top-level function. Equivalent to Optimizer(...).optimize()."""
    opt = Optimizer(tasks_df, vehicles_df, apron_graph, config)
    return opt.optimize()
