"""
DataGenerator — Airport Ground Handling Optimizer
Generates synthetic flights, tasks, vehicles, and apron graph.
"""

import math
import warnings
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd


class DataGenerator:
    """Generates all synthetic data for the pipeline."""

    TASK_TYPE_TO_VEHICLE = {
        "deicing": "deicing_truck",
        "fueling": "fuel_truck",
        "catering": "catering_truck",
    }

    TASK_TYPE_TO_PRIORITY = {
        "deicing": 1,
        "fueling": 2,
        "catering": 3,
    }

    def __init__(self, config: dict, seed: int = 42):
        if not config:
            raise ValueError("config must not be empty")

        required_sections = {"data_generator", "vehicles", "apron"}
        missing = required_sections - set(config.keys())
        if missing:
            raise ValueError(f"config missing required sections: {missing}")

        self.cfg = config
        self.dg_cfg = config["data_generator"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._validate_config()

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        dg = self.dg_cfg

        # n_flights
        if dg.get("n_flights", 0) < 1:
            raise ValueError("n_flights must be >= 1")

        # n_stands
        if dg.get("n_stands", 0) < 1:
            raise ValueError("n_stands must be >= 1")

        # aircraft_probs must sum to 1.0
        aircraft_types = dg.get("aircraft_types", {})
        if aircraft_types:
            total = sum(aircraft_types.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"aircraft_types probabilities must sum to 1.0, got {total:.6f}"
                )

        # task_types must be known
        unknown = set(dg.get("task_types", [])) - set(self.TASK_TYPE_TO_PRIORITY)
        if unknown:
            raise ValueError(
                f"Unknown task_types in config: {unknown}. "
                f"Supported: {set(self.TASK_TYPE_TO_PRIORITY)}"
            )

        # warn if time window is too narrow
        window_min = dg.get("time_window_hours", 0) * 60
        if window_min <= 90:
            warnings.warn(
                f"time_window_hours ({dg['time_window_hours']}h) leaves no room for STA "
                "spread — all flights will get STA = start_time",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, nx.Graph]:
        """
        Returns
        -------
        flights_df, tasks_df, vehicles_df, apron_graph
        """
        apron_graph = self._build_apron_graph()
        flights_df = self._generate_flights(apron_graph)
        tasks_df = self._generate_tasks(flights_df)
        vehicles_df = self._generate_vehicles()
        return flights_df, tasks_df, vehicles_df, apron_graph

    def generate_history(self, n: int) -> pd.DataFrame:
        """
        Generate historical tasks for ML training only.

        No apron graph or vehicles are built — just flights + tasks.
        Uses seed+1 so historical data does not overlap with operational data.

        Parameters
        ----------
        n : number of historical flights to generate

        Returns
        -------
        tasks_df : DataFrame with the same columns as generate(), ready for ML training
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        # Separate RNG — must not share state with the operational generator
        hist_rng = np.random.default_rng(self.seed + 1)

        dg = self.dg_cfg
        start_dt = datetime.fromisoformat(dg["start_time"])
        window_min = dg["time_window_hours"] * 60

        # Derive stand_ids from config without building the full graph
        n_stands = dg["n_stands"]
        stand_prefix = self.cfg["apron"]["stand_prefix"]
        stands = [f"{stand_prefix}{i:02d}" for i in range(1, n_stands + 1)]

        aircraft_types = list(dg["aircraft_types"].keys())
        aircraft_probs = [dg["aircraft_types"][t] for t in aircraft_types]
        svc_base = dg["service_time_base"]
        task_types = dg["task_types"]
        noise_std = dg["service_time_noise_std"]
        buffer_min = dg["earliest_start_buffer_min"]

        # Generate flights
        flight_rows = []
        for i in range(1, n + 1):
            ac_type = hist_rng.choice(aircraft_types, p=aircraft_probs)
            sta_offset_min = hist_rng.integers(0, max(1, window_min - 90))
            sta = start_dt + timedelta(minutes=int(sta_offset_min))
            total_base = sum(svc_base[tt][ac_type] for tt in task_types)
            turnaround_min = total_base + int(hist_rng.integers(20, 51))
            std = sta + timedelta(minutes=turnaround_min)
            stand_id = stands[(i - 1) % len(stands)]
            flight_rows.append(
                {
                    "flight_id": f"HFL{i:05d}",
                    "aircraft_type": ac_type,
                    "STA": sta,
                    "STD": std,
                    "stand_id": stand_id,
                    "turnaround_min": turnaround_min,
                }
            )

        flights_df = pd.DataFrame(flight_rows)

        # Generate tasks
        task_rows = []
        task_counter = 1
        for _, flight in flights_df.iterrows():
            ac_type = flight["aircraft_type"]
            for tt in task_types:
                base = svc_base[tt][ac_type]
                noise = float(hist_rng.normal(0, noise_std))
                service_time_actual = max(1.0, round(base + noise, 2))
                earliest_start = flight["STA"] + timedelta(minutes=buffer_min)
                task_rows.append(
                    {
                        "task_id": f"HT{task_counter:06d}",
                        "flight_id": flight["flight_id"],
                        "task_type": tt,
                        "priority_group": self.TASK_TYPE_TO_PRIORITY[tt],
                        "STA": flight["STA"],
                        "STD": flight["STD"],
                        "earliest_start": earliest_start,
                        "vehicle_type_req": self.TASK_TYPE_TO_VEHICLE[tt],
                        "service_time_actual": service_time_actual,
                        "aircraft_type": ac_type,
                        "stand_id": flight["stand_id"],
                    }
                )
                task_counter += 1

        tasks_df = pd.DataFrame(task_rows)
        tasks_df["hour_of_day"] = tasks_df["STA"].dt.hour
        return tasks_df

    # ------------------------------------------------------------------
    # Flights
    # ------------------------------------------------------------------

    def _generate_flights(self, apron_graph: nx.Graph) -> pd.DataFrame:
        dg = self.dg_cfg
        n = dg["n_flights"]

        start_dt = datetime.fromisoformat(dg["start_time"])
        window_min = dg["time_window_hours"] * 60

        # Retrieve stand node IDs (exclude depot)
        stand_prefix = self.cfg["apron"]["stand_prefix"]
        stands = sorted(
            [nd for nd in apron_graph.nodes if nd.startswith(stand_prefix)]
        )

        aircraft_types = list(dg["aircraft_types"].keys())
        aircraft_probs = [dg["aircraft_types"][t] for t in aircraft_types]

        svc_base = dg["service_time_base"]
        task_types = dg["task_types"]

        rows = []
        for i in range(1, n + 1):
            flight_id = f"FL{i:03d}"
            ac_type = self.rng.choice(aircraft_types, p=aircraft_probs)

            # STA — uniformly distributed across the shift (leave 90 min buffer at end)
            sta_offset_min = self.rng.integers(0, max(1, window_min - 90))
            sta = start_dt + timedelta(minutes=int(sta_offset_min))

            # turnaround_min = sum of base service times + random buffer [20, 50]
            total_base = sum(svc_base[tt][ac_type] for tt in task_types)
            buffer = int(self.rng.integers(20, 51))
            turnaround_min = total_base + buffer

            std = sta + timedelta(minutes=turnaround_min)

            # Assign stand (round-robin for even distribution)
            stand_id = stands[(i - 1) % len(stands)]

            rows.append(
                {
                    "flight_id": flight_id,
                    "aircraft_type": ac_type,
                    "STA": sta,
                    "STD": std,
                    "stand_id": stand_id,
                    "turnaround_min": turnaround_min,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def _generate_tasks(self, flights_df: pd.DataFrame) -> pd.DataFrame:
        if flights_df.empty:
            raise ValueError("flights_df must not be empty")

        dg = self.dg_cfg
        svc_base = dg["service_time_base"]
        noise_std = dg["service_time_noise_std"]
        buffer_min = dg["earliest_start_buffer_min"]
        task_types = dg["task_types"]

        rows = []
        task_counter = 1
        for _, flight in flights_df.iterrows():
            ac_type = flight["aircraft_type"]
            for tt in task_types:
                task_id = f"T{task_counter:04d}"
                task_counter += 1

                base = svc_base[tt][ac_type]
                noise = float(self.rng.normal(0, noise_std))
                service_time_actual = max(1.0, round(base + noise, 2))

                earliest_start = flight["STA"] + timedelta(minutes=buffer_min)

                rows.append(
                    {
                        "task_id": task_id,
                        "flight_id": flight["flight_id"],
                        "task_type": tt,
                        "priority_group": self.TASK_TYPE_TO_PRIORITY[tt],
                        "STA": flight["STA"],
                        "STD": flight["STD"],
                        "earliest_start": earliest_start,
                        "vehicle_type_req": self.TASK_TYPE_TO_VEHICLE[tt],
                        "service_time_actual": service_time_actual,
                        # service_time_pred filled by ml_model.py
                    }
                )

        df = pd.DataFrame(rows)
        # Merge aircraft_type and stand_id from flights for ML features
        # turnaround_min is intentionally excluded: it is derived from service times
        # and would constitute data leakage for the ML model.
        flights_slim = flights_df[["flight_id", "aircraft_type", "stand_id"]]
        df = df.merge(flights_slim, on="flight_id", how="left")
        df["hour_of_day"] = df["STA"].dt.hour
        return df

    # ------------------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------------------

    def _generate_vehicles(self) -> pd.DataFrame:
        vehicles_cfg = self.cfg["vehicles"]
        depot_id = self.cfg["apron"]["depot_id"]
        start_dt = datetime.fromisoformat(self.dg_cfg["start_time"])

        rows = []
        v_counter = 1
        for spec in vehicles_cfg:
            for _ in range(spec["count"]):
                vehicle_id = f"V{v_counter:02d}"
                v_counter += 1
                rows.append(
                    {
                        "vehicle_id": vehicle_id,
                        "vehicle_type": spec["vehicle_type"],
                        "speed_kmh": float(spec["speed_kmh"]),
                        "capacity": float(spec["capacity"]),
                        "start_stand": depot_id,
                        "free_at": start_dt,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("vehicles_df must not be empty — check config 'vehicles'")
        return df

    # ------------------------------------------------------------------
    # Apron graph
    # ------------------------------------------------------------------

    def _build_apron_graph(self) -> nx.Graph:
        apron_cfg = self.cfg["apron"]
        n_stands = self.dg_cfg["n_stands"]

        rows = apron_cfg["grid_rows"]
        cols = apron_cfg["grid_cols"]
        spacing = apron_cfg["stand_spacing_m"]
        depot_dist = apron_cfg["depot_to_first_stand_m"]
        depot_id = apron_cfg["depot_id"]
        prefix = apron_cfg["stand_prefix"]

        G = nx.Graph()

        # Add depot
        G.add_node(depot_id, node_type="depot")

        # Build stand node IDs in row-major order: S01…S{n_stands}
        # Arrange them in a grid: rows × cols (trim to n_stands)
        stand_ids = [f"{prefix}{i:02d}" for i in range(1, n_stands + 1)]
        if len(stand_ids) > rows * cols:
            raise ValueError(
                f"n_stands ({n_stands}) exceeds grid capacity ({rows}×{cols}={rows*cols})"
            )

        # Add stand nodes with (row, col) metadata
        node_pos: dict[str, tuple[int, int]] = {}
        for idx, sid in enumerate(stand_ids):
            r, c = divmod(idx, cols)
            G.add_node(sid, node_type="stand", row=r, col=c)
            node_pos[sid] = (r, c)

        # Edges between horizontally adjacent stands
        for idx in range(len(stand_ids)):
            r, c = divmod(idx, cols)
            # Right neighbour
            right_idx = idx + 1
            if right_idx < len(stand_ids):
                rr, rc = divmod(right_idx, cols)
                if rr == r:  # same row
                    G.add_edge(stand_ids[idx], stand_ids[right_idx], distance_m=float(spacing))

        # Edges between vertically adjacent stands
        for idx in range(len(stand_ids)):
            r, c = divmod(idx, cols)
            down_idx = idx + cols
            if down_idx < len(stand_ids):
                G.add_edge(stand_ids[idx], stand_ids[down_idx], distance_m=float(spacing))

        # Depot connected to all stands in the first column (col=0) with increasing distance
        for idx, sid in enumerate(stand_ids):
            r, c = node_pos[sid]
            if c == 0:
                dist = depot_dist + r * spacing
                G.add_edge(depot_id, sid, distance_m=float(dist))

        # Verify connectivity
        if not nx.is_connected(G):
            raise RuntimeError("apron_graph is not connected — check grid parameters")

        return G


# ---------------------------------------------------------------------------
# Convenience function used by pipeline.py
# ---------------------------------------------------------------------------

def generate_data(
    config: dict, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, nx.Graph]:
    """Top-level function. Equivalent to DataGenerator(config, seed).generate()."""
    gen = DataGenerator(config, seed)
    return gen.generate()
