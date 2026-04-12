"""
src/pipeline.py — Airport Ground Handling Optimizer entry point.

Orchestrates: DataGenerator → MLForecast → Optimizer → Simulator → Metrics.
No business logic here — only calls and structured logging.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from data_generator import DataGenerator, generate_data
from ml_model import run_ml_forecast
from optimizer import run_optimizer
from simulator import run_simulation
from metrics import compute_and_report

_REQUIRED_SECTIONS = ("data_generator", "vehicles", "apron", "optimizer", "metrics")


def _log(module: str, level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{module}] {level} — {message}")


def setup_logging() -> None:
    """No-op: all pipeline output uses structured print via _log()."""


def load_config(config_path: str) -> dict:
    """
    Load and validate a YAML pipeline config.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed config.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If a required section is missing from the config.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")
    return config


def main() -> None:
    """Entry point. Parses CLI args, loads config, runs all pipeline stages in order."""
    parser = argparse.ArgumentParser(description="Airport Ground Handling Optimizer")
    parser.add_argument("--config", required=True, help="Path to scenario YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    setup_logging()
    pipeline_start = time.perf_counter()

    _log("Pipeline", "START", f"config: {args.config}, seed: {args.seed}")

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except (FileNotFoundError, KeyError) as exc:
        _log("Pipeline", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    # ── 1. DataGenerator ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        flights_df, tasks_df, vehicles_df, apron_graph = generate_data(config, seed=args.seed)
    except Exception as exc:
        _log("DataGenerator", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    n_stands = apron_graph.number_of_nodes() - 1  # exclude DEPOT node
    _log(
        "DataGenerator", "OK",
        f"{len(flights_df)} flights, {len(vehicles_df)} vehicles,"
        f" {n_stands} stands generated ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 1b. Historical data for ML training ───────────────────────────────────
    n_history = config["data_generator"].get("ml_history_flights", 0)
    history_df = None
    if n_history > 0:
        try:
            history_df = DataGenerator(config, seed=args.seed + 1).generate_history(n_history)
        except Exception as exc:
            _log("DataGenerator", "WARN", f"history generation failed, ML will train on operational data — {exc}")

    # ── 2. MLForecast ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        tasks_df, mae = run_ml_forecast(tasks_df, seed=args.seed, config=config, history_df=history_df)
    except Exception as exc:
        _log("MLForecast", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    _log("MLForecast", "OK", f"MAE: {mae:.1f} min ({time.perf_counter() - t0:.2f}s)")

    # ── 3. Optimizer ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        assigned_routes, violations = run_optimizer(tasks_df, vehicles_df, apron_graph, config)
    except Exception as exc:
        _log("Optimizer", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    _log(
        "Optimizer", "OK",
        f"{len(assigned_routes)}/{len(tasks_df)} tasks assigned,"
        f" {len(violations)} violations ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 4. Simulator ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        executed_routes, sim_violations, sim_stats = run_simulation(
            assigned_routes, tasks_df, vehicles_df, apron_graph, config
        )
    except Exception as exc:
        _log("Simulator", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    on_time = sim_stats.get("on_time", 0)
    total_sim = sim_stats.get("total_tasks", len(executed_routes))
    cascades = sim_stats.get("cascade_count", 0)
    _log(
        "Simulator", "OK",
        f"{on_time}/{total_sim} on_time, {cascades} cascade(s)"
        f" ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        compute_and_report(executed_routes, sim_violations, sim_stats, tasks_df, config)
    except Exception as exc:
        _log("Metrics", "ERROR", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    _log("Metrics", "OK", f"reports/ saved (gantt, load_chart, results.csv) ({time.perf_counter() - t0:.2f}s)")

    _log("Pipeline", "DONE", f"total time: {time.perf_counter() - pipeline_start:.1f}s")


if __name__ == "__main__":
    main()
