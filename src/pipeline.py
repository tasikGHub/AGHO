"""
src/pipeline.py — Airport Ground Handling Optimizer entry point.

Orchestrates: DataGenerator → MLForecast → Optimizer → Simulator → Metrics.
No business logic here — only calls and structured logging.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from data_generator import generate_data
from ml_model import run_ml_forecast
from optimizer import run_optimizer
from simulator import run_simulation
from metrics import compute_and_report

_REQUIRED_SECTIONS = ("data_generator", "vehicles", "apron", "optimizer", "metrics")


def setup_logging() -> None:
    """Configure root logger with [YYYY-MM-DD HH:MM:SS] [Module] format to stdout."""
    fmt = "[%(asctime)s] [%(name)-15s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
        force=True,
    )


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

    root_log = logging.getLogger("Pipeline")
    root_log.info(f"START — config: {args.config}, seed: {args.seed}")

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except (FileNotFoundError, KeyError) as exc:
        root_log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)

    # ── 1. DataGenerator ──────────────────────────────────────────────────────
    log = logging.getLogger("DataGenerator")
    t0 = time.perf_counter()
    try:
        flights_df, tasks_df, vehicles_df, apron_graph = generate_data(config, seed=args.seed)
    except Exception as exc:
        log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)
    n_stands = apron_graph.number_of_nodes() - 1  # exclude DEPOT node
    log.info(
        f"OK — {len(flights_df)} flights, {len(vehicles_df)} vehicles,"
        f" {n_stands} stands generated ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 2. MLForecast ─────────────────────────────────────────────────────────
    log = logging.getLogger("MLForecast")
    t0 = time.perf_counter()
    try:
        tasks_df, mae = run_ml_forecast(tasks_df, seed=args.seed)
    except Exception as exc:
        log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)
    log.info(f"OK — MAE: {mae:.1f} min ({time.perf_counter() - t0:.2f}s)")

    # ── 3. Optimizer ──────────────────────────────────────────────────────────
    log = logging.getLogger("Optimizer")
    t0 = time.perf_counter()
    try:
        assigned_routes, violations = run_optimizer(tasks_df, vehicles_df, apron_graph, config)
    except Exception as exc:
        log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)
    log.info(
        f"OK — {len(assigned_routes)}/{len(tasks_df)} tasks assigned,"
        f" {len(violations)} violations ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 4. Simulator ──────────────────────────────────────────────────────────
    log = logging.getLogger("Simulator")
    t0 = time.perf_counter()
    try:
        executed_routes, sim_violations, sim_stats = run_simulation(
            assigned_routes, tasks_df, vehicles_df, apron_graph, config
        )
    except Exception as exc:
        log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)
    on_time = sim_stats.get("on_time", 0)
    total_sim = sim_stats.get("total_tasks", len(executed_routes))
    cascades = sim_stats.get("cascade_count", 0)
    log.info(
        f"OK — {on_time}/{total_sim} on_time, {cascades} cascade"
        f" ({time.perf_counter() - t0:.2f}s)"
    )

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    log = logging.getLogger("Metrics")
    t0 = time.perf_counter()
    try:
        compute_and_report(executed_routes, sim_violations, sim_stats, tasks_df, config)
    except Exception as exc:
        log.error(f"ERROR — {type(exc).__name__}: {exc}")
        sys.exit(1)
    log.info(f"OK — reports/ saved (gantt, load_chart, results.csv) ({time.perf_counter() - t0:.2f}s)")

    root_log.info(f"DONE — total time: {time.perf_counter() - pipeline_start:.1f}s")


if __name__ == "__main__":
    main()
