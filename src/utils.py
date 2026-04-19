"""
Shared utilities for the Airport Ground Handling Optimizer pipeline.

Centralises cross-cutting helpers so the five pipeline modules
(data_generator, ml_model, optimizer, simulator, metrics) don't each
reimplement timestamped logging or travel-time conversion.
"""

from datetime import datetime


def log(module: str, level: str, message: str) -> None:
    """
    Emit a timestamped structured log line to stdout.

    Format: ``[YYYY-MM-DD HH:MM:SS] [Module] LEVEL — message``

    The module name is left-padded to a fixed width so consecutive
    pipeline stages align visually in terminal output.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{module:<14}] {level} — {message}")


def compute_travel_time_min(
    dist_m: float,
    vehicle_speed_kmh: float,
    max_speed_kmh: float,
) -> float:
    """
    Convert apron distance (metres) to travel time (minutes).

    Effective speed is capped at ``max_speed_kmh`` regardless of the
    vehicle's nominal top speed, matching the apron safety rule.
    """
    effective_speed_kmh = min(vehicle_speed_kmh, max_speed_kmh)
    return dist_m / (effective_speed_kmh * 1000.0 / 60.0)
