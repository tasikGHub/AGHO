"""
model_report.py — saves ML model metrics, plots, and summary to model_params/
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_model_report(
    eval_model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_cols: list[str],
    mae: float,
    baseline_mae: float,
    improvement: float,
    seed: int,
    config: dict,
) -> None:
    """
    Save ML model report artefacts to model_params/:
      - metrics.json
      - feature_importance.png
      - correlation_matrix.png
      - model_summary.txt
    """
    out_dir = Path(
        (config.get("metrics") or {}).get("model_params_dir", "model_params")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(X_train)
    n_test = len(X_test)
    total = n_train + n_test
    test_size = round(n_test / total, 2) if total > 0 else 0.2
    n_features = len(feature_cols)
    n_estimators = int((config.get("ml_model") or {}).get("n_estimators", 100))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. metrics.json ──────────────────────────────────────────────────────
    metrics = {
        "mae_rf": round(mae, 1),
        "mae_baseline": round(baseline_mae, 1),
        "improvement_pct": round(improvement, 1),
        "n_train": n_train,
        "n_test": n_test,
        "test_size": test_size,
        "n_features": n_features,
        "n_estimators": n_estimators,
        "seed": seed,
        "timestamp": timestamp,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ── 2. feature_importance.png ─────────────────────────────────────────────
    importances = eval_model.feature_importances_
    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(feature_cols))
    ax.barh(y_pos, importances, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (RandomForest)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=100)
    plt.close(fig)

    # ── 3. correlation_matrix.png ─────────────────────────────────────────────
    df_corr = pd.DataFrame(X_test, columns=feature_cols)
    df_corr["service_time_actual"] = y_test
    corr = df_corr.corr()
    labels = list(corr.columns)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=100)
    plt.close(fig)

    # ── 4. model_summary.txt ──────────────────────────────────────────────────
    lines = [
        "=== ML Model Report ===",
        f"Timestamp:        {timestamp}",
        f"Training samples: {n_train}",
        f"Test samples:     {n_test}",
        f"Test size:        {int(test_size * 100)}%",
        "",
        "--- Performance ---",
        f"MAE (RandomForest):  {mae:.1f} min",
        f"MAE (Baseline mean): {baseline_mae:.1f} min",
        f"Improvement:         +{improvement:.0f}%",
        "",
        "--- Feature Importance ---",
    ]
    for col, imp in zip(feature_cols, importances):
        lines.append(f"{col:<20} {imp:.3f}")

    with open(out_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
