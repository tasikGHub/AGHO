"""
MLForecast — Airport Ground Handling Optimizer
Predicts service_time for each task using RandomForestRegressor.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def _log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [MLForecast]    {level} — {message}")


class MLForecast:
    """Trains a RandomForestRegressor on synthetic task data and predicts service times."""

    FEATURE_COLS = [
        "task_type_enc",
        "aircraft_type_enc",
        "hour_of_day",
        "stand_id_enc",
    ]

    # Deterministic, fixed encodings
    _TASK_TYPE_MAP = {"deicing": 0, "fueling": 1, "catering": 2}
    _AIRCRAFT_TYPE_MAP = {"narrow": 0, "wide": 1}

    def __init__(self, seed: int = 42, config: dict | None = None):
        ml_cfg = (config or {}).get("ml_model", {})
        n_estimators = int(ml_cfg.get("n_estimators", 100))
        self.seed = seed
        self._n_estimators = n_estimators
        self._test_size = float(ml_cfg.get("test_size", 0.2))
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        self._stand_id_map: dict[str, int] = {}
        self._fallback_mean: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        tasks_df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> tuple[pd.DataFrame, float]:
        """
        Train on history_df (if provided) or tasks_df, then predict on tasks_df.

        Parameters
        ----------
        tasks_df : DataFrame with columns task_type, aircraft_type,
                   hour_of_day, stand_id, service_time_actual
        history_df : optional historical DataFrame with same columns — used for
                     training only; tasks_df is used exclusively for prediction.
                     If None, training and prediction both use tasks_df.

        Returns
        -------
        tasks_df : original DataFrame + column service_time_pred (float)
        mae      : mean absolute error on held-out 20% split of training data
        """
        if tasks_df.empty:
            raise ValueError("tasks_df must not be empty")

        train_df = history_df if history_df is not None else tasks_df

        if train_df.empty:
            raise ValueError("history_df must not be empty when provided")

        y_all = train_df["service_time_actual"].values.astype(float)
        self._fallback_mean = float(np.mean(y_all))

        # Build stand_id encoding from union of training + operational stands (deterministic)
        all_stands = sorted(set(train_df["stand_id"].unique()) | set(tasks_df["stand_id"].unique()))
        self._stand_id_map = {s: i for i, s in enumerate(all_stands)}

        try:
            train_enc = self._encode_features(train_df)
            X_all = train_enc[self.FEATURE_COLS].values.astype(float)

            # Evaluate MAE on held-out split of training data
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=self._test_size, random_state=self.seed
            )
            eval_model = RandomForestRegressor(n_estimators=self._n_estimators, random_state=self.seed)
            eval_model.fit(X_tr, y_tr)
            mae = float(mean_absolute_error(y_te, eval_model.predict(X_te)))

            # Baseline MAE — predicting mean of training targets for all test samples
            baseline_mae = float(mean_absolute_error(y_te, np.full(len(y_te), np.mean(y_tr))))
            improvement = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae > 0 else 0.0

            # Save model report artefacts
            try:
                from model_report import save_model_report
                save_model_report(
                    eval_model=eval_model,
                    X_train=X_tr, X_test=X_te,
                    y_train=y_tr, y_test=y_te,
                    feature_cols=self.FEATURE_COLS,
                    mae=mae,
                    baseline_mae=baseline_mae,
                    improvement=improvement,
                    seed=self.seed,
                    config=config or {},
                )
            except Exception as report_exc:
                _log("WARN", f"model report skipped — {report_exc}")

            # Final model trained on all training data
            self.model.fit(X_all, y_all)

            # Predict on operational tasks
            tasks_enc = self._encode_features(tasks_df)
            X_pred = tasks_enc[self.FEATURE_COLS].values.astype(float)
            preds = np.clip(self.model.predict(X_pred), 1.0, None).round(2)

            if history_df is not None:
                _log("OK", (
                    f"trained on {len(train_df)} historical tasks | "
                    f"MAE (RF): {mae:.1f} min | "
                    f"MAE (baseline): {baseline_mae:.1f} min | "
                    f"Improvement: +{improvement:.0f}%"
                ))
            else:
                _log("OK", (
                    f"MAE (RF): {mae:.1f} min | "
                    f"MAE (baseline): {baseline_mae:.1f} min | "
                    f"Improvement: +{improvement:.0f}%"
                ))

        except Exception as exc:
            # Fallback: predict mean for every task
            y_tasks = tasks_df["service_time_actual"].values.astype(float)
            fallback_arr = np.full(len(y_tasks), self._fallback_mean)
            mae = float(mean_absolute_error(y_tasks, fallback_arr))
            preds = np.clip(fallback_arr, 1.0, None)
            _log("WARN", f"training failed, fallback to mean (MAE: {mae:.1f} min) — {exc}")

        result = tasks_df.copy()
        result["service_time_pred"] = preds
        return result, mae

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["task_type_enc"] = df["task_type"].map(self._TASK_TYPE_MAP).fillna(-1).astype(int)
        df["aircraft_type_enc"] = df["aircraft_type"].map(self._AIRCRAFT_TYPE_MAP).fillna(-1).astype(int)
        df["stand_id_enc"] = df["stand_id"].map(self._stand_id_map).fillna(-1).astype(int)
        return df


def run_ml_forecast(
    tasks_df: pd.DataFrame,
    seed: int = 42,
    config: dict | None = None,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    """Convenience wrapper: instantiate MLForecast and run fit_predict."""
    model = MLForecast(seed=seed, config=config)
    return model.fit_predict(tasks_df, history_df=history_df, config=config)
