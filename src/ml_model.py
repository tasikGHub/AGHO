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

    _RF_PARAMS = {"n_estimators": 100}

    # Deterministic, fixed encodings
    _TASK_TYPE_MAP = {"deicing": 0, "fueling": 1, "catering": 2}
    _AIRCRAFT_TYPE_MAP = {"narrow": 0, "wide": 1}

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = RandomForestRegressor(**self._RF_PARAMS, random_state=seed)
        self._stand_id_map: dict[str, int] = {}
        self._fallback_mean: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        tasks_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        """
        Train on tasks_df and predict service_time_pred for all tasks.

        Parameters
        ----------
        tasks_df : DataFrame with columns task_type, aircraft_type,
                   hour_of_day, stand_id, service_time_actual
        seed : int  (set via __init__)

        Returns
        -------
        tasks_df : original DataFrame + column service_time_pred (float)
        mae      : mean absolute error on held-out 20% split (float)
        """
        if tasks_df.empty:
            raise ValueError("tasks_df must not be empty")

        df = tasks_df.copy()
        y = df["service_time_actual"].values.astype(float)
        self._fallback_mean = float(np.mean(y))

        # Build stand_id encoding from sorted unique values (deterministic)
        sorted_stands = sorted(df["stand_id"].unique())
        self._stand_id_map = {s: i for i, s in enumerate(sorted_stands)}

        try:
            df_enc = self._encode_features(df)
            X = df_enc[self.FEATURE_COLS].values.astype(float)

            # Evaluate MAE on held-out 20%
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            eval_model = RandomForestRegressor(**self._RF_PARAMS, random_state=self.seed)
            eval_model.fit(X_train, y_train)
            mae = float(mean_absolute_error(y_test, eval_model.predict(X_test)))

            # Final model trained on all data
            self.model.fit(X, y)
            preds = np.clip(self.model.predict(X), 1.0, None).round(2)

            _log("OK", f"MAE: {mae:.1f} min")

        except Exception as exc:
            # Fallback: predict mean for every task
            fallback_arr = np.full(len(y), self._fallback_mean)
            mae = float(mean_absolute_error(y, fallback_arr))
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

        unknown_tasks = set(df["task_type"].unique()) - set(self._TASK_TYPE_MAP)
        if unknown_tasks:
            raise ValueError(f"Unknown task_type values: {unknown_tasks}")

        unknown_aircraft = set(df["aircraft_type"].unique()) - set(self._AIRCRAFT_TYPE_MAP)
        if unknown_aircraft:
            raise ValueError(f"Unknown aircraft_type values: {unknown_aircraft}")

        df["task_type_enc"] = df["task_type"].map(self._TASK_TYPE_MAP)
        df["aircraft_type_enc"] = df["aircraft_type"].map(self._AIRCRAFT_TYPE_MAP)
        # Unknown stand_id → -1 (new stand not seen during fit is acceptable)
        df["stand_id_enc"] = df["stand_id"].map(self._stand_id_map).fillna(-1)
        return df


# ---------------------------------------------------------------------------
# Convenience function used by pipeline.py
# ---------------------------------------------------------------------------

def run_ml_forecast(
    tasks_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Top-level function. Equivalent to MLForecast(seed).fit_predict(tasks_df)."""
    model = MLForecast(seed=seed)
    return model.fit_predict(tasks_df)
