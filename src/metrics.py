from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    sse = float(np.sum(err**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE_MB": float(np.mean(np.abs(err))),
        "RMSE_MB": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / y_true)),
        "R2": 1 - sse / sst,
        "under_predicted_%": float(100 * np.mean(err < 0)),
    }
