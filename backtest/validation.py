"""
Split-sample calibration validation: do the probabilities survive out of
sample?

Everything else in the harness calibrates and scores on the same window,
which flatters the numbers. Here the calibration material (residual clouds,
conviction-bucket accuracies) is built ONLY from as-of dates before the
split year, then applied to predictions made from the split year onward.
If the out-of-sample columns roughly match the in-sample ones, the
probabilities are trustworthy enough to publish; if they collapse, the
in-sample numbers were overfit and no subscriber should see them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import monthly as monthly_mod
from .probability import delta_residuals, residual_quad_probability
from .scoring import _attach_realized


def _split(df: pd.DataFrame, split_year: int):
    cut = pd.Timestamp(f"{split_year}-01-01")
    asof = pd.to_datetime(df["asof"])
    return df[asof < cut], df[asof >= cut]


def split_validation(preds: pd.DataFrame, realized: pd.DataFrame,
                     mdir: pd.DataFrame, split_year: int = 2019,
                     horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """One row per metric: in-sample (train window, calibrated on itself)
    vs out-of-sample (test window, calibrated only on the train window)."""
    rows = []

    train_p, test_p = _split(preds, split_year)
    if len(train_p) and len(test_p):
        resid_train = delta_residuals(train_p, realized, horizons)
        jtrain = _attach_realized(train_p, realized)
        jtest = _attach_realized(test_p, realized)
        for h in horizons:
            tr = jtrain[jtrain["horizon"] == h]
            te = jtest[jtest["horizon"] == h]
            if not len(tr) or not len(te) or not len(resid_train[h]):
                continue

            def mean_p_realized(sub):
                ps = [residual_quad_probability(
                    r["pred_d_growth"], r["pred_d_inflation"],
                    resid_train[h])[int(r["real_quad"])]
                    for _, r in sub.iterrows()]
                return float(np.mean(ps))

            rows.append({
                "metric": f"quad_hit_h{h}",
                "n_train": len(tr), "n_test": len(te),
                "in_sample": float((tr["pred_quad"]
                                    == tr["real_quad"]).mean()),
                "out_of_sample": float((te["pred_quad"]
                                        == te["real_quad"]).mean()),
            })
            rows.append({
                "metric": f"quad_mean_prob_of_outcome_h{h}",
                "n_train": len(tr), "n_test": len(te),
                "in_sample": mean_p_realized(tr),
                "out_of_sample": mean_p_realized(te),
            })

    train_m, test_m = _split(mdir, split_year)
    if len(train_m) and len(test_m):
        for lo, hi, label in monthly_mod.CONVICTION_BUCKETS:
            tr = train_m[(train_m["conviction_pp"] >= lo)
                         & (train_m["conviction_pp"] < hi)]
            te = test_m[(test_m["conviction_pp"] >= lo)
                        & (test_m["conviction_pp"] < hi)]
            if len(tr) < 5 or len(te) < 5:
                continue
            rows.append({
                "metric": f"monthly_dir [{label}]",
                "n_train": len(tr), "n_test": len(te),
                "in_sample": float(tr["hit"].mean()),
                "out_of_sample": float(te["hit"].mean()),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["oos_minus_is"] = out["out_of_sample"] - out["in_sample"]
    return out
