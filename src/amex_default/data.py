"""Customer table validation and fold-specific category handling."""

import numpy as np
import pandas as pd


def validate_ids(frame, name):
    if "customer_ID" not in frame or frame["customer_ID"].isna().any():
        raise ValueError(f"{name} needs a complete customer_ID column")
    if frame["customer_ID"].duplicated().any():
        raise ValueError(f"{name} must contain one row per customer")


def align_labels(features, labels):
    validate_ids(features, "Features")
    validate_ids(labels, "Labels")
    if not labels["target"].isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if set(features.customer_ID) != set(labels.customer_ID):
        raise ValueError("Feature and label customer IDs must match exactly")
    return labels.set_index("customer_ID").loc[features.customer_ID, "target"].to_numpy(dtype=np.int8)


def category_maps(frame, categorical):
    return {col: sorted(frame[col].dropna().astype(str).unique().tolist()) for col in categorical}


def model_frame(frame, names, maps):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    result = frame.loc[:, names].copy()
    for col in names:
        if col in maps:
            result[col] = pd.Categorical(result[col].astype("string"), categories=maps[col])
        else:
            result[col] = pd.to_numeric(result[col], errors="raise").replace([np.inf, -np.inf], np.nan).astype("float32")
    return result
