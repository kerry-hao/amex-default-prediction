"""Batch inference, mean-margin ensemble, and ID-aligned submission."""

import numpy as np
import pandas as pd

from .config import read_json, resolve, save_json
from .data import model_frame, validate_ids
from .metrics import sigmoid


def align_submission(predictions, sample):
    validate_ids(predictions, "Predictions")
    validate_ids(sample, "Sample submission")
    if set(predictions.customer_ID) != set(sample.customer_ID):
        raise ValueError("Prediction IDs must exactly match sample submission IDs")
    result = sample[["customer_ID"]].merge(predictions, on="customer_ID", how="left", validate="one_to_one")
    if not np.isfinite(result.prediction).all() or not result.prediction.between(0, 1).all():
        raise ValueError("Submission needs one finite probability per customer")
    return result


def predict(cfg):
    import lightgbm as lgb
    import pyarrow.dataset as ds

    features, output = resolve(cfg, "features"), resolve(cfg, "output")
    run = read_json(output / "run.json")
    train_meta = read_json(features / "train" / "_metadata.json")
    test_meta = read_json(features / "test" / "_metadata.json")
    if train_meta["prepare_id"] != run["train_prepare_id"]:
        raise ValueError("Train features changed; use the matching feature set or retrain")
    if test_meta["schema_signature"] != run["schema_signature"]:
        raise ValueError("Prepare test features with the training schema and recipe")
    # This small ID scan checks separation without loading test predictors into RAM.
    train_ids = pd.read_parquet(features / "train", columns=["customer_ID"])
    test_ids = pd.read_parquet(features / "test", columns=["customer_ID"])
    validate_ids(test_ids, "Test features")
    if set(train_ids.customer_ID) & set(test_ids.customer_ID):
        raise ValueError("Training and test customer IDs must be disjoint")
    if cfg["prediction"]["ensemble"] != "mean_raw_margin_then_sigmoid":
        raise ValueError("Use mean_raw_margin_then_sigmoid for this pipeline")
    dataset = ds.dataset(features / "test", format="parquet", exclude_invalid_files=True)
    scores = np.zeros(len(test_ids), dtype=np.float64)
    # Process one fold at a time to keep only one booster in memory.
    ordered_ids = None
    for fold in range(run["folds"]):
        model = lgb.Booster(model_file=str(output / "models" / f"fold_{fold}.txt"))
        maps = read_json(output / "models" / f"fold_{fold}_categories.json")
        offset, ids = 0, []
        for batch in dataset.to_batches(batch_size=cfg["prediction"]["batch_size"]):
            frame = batch.to_pandas()
            x = model_frame(frame, run["feature_names"], maps)
            raw = model.predict(x, raw_score=True)
            scores[offset:offset+len(frame)] += raw / run["folds"]
            offset += len(frame)
            ids.extend(frame.customer_ID.tolist())
        if offset != len(scores):
            raise ValueError("Unexpected test row count")
        if ordered_ids is None:
            ordered_ids = ids
        elif ordered_ids != ids:
            raise ValueError("Test row order changed between folds")
        del model
    predictions = pd.DataFrame({"customer_ID": ordered_ids, "prediction": sigmoid(scores)})
    sample = pd.read_csv(resolve(cfg, "sample_submission"), dtype={"customer_ID": "string"})
    submission = align_submission(predictions, sample)
    # Write only after every ID and value passes validation.
    submission.to_csv(output / "submission.csv", index=False)
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    save_json(output / "prediction_summary.json", {"customers": len(predictions),
              "mean_prediction": float(predictions.prediction.mean()),
              "test_prepare_id": test_meta["prepare_id"],
              "ensemble": cfg["prediction"]["ensemble"]})
    return {"submission": str(output / "submission.csv"), "customers": len(submission)}
