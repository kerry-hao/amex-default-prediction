"""Integration check with tiny generated fixtures; all files live in pytest tmp_path."""

import numpy as np
import pandas as pd

from amex_default.config import read_json
from amex_default.features import prepare
from amex_default.prediction import predict
from amex_default.reporting import profile, report
from amex_default.training import train
from test_features_and_alignment import config_for


def test_complete_pipeline_on_generated_fixture(tmp_path):
    cfg = config_for(tmp_path)
    cfg["cv"]["folds"] = 3
    cfg["model"].update(n_estimators=12, num_leaves=7, min_child_samples=2,
                        colsample_bytree=1.0, reg_lambda=0.1, n_jobs=1)
    cfg["training"].update(early_stopping_rounds=4, log_period=0)
    cfg["prediction"]["batch_size"] = 7
    rng = np.random.default_rng(8)
    for split, count in [("train", 90), ("test", 17)]:
        ids = [f"{split}-{i:03d}" for i in range(count)]
        rows = []
        for i, customer in enumerate(ids):
            for month in [1, 2, 3]:
                rows.append({"customer_ID": customer, "S_2": f"2020-0{month}-01",
                             "P_2": float(i % 3 == 0) + rng.normal(0, 0.1),
                             "B_1": rng.uniform(), "D_63": "A" if i % 2 else "B"})
        pd.DataFrame(rows).sample(frac=1, random_state=11).to_csv(tmp_path / cfg["paths"][split], index=False)
        if split == "train":
            pd.DataFrame({"customer_ID": ids[::-1], "target": [int(i % 3 == 0) for i in range(count)][::-1]}).to_csv(
                tmp_path / cfg["paths"]["labels"], index=False)
        else:
            pd.DataFrame({"customer_ID": ids[::-1], "prediction": 0}).to_csv(
                tmp_path / cfg["paths"]["sample_submission"], index=False)
        prepare(cfg, split)
    metrics = train(cfg)
    output = tmp_path / cfg["paths"]["output"]
    oof = pd.read_parquet(output / "oof.parquet")
    assert len(oof) == 90 and oof.customer_ID.is_unique
    assert oof.prediction.between(0, 1).all()
    assert set(oof.fold) == {0, 1, 2}
    assert metrics["pooled_oof"]["roc_auc"] > 0.8
    predict(cfg)
    sample = pd.read_csv(tmp_path / cfg["paths"]["sample_submission"])
    submission = pd.read_csv(output / "submission.csv")
    assert submission.customer_ID.tolist() == sample.customer_ID.tolist()
    assert submission.prediction.between(0, 1).all()
    report(cfg)
    profile(cfg)
    assert (output / "results.md").is_file()
    assert (output / "figures/calibration_and_scores.png").is_file()
    assert len(read_json(output / "run.json")["feature_names"]) == metrics["features"]
