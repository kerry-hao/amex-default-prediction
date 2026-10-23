"""Stratified customer folds, OOF predictions, and portable LightGBM models."""

import importlib.metadata
import shutil
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import read_json, resolve, save_json
from .data import align_labels, category_maps, model_frame
from .metrics import evaluate, lightgbm_amex


def train(cfg, overwrite=False):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    feature_dir, output = resolve(cfg, "features"), resolve(cfg, "output")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} contains a run; choose a new experiment path or --overwrite")
    meta = read_json(feature_dir / "train" / "_metadata.json")
    if meta["schema"]["recipe"] != cfg["features"]["recipe"] or meta["schema"]["data"] != cfg["data"]:
        raise ValueError("Prepare features with the current configuration before training")
    frame = pd.read_parquet(feature_dir / "train").sort_values("customer_ID").reset_index(drop=True)
    labels = pd.read_csv(resolve(cfg, "labels"), dtype={"customer_ID": "string"})
    target = align_labels(frame, labels)
    folds = int(cfg["cv"]["folds"])
    if np.bincount(target, minlength=2).min() < folds:
        raise ValueError("Each class needs at least one customer per fold")
    names = [c for c in frame.columns if c != "customer_ID"]
    categorical = [c + "__last" for c in meta["schema"]["categorical"]]
    work = output.parent / (".run-" + uuid.uuid4().hex)
    models = work / "models"
    models.mkdir(parents=True)
    oof = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_metrics, importance = [], []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    try:
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame, target)):
            maps = category_maps(frame.iloc[train_idx], categorical)
            x_train = model_frame(frame.iloc[train_idx], names, maps)
            x_valid = model_frame(frame.iloc[valid_idx], names, maps)
            parameters = dict(cfg["model"])
            parameters.update(objective="binary", metric="None", verbosity=-1,
                              random_state=cfg["seed"] + fold, deterministic=True,
                              force_col_wise=True)
            model = lgb.LGBMClassifier(**parameters)
            model.fit(x_train, target[train_idx], categorical_feature=categorical,
                      eval_set=[(x_valid, target[valid_idx])], eval_metric=lightgbm_amex,
                      callbacks=[lgb.early_stopping(cfg["training"]["early_stopping_rounds"],
                                                   first_metric_only=True, verbose=False),
                                 lgb.log_evaluation(cfg["training"]["log_period"])])
            best = int(model.best_iteration_ or model.n_estimators_)
            prediction = model.predict_proba(x_valid, num_iteration=best)[:, 1]
            oof[valid_idx] = prediction
            fold_ids[valid_idx] = fold
            result = {"fold": fold, "train_customers": len(train_idx),
                      "valid_customers": len(valid_idx), "best_iteration": best,
                      **evaluate(target[valid_idx], prediction)}
            fold_metrics.append(result)
            model.booster_.save_model(str(models / f"fold_{fold}.txt"), num_iteration=best)
            save_json(models / f"fold_{fold}_categories.json", maps)
            importance.append(pd.DataFrame({"feature": names,
                                            "gain": model.booster_.feature_importance("gain"),
                                            "split_count": model.booster_.feature_importance("split")}))
            print(f"Fold {fold}: AMEX {result['amex']:.6f}, {best} trees", flush=True)
            del model, x_train, x_valid
        if not np.isfinite(oof).all() or (fold_ids < 0).any():
            raise RuntimeError("OOF predictions must cover every customer exactly once")
        pooled = evaluate(target, oof)
        metrics = {"pooled_oof": pooled,
                   "mean_fold_amex": float(np.mean([m["amex"] for m in fold_metrics])),
                   "std_fold_amex": float(np.std([m["amex"] for m in fold_metrics])),
                   "customers": len(frame), "features": len(names),
                   "probability_population": "sampled competition population"}
        pd.DataFrame({"customer_ID": frame.customer_ID, "target": target,
                      "fold": fold_ids, "prediction": oof}).to_parquet(work / "oof.parquet", index=False)
        pd.DataFrame(fold_metrics).to_csv(work / "fold_metrics.csv", index=False)
        gains = pd.concat(importance).groupby("feature", as_index=False)[["gain", "split_count"]].mean()
        gains.sort_values("gain", ascending=False).to_csv(work / "feature_importance.csv", index=False)
        save_json(work / "metrics.json", metrics)
        versions = {name: importlib.metadata.version(name) for name in
                    ["numpy", "pandas", "lightgbm", "scikit-learn", "duckdb", "pyarrow"]}
        save_json(work / "run.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "config": cfg, "versions": versions,
                                     "train_prepare_id": meta["prepare_id"],
                                     "schema_signature": meta["schema_signature"],
                                     "feature_names": names, "folds": folds,
                                     "categorical_features": categorical})
        if output.exists():
            shutil.rmtree(output)
        work.rename(output)
        print(f"Pooled OOF AMEX: {pooled['amex']:.6f}", flush=True)
        return metrics
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
