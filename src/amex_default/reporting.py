"""Compact OOF charts and customer-level data profiles."""

import numpy as np
import pandas as pd

from .config import read_json, resolve
from .data import align_labels


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.bbox": "tight"})
    return plt


def report(cfg):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    plt = setup_plotting()
    output = resolve(cfg, "output")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(output / "oof.parquet")
    metrics = read_json(output / "metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    RocCurveDisplay.from_predictions(oof.target, oof.prediction, ax=axes[0],
                                    curve_kwargs={"color": "#176B63"})
    PrecisionRecallDisplay.from_predictions(oof.target, oof.prediction, ax=axes[1], color="#176B63")
    axes[0].set_title("OOF ROC curve")
    axes[1].set_title("OOF precision recall curve")
    fig.savefig(figures / "discrimination.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    observed, predicted = calibration_curve(oof.target, oof.prediction, n_bins=10, strategy="quantile")
    axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF")
    axes[0].plot(predicted, observed, "o-", color="#176B63")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed default fraction",
                title="OOF calibration in sampled population")
    for target, color in [(0, "#176B63"), (1, "#D28B40")]:
        axes[1].hist(oof.loc[oof.target == target, "prediction"], bins=np.linspace(0, 1, 41),
                     density=True, alpha=0.6, color=color, label=f"target = {target}")
    axes[1].set(xlabel="Predicted probability", ylabel="Density", title="OOF score distributions")
    axes[1].legend()
    fig.savefig(figures / "calibration_and_scores.png")
    plt.close(fig)
    importance = pd.read_csv(output / "feature_importance.csv").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.barh(importance.feature, importance.gain, color="#176B63")
    ax.set(xlabel="Mean gain across folds", title="Top 20 features")
    fig.savefig(figures / "feature_importance.png")
    plt.close(fig)
    rows = "\n".join(f"| {key} | {value:.6f} |" for key, value in metrics["pooled_oof"].items())
    text = ("# Customer level validation results\n\n"
            f"Customers: {metrics['customers']:,}. Features: {metrics['features']:,}.\n\n"
            "| Metric | Pooled OOF value |\n| --- | ---: |\n" + rows + "\n\n"
            f"Mean fold AMEX: {metrics['mean_fold_amex']:.6f}.\n\n"
            "Probability diagnostics describe the sampled competition population.\n\n"
            "![ROC and precision recall](figures/discrimination.png)\n\n"
            "![Calibration and score distributions](figures/calibration_and_scores.png)\n\n"
            "![Feature importance](figures/feature_importance.png)\n")
    (output / "results.md").write_text(text, encoding="utf-8")
    return {"report": str(output / "results.md")}


def profile(cfg):
    """Summarize prepared customer features, loading train and test sequentially."""
    plt = setup_plotting()
    root, output = resolve(cfg, "features"), resolve(cfg, "output") / "eda"
    output.mkdir(parents=True, exist_ok=True)
    summaries, missing = [], []
    for split in ["train", "test"]:
        if not (root / split).exists():
            continue
        frame = pd.read_parquet(root / split)
        meta = read_json(root / split / "_metadata.json")
        summaries.append({"split": split, **meta["statistics"], "features": len(frame.columns)-1})
        missing.append(frame.drop(columns="customer_ID").isna().mean().rename(split))
        counts = frame.n_statements.value_counts().sort_index()
        counts.rename("customers").to_csv(output / f"{split}_statement_counts.csv", index_label="statements")
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        ax.bar(counts.index, counts.values, color="#176B63")
        ax.set(xlabel="Statements per customer", ylabel="Customers", title=f"{split.title()} history length")
        fig.savefig(output / f"{split}_history.png")
        plt.close(fig)
        if split == "train":
            labels = pd.read_csv(resolve(cfg, "labels"), dtype={"customer_ID": "string"})
            y = align_labels(frame, labels)
            pd.Series(y).value_counts().sort_index().rename("customers").to_csv(output / "label_counts.csv", index_label="target")
        del frame
    if not summaries:
        raise FileNotFoundError("Prepare train features before generating a data profile")
    pd.DataFrame(summaries).to_csv(output / "data_summary.csv", index=False)
    rates = pd.concat(missing, axis=1)
    if "train" in rates and "test" in rates:
        rates["test_minus_train"] = rates.test - rates.train
    rates.to_csv(output / "feature_missingness.csv", index_label="feature")
    return {"profile": str(output)}
