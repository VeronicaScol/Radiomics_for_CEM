"""Modeling pipeline (v2): preprocessing + feature selection + model fitting/evaluation.

What this script does
---------------------
1) Loads train/test CSVs and splits into X/y.
2) Preprocesses features (fit on train, apply to test) via utils_refactored.
3) For each (model, feature-selection strategy) pair:
   - selects features on TRAIN ONLY
   - trains the model on selected TRAIN features
   - picks an optimal threshold on train probabilities
   - evaluates test-set metrics with bootstrap CIs
   - computes confusion matrix (raw + normalized)
4) Outputs two comparison tables:
   - metrics_comparison_v2.csv
   - feature_overlap_jaccard_v2.csv

Requirements
-----------
- feature_selection_utils.py must be importable (same folder or PYTHONPATH)
- utils_refactored.py must be importable
- xgboost installed (for XGBClassifier).

Run
---
python modeling_pipeline_refactored_v2.py --train_csv path/to/train.csv --test_csv path/to/test.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import xgboost as xgb

# local utilities
from utils_refactored import (
    preprocessing_train,
    preprocessing_test,
    predict_proba_1,
    get_optimal_threshold,
    get_stats_with_ci,
)

# feature selection utilities (your module)
import feature_selection_utils as fsu


@dataclass
class ExperimentResult:
    model_name: str
    selector_name: str
    selected_features: List[str]
    support_mask: np.ndarray
    optimal_threshold: float
    confusion_matrix: np.ndarray
    confusion_matrix_norm: np.ndarray
    metrics_test: pd.DataFrame
    extra: Dict[str, Any]


def load_feature_csv(
    train_csv: str,
    test_csv: str,
    outcome_col: str = "outcome",
    drop_cols: Tuple[str, ...] = ("mask_name", "outcome"),
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load train/test CSVs and split into X/y."""
    df_train = pd.read_csv(train_csv)
    y_train = df_train[outcome_col].to_numpy()
    X_train = df_train.drop(list(drop_cols), axis=1, errors="ignore")

    df_test = pd.read_csv(test_csv)
    y_test = df_test[outcome_col].to_numpy()
    X_test = df_test.drop(list(drop_cols), axis=1, errors="ignore")

    return X_train, y_train, X_test, y_test


def _top_n_from_ranking(ranking: np.ndarray, n: int) -> np.ndarray:
    """Convert a sklearn-style ranking_ array into a boolean mask selecting the top-n features.

    ranking_: 1 is best, larger is worse.
    """
    order = np.argsort(ranking)
    keep_idx = order[:n]
    mask = np.zeros_like(ranking, dtype=bool)
    mask[keep_idx] = True
    return mask


def select_features(
    selector_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model,
    top_n: int = 10,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """Dispatch to a feature-selection strategy.

    Returns:
        selected_features: list of column names
        support_mask: boolean mask aligned to X_train.columns
        extra: metadata (e.g., best_k, cv scores)
    """
    extra: Dict[str, Any] = {}

    if selector_name == "anova_cv":
        # SelectKBest (ANOVA by default) where k is tuned by CV performance of the model.
        selected, best_k, best_score, mask = fsu.filter_method_cv(
            X_train,
            y_train,
            model=model,
            return_support=True,
        )
        extra.update({"best_k": best_k, "best_score": best_score})
        return selected, mask.astype(bool), extra

    if selector_name == "rfe_no_cv_top10":
        selected, mask = fsu.rfe_no_cv(
            X_train,
            y_train,
            n_features=top_n,
            estimator=model,
            return_support=True,
        )
        return selected, mask.astype(bool), extra

    if selector_name == "rfecv_rank_top10":
        # RFECV is used to rank features using CV; final training uses top-N by ranking.
        # This satisfies the requirement: "RFECV should select the top 10 features".
        _selected, best_n, best_score, _mask, selector = fsu.rfe_with_cv(
            X_train,
            y_train,
            estimator=model,
            # ensure it can evaluate down to 10 features
            min_features_to_select=top_n,
            n_splits=10,
            scoring="roc_auc",
            return_support=True,
        )
        ranking = getattr(selector, "ranking_", None)
        if ranking is None:
            raise RuntimeError("RFECV selector did not expose ranking_.")

        mask = _top_n_from_ranking(ranking, top_n)
        selected = list(X_train.columns[mask])
        extra.update({"rfecv_best_n_features": best_n, "rfecv_best_score": best_score, "rfecv": selector})
        return selected, mask.astype(bool), extra

    if selector_name == "embedded_method":
        # For tree/boosting models, embedded importance is natural.
        selected, mask = fsu.embedded_method(
            X_train,
            y_train,
            n_features=top_n,
            model=model,
            return_support=True,
        )
        return selected, mask.astype(bool), extra

    if selector_name == "lasso_logregcv":
        # L1-logistic CV selector (its own model inside a pipeline). Returns selected features.
        selected, best_C, n_selected, mask, pipe = fsu.embedded_l1_logregcv(
            X_train,
            y_train,
            return_support=True,
        )
        # If it selects more than top_n, we keep all by design; project requirement did not
        # require forcing lasso to 10. If you *do* want exactly 10, say so.
        extra.update({"best_C": best_C, "n_selected": n_selected, "pipe": pipe})
        return selected, mask.astype(bool), extra

    raise ValueError(f"Unknown selector_name: {selector_name}")


def fit_and_evaluate(
    model,
    X_train_sel: pd.DataFrame,
    y_train: np.ndarray,
    X_test_sel: pd.DataFrame,
    y_test: np.ndarray,
    label: str,
    nsamples_ci: int = 2000,
) -> Tuple[float, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit model, pick threshold on train, evaluate test metrics and confusion matrices."""
    m = clone(model)
    m.fit(X_train_sel, y_train)

    proba_train = predict_proba_1(m, X_train_sel)
    proba_test = predict_proba_1(m, X_test_sel)

    thr = float(get_optimal_threshold(y_train, proba_train))

    # Metrics with CI strings (already includes thresholded metrics internally)
    _, df_metrics = get_stats_with_ci(y_test, proba_test, label, thr, nsamples=nsamples_ci)

    # Confusion matrices on test
    y_test_pred = (np.asarray(proba_test) > thr).astype(int)
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    # Normalized per true class (rows sum to 1). Guard against division-by-zero.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Add confusion matrix values into the metrics row for easy side-by-side comparison.
    tn, fp, fn, tp = cm.ravel()
    df_metrics = df_metrics.copy()
    df_metrics["TN"] = int(tn)
    df_metrics["FP"] = int(fp)
    df_metrics["FN"] = int(fn)
    df_metrics["TP"] = int(tp)
    # normalized entries
    df_metrics["TN_rate"] = float(cm_norm[0, 0])
    df_metrics["FP_rate"] = float(cm_norm[0, 1])
    df_metrics["FN_rate"] = float(cm_norm[1, 0])
    df_metrics["TP_rate"] = float(cm_norm[1, 1])

    return thr, cm, cm_norm, df_metrics


def build_model_specs(random_state: int = 27) -> List[Tuple[str, Any]]:
    """Define the base models used in experiments."""
    specs: List[Tuple[str, Any]] = []

    specs.append((
        "logreg",
        LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
    ))

    specs.append((
        "svm_linear",
        SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    ))

    specs.append((
        "random_forest",
        RandomForestClassifier(
            min_samples_leaf=8,
            random_state=random_state,
            class_weight="balanced",
        ),
    ))

    specs.append((
        "xgb",
        xgb.XGBClassifier(
            use_label_encoder=False,
            colsample_bytree=1,
            objective="binary:logistic",
            eval_metric="logloss",
            nthread=4,
            scale_pos_weight=1,
            seed=random_state,
        ),
    ))

    return specs


def run_experiments(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_n: int = 10,
    nsamples_ci: int = 2000,
) -> Tuple[List[ExperimentResult], pd.DataFrame, pd.DataFrame]:
    """Run all configured experiments and return results plus comparison tables."""

    # 1) Preprocess (fit on train, apply to test)
    mean_std, var_selector, to_drop, X_train_p = preprocessing_train(X_train)
    X_test_p = preprocessing_test(X_test, mean_std, var_selector, to_drop)

    # 2) Define model + selector grid
    models = build_model_specs(random_state=27)

    selectors_for_all = [
        "anova_cv",          # ANOVA SelectKBest with CV-tuned k
        "rfe_no_cv_top10",   # RFE selects exactly top 10
        "rfecv_rank_top10",  # RFECV ranks with CV, then keep top 10
    ]

    # Additional selectors that are more model-appropriate (optional but useful)
    # We'll include embedded importance for tree/boosting models.
    tree_embedded_selector = "embedded_method"

    # L1 logistic CV selector (its own embedded model). We'll run it as a separate experiment.
    lasso_selector = "lasso_logregcv"

    experiments: List[Tuple[str, Any, str]] = []

    for model_name, model in models:
        # Run ANOVA + both RFE strategies for every model as requested.
        for sel in selectors_for_all:
            experiments.append((model_name, model, sel))

        # Also run embedded selection for tree/boosting models to compare.
        if model_name in {"random_forest", "xgb"}:
            experiments.append((model_name, model, tree_embedded_selector))

    # Add lasso-style embedded selection (using LogisticRegressionCV with L1)
    experiments.append(("lasso_logregcv", LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=27,
    ), lasso_selector))

    results: List[ExperimentResult] = []

    for model_name, model, selector_name in experiments:
        selected, mask, extra = select_features(
            selector_name,
            X_train_p,
            y_train,
            model,
            top_n=top_n,
        )

        X_train_sel = X_train_p.loc[:, mask]
        X_test_sel = X_test_p.loc[:, mask]

        label = f"{model_name} | {selector_name}"
        thr, cm, cm_norm, df_metrics = fit_and_evaluate(
            model,
            X_train_sel,
            y_train,
            X_test_sel,
            y_test,
            label=label,
            nsamples_ci=nsamples_ci,
        )

        results.append(ExperimentResult(
            model_name=model_name,
            selector_name=selector_name,
            selected_features=selected,
            support_mask=mask,
            optimal_threshold=thr,
            confusion_matrix=cm,
            confusion_matrix_norm=cm_norm,
            metrics_test=df_metrics,
            extra=extra,
        ))

    # 3) Comparison tables
    metrics_table = pd.concat([r.metrics_test for r in results], axis=0)

    # Feature overlap table (Jaccard similarity)
    names = [f"{r.model_name}|{r.selector_name}" for r in results]
    jacc = pd.DataFrame(index=names, columns=names, dtype=float)
    masks = [r.support_mask.astype(bool) for r in results]

    for i in range(len(results)):
        for j in range(len(results)):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            jacc.iloc[i, j] = (inter / union) if union else 0.0

    return results, metrics_table, jacc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--nsamples_ci", type=int, default=2000)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_feature_csv(args.train_csv, args.test_csv)

    results, metrics_table, jaccard = run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        top_n=args.top_n,
        nsamples_ci=args.nsamples_ci,
    )

    print("\n=== Test-set metrics (with CI strings + confusion matrices) ===")
    print(metrics_table)

    print("\n=== Feature-set overlap (Jaccard) ===")
    print(jaccard)

    metrics_table.to_csv("metrics_comparison_v2.csv")
    jaccard.to_csv("feature_overlap_jaccard_v2.csv")
    print("\nSaved: metrics_comparison_v2.csv, feature_overlap_jaccard_v2.csv")


if __name__ == "__main__":
    main()
