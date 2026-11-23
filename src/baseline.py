"""
src/baseline.py - Leak-Free Tabular LightGBM Baseline (Locked CV + Probability Output)
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor

from src.dataset import load_data, get_cv_fold_indices
from src.metrics import compute_metric

PROB_METRICS = {"roc_auc", "auc", "pr_auc", "average_precision", "log_loss", "logloss"}


def load_config():
    cfg_path = Path("configs/competition.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_baseline():
    config = load_config()
    target_col = config.get("target_column", "target")
    id_col = config.get("id_column", "id")
    task_type = config.get("task_type", "multiclass_classification")
    metric_name = config.get("metric_name", "f1_macro")
    seed = config.get("seed", 42)
    n_splits = config.get("n_splits", 5)
    cv_strategy_cfg = config.get("cv_strategy", "stratified")
    drifting_warn = set(config.get("drifting_features_warning", []))
    time_col = config.get("time_column")

    print("=== Starting Leak-Free LightGBM Baseline ===")
    print(f"Target: {target_col} | ID: {id_col} | Task: {task_type} | Metric: {metric_name}")

    train_df, test_df, sub_df = load_data()
    print(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")

    if target_col not in train_df.columns:
        possible = [c for c in train_df.columns if c in ["category", "kategori", "target", "label"]]
        target_col = possible[0] if possible else train_df.columns[-1]

    excluded = {id_col, target_col, "tanggal", "date", "fold"}
    if time_col:
        excluded.add(time_col)
    group_col_cfg = config.get("group_column")
    if group_col_cfg and not str(group_col_cfg).startswith("__id_prefix__:"):
        excluded.add(group_col_cfg)
    for col in test_df.columns:
        cl = col.lower()
        if cl in {"match_id", "game_id", "session_id", "group_id"} or cl.endswith("_id"):
            excluded.add(col)
    if drifting_warn:
        print(f"Dropping drifting monotonic/ID features: {sorted(drifting_warn)}")
        excluded |= drifting_warn

    features = [c for c in test_df.columns if c not in excluded and c in train_df.columns]
    print(f"Features used ({len(features)}): {features}")

    is_classification = "class" in task_type.lower()
    label_encoder = None
    y = train_df[target_col].copy()

    if is_classification and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))
        all_classes = np.arange(len(label_encoder.classes_))
        num_classes = len(all_classes)
        print(f"Encoded {num_classes} classes: {list(label_encoder.classes_)}")
    elif is_classification:
        y = y.astype(int).values
        all_classes = np.sort(np.unique(y))
        num_classes = int(len(all_classes))
    else:
        y = y.astype(float).values
        all_classes = None
        num_classes = 1
    cls_to_col = {c: i for i, c in enumerate(all_classes)} if all_classes is not None else {}

    X = train_df[features].copy()
    X_test = test_df[features].copy()

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_cols:
        tr_s = X[col].fillna("Missing").astype(str)
        te_s = X_test[col].fillna("Missing").astype(str)
        cats = sorted(list(set(tr_s.unique()) | set(te_s.unique())))
        X[col] = pd.Categorical(tr_s, categories=cats)
        X_test[col] = pd.Categorical(te_s, categories=cats)

    cv_used, folds = get_cv_fold_indices(
        train_df=train_df,
        target_col=target_col,
        n_splits=n_splits,
        cv_type=cv_strategy_cfg,
        seed=seed,
        group_col=config.get("group_column"),
    )
    print(f"Using Locked CV Strategy: {cv_used} ({len(folds)} folds)")

    if is_classification and num_classes > 2:
        oof_preds = np.zeros((len(train_df), num_classes), dtype=float)
        test_probs = np.zeros((len(test_df), num_classes), dtype=float)
    else:
        oof_preds = np.zeros(len(train_df), dtype=float)
        test_probs = np.zeros(len(test_df), dtype=float)

    oof_mask = np.zeros(len(train_df), dtype=bool)
    fold_scores = []

    for fold, (idx_tr, idx_va) in enumerate(folds):
        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        if is_classification:
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                min_child_samples=30,
                colsample_bytree=0.8,
                subsample=0.8,
                subsample_freq=1,
                reg_lambda=3.0,
                random_state=seed + fold,
                verbosity=-1,
                n_jobs=4,
            )
            model.fit(X_tr, y_tr)
            va_probs_raw = model.predict_proba(X_va)
            te_probs_raw = model.predict_proba(X_test)
            if num_classes > 2:
                va_probs = np.zeros((len(idx_va), num_classes), dtype=float)
                te_probs = np.zeros((len(X_test), num_classes), dtype=float)
                for src_j, cls_val in enumerate(model.classes_):
                    if cls_val in cls_to_col:
                        dst_j = cls_to_col[cls_val]
                        va_probs[:, dst_j] = va_probs_raw[:, src_j]
                        te_probs[:, dst_j] = te_probs_raw[:, src_j]
                oof_preds[idx_va] = va_probs
                test_probs += te_probs / len(folds)
                val_eval = va_probs
            else:
                pos_col = list(model.classes_).index(all_classes[-1]) if all_classes[-1] in model.classes_ else -1
                va_pos = va_probs_raw[:, pos_col] if pos_col >= 0 else np.zeros(len(idx_va), dtype=float)
                te_pos = te_probs_raw[:, pos_col] if pos_col >= 0 else np.zeros(len(X_test), dtype=float)
                oof_preds[idx_va] = va_pos
                test_probs += te_pos / len(folds)
                val_eval = va_pos
        else:
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                min_child_samples=30,
                colsample_bytree=0.8,
                subsample=0.8,
                subsample_freq=1,
                reg_lambda=3.0,
                random_state=seed + fold,
                verbosity=-1,
                n_jobs=4,
            )
            model.fit(X_tr, y_tr)
            val_eval = model.predict(X_va)
            oof_preds[idx_va] = val_eval
            test_probs += model.predict(X_test) / len(folds)

        oof_mask[idx_va] = True
        score = compute_metric(y_va, val_eval, metric_name=metric_name)
        fold_scores.append(score)
        print(f"Fold {fold+1}/{len(folds)} | Score ({metric_name}): {score:.5f}")

    # For purged_rolling_temporal splits, retrain a 3-seed full-train ensemble on 100% of data
    # so test predictions do NOT rely on stale expanding historical fold models.
    if cv_used == "purged_rolling_temporal":
        refit_seeds = [seed, seed + 101, seed + 202]
        if is_classification and num_classes > 2:
            test_probs = np.zeros((len(X_test), num_classes), dtype=float)
        else:
            test_probs = np.zeros(len(X_test), dtype=float)
        for rs in refit_seeds:
            if is_classification:
                m_full = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, min_child_samples=30,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                    reg_lambda=3.0, random_state=rs, verbosity=-1, n_jobs=4,
                )
                m_full.fit(X, y)
                te_raw = m_full.predict_proba(X_test)
                if num_classes > 2:
                    te_aligned = np.zeros((len(X_test), num_classes), dtype=float)
                    for src_j, cls_val in enumerate(m_full.classes_):
                        if cls_val in cls_to_col:
                            te_aligned[:, cls_to_col[cls_val]] = te_raw[:, src_j]
                    test_probs += te_aligned / len(refit_seeds)
                else:
                    pos_col = list(m_full.classes_).index(all_classes[-1]) if all_classes[-1] in m_full.classes_ else -1
                    te_pos = te_raw[:, pos_col] if pos_col >= 0 else np.zeros(len(X_test), dtype=float)
                    test_probs += te_pos / len(refit_seeds)
            else:
                m_full = LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, min_child_samples=30,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                    reg_lambda=3.0, random_state=rs, verbosity=-1, n_jobs=4,
                )
                m_full.fit(X, y)
                test_probs += m_full.predict(X_test) / len(refit_seeds)
        print(f"[Temporal Refit] Trained 3-seed full-train model on 100% of train rows ({len(X)}) for test inference.")

    overall_score = compute_metric(y[oof_mask], oof_preds[oof_mask], metric_name=metric_name)
    print("\n==========================================")
    print(f"Overall OOF Score ({metric_name}): {overall_score:.5f} (coverage={oof_mask.mean():.1%})")
    print(f"Fold Mean: {np.mean(fold_scores):.5f} | Fold Std: {np.std(fold_scores):.5f}")
    print("==========================================")

    # Distribution Sanity Check (OOF vs Test)
    metric_norm = metric_name.lower().replace("-", "_").replace(" ", "_")
    is_prob_export = is_classification and (metric_norm in PROB_METRICS) and (num_classes == 2 or test_probs.ndim == 1)

    if is_classification:
        oof_cls = np.argmax(oof_preds[oof_mask], axis=1) if num_classes > 2 else (oof_preds[oof_mask] >= 0.5).astype(int)
        tst_cls = np.argmax(test_probs, axis=1) if num_classes > 2 else (test_probs >= 0.5).astype(int)
        oof_dist = pd.Series(oof_cls).value_counts(normalize=True).round(4).to_dict()
        tst_dist = pd.Series(tst_cls).value_counts(normalize=True).round(4).to_dict()
        print(f"OOF Pred Class Proportions : {oof_dist}")
        print(f"Test Pred Class Proportions: {tst_dist}")
        if is_prob_export:
            final_test_labels = test_probs
        elif label_encoder is not None:
            final_test_labels = label_encoder.inverse_transform(tst_cls.astype(int))
        else:
            final_test_labels = tst_cls
    else:
        print(f"OOF Pred Mean/Std : {oof_preds[oof_mask].mean():.4f} +/- {oof_preds[oof_mask].std():.4f}")
        print(f"Test Pred Mean/Std: {test_probs.mean():.4f} +/- {test_probs.std():.4f}")
        if config.get("postprocess_round_int", False):
            final_test_labels = np.round(np.clip(test_probs, 0, None)).astype(int)
        else:
            final_test_labels = test_probs

    Path("submissions").mkdir(parents=True, exist_ok=True)
    sub_path = Path("submissions/baseline_sub.csv")
    if sub_df is not None and id_col in sub_df.columns:
        submission = sub_df.copy()
        submission[target_col] = final_test_labels
    else:
        submission = pd.DataFrame({id_col: test_df[id_col], target_col: final_test_labels})

    if "opp_goals" in submission.columns and target_col == "team_goals" and len(final_test_labels) % 2 == 0:
        opp_vals = np.zeros_like(final_test_labels)
        opp_vals[0::2] = final_test_labels[1::2]
        opp_vals[1::2] = final_test_labels[0::2]
        submission["opp_goals"] = opp_vals

    submission.to_csv(sub_path, index=False)
    print(f"Saved baseline submission to: {sub_path} (Rows: {len(submission)})")


if __name__ == "__main__":
    run_baseline()
