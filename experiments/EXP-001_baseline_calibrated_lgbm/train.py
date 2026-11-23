"""
EXP-001_baseline_calibrated_lgbm/train.py
Unweighted Multiclass Log-Loss LightGBM Baseline on Locked 5-Fold Stratified CV.
"""

import gc
import json
import os
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
if not (PROJECT_ROOT / "data" / "raw" / "train.csv").exists():
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_data

SEED = 42
N_JOBS = 4
TARGET_COL = "status_penanganan"
ID_COL = "shipment_id"
DRIFTING_IDS = {
    "customer_id",
    "courier_id",
    "hub_id",
    "package_id",
    "promo_id",
    "cour_hub_id",
}
NOISE_COLS = {
    "skor_prioritas_kirim",
    "indeks_rute_sekunder",
    "faktor_muatan_internal",
    "varian_cuaca_rute",
    "indeks_waktu_proses",
    "skor_kualitas_cek",
}


def set_seed(seed: int = SEED) -> None:
    """Set deterministic random seeds across Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def align_proba(
    probs_raw: np.ndarray,
    model_classes: np.ndarray,
    cls_to_col: dict[int, int],
    num_classes: int,
) -> np.ndarray:
    """Align predict_proba output columns into a fixed (N, num_classes) matrix."""
    aligned = np.zeros((probs_raw.shape[0], num_classes), dtype=np.float64)
    for src_j, cls_val in enumerate(model_classes):
        cls_int = int(cls_val)
        if cls_int in cls_to_col:
            aligned[:, cls_to_col[cls_int]] = probs_raw[:, src_j]
    return aligned


def main() -> None:
    set_seed(SEED)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    splits_path = PROJECT_ROOT / "data" / "processed" / "cv_splits.json"
    submissions_dir = PROJECT_ROOT / "submissions"

    print("=== EXP-001: Calibrated Log-Loss LightGBM Baseline ===")
    train_df, test_df, sub_df = load_data(raw_data_dir=str(raw_dir), optimize_mem=True)
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    with open(splits_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    folds = cv_data["folds"]
    cv_strategy = cv_data.get("cv_strategy", "stratified")
    print(f"Loaded locked CV splits: strategy={cv_strategy}, n_folds={len(folds)}")

    excluded = {
        ID_COL,
        TARGET_COL,
        "tanggal",
        "date",
        "fold",
        "match_id",
        "game_id",
        "session_id",
        "group_id",
    } | DRIFTING_IDS | NOISE_COLS

    for col in test_df.columns:
        if col.lower().endswith("_id"):
            excluded.add(col)

    features = [c for c in test_df.columns if c not in excluded and c in train_df.columns]
    print(f"Selected {len(features)} features (excluded {len(test_df.columns) - len(features)} columns).")

    raw_target = train_df[TARGET_COL]
    if not pd.api.types.is_numeric_dtype(raw_target):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(raw_target.astype(str))
        class_names = list(label_encoder.classes_)
    else:
        label_encoder = None
        y = raw_target.astype(int).to_numpy()
        class_names = [str(c) for c in np.sort(np.unique(y))]

    all_classes = np.arange(len(class_names), dtype=int)
    num_classes = len(all_classes)
    cls_to_col = {int(c): i for i, c in enumerate(all_classes)}
    print(f"Classes ({num_classes}): {class_names}")

    test_ids = test_df[ID_COL].copy()
    n_train = len(train_df)
    n_test = len(test_df)

    # Keep only feature columns in memory and release raw DataFrames
    X = train_df[features]
    X_test = test_df[features]
    del train_df, test_df
    gc.collect()

    X = X.copy()
    X_test = X_test.copy()

    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"Casting {len(cat_cols)} categorical columns with unified train+test categories...")
    for col in cat_cols:
        tr_s = X[col].fillna("Missing").astype(str)
        te_s = X_test[col].fillna("Missing").astype(str)
        cats = sorted(set(tr_s.unique()) | set(te_s.unique()))
        X[col] = pd.Categorical(tr_s, categories=cats)
        X_test[col] = pd.Categorical(te_s, categories=cats)

    oof_preds = np.zeros((n_train, num_classes), dtype=np.float64)
    test_preds = np.zeros((n_test, num_classes), dtype=np.float64)
    oof_mask = np.zeros(n_train, dtype=bool)

    fold_scores: list[float] = []
    train_scores: list[float] = []
    fold_logloss: list[float] = []
    best_iterations: list[int] = []
    gain_importances = np.zeros(len(features), dtype=np.float64)

    lgb_params = dict(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_jobs=N_JOBS,
        verbosity=-1,
    )

    for fold, fd in enumerate(folds):
        idx_tr = np.asarray(fd["train_idx"], dtype=int)
        idx_va = np.asarray(fd["val_idx"], dtype=int)

        X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
        X_va, y_va = X.iloc[idx_va], y[idx_va]

        model = LGBMClassifier(**lgb_params, random_state=SEED + fold)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=250),
            ],
        )

        best_iter = int(model.best_iteration_ or lgb_params["n_estimators"])
        best_iterations.append(best_iter)

        tr_probs = align_proba(model.predict_proba(X_tr), model.classes_, cls_to_col, num_classes)
        va_probs = align_proba(model.predict_proba(X_va), model.classes_, cls_to_col, num_classes)

        oof_preds[idx_va] = va_probs
        oof_mask[idx_va] = True

        if cv_strategy != "purged_rolling_temporal":
            te_probs = align_proba(model.predict_proba(X_test), model.classes_, cls_to_col, num_classes)
            test_preds += te_probs / len(folds)

        gain_importances += model.booster_.feature_importance(importance_type="gain") / len(folds)

        tr_f1 = float(f1_score(y_tr, np.argmax(tr_probs, axis=1), average="macro", zero_division=0))
        va_f1 = float(f1_score(y_va, np.argmax(va_probs, axis=1), average="macro", zero_division=0))
        va_ll = float(log_loss(y_va, np.clip(va_probs, 1e-15, 1.0), labels=all_classes))

        train_scores.append(tr_f1)
        fold_scores.append(va_f1)
        fold_logloss.append(va_ll)
        print(
            f"Fold {fold + 1}/{len(folds)} | best_iter={best_iter:4d} | "
            f"val_f1_macro={va_f1:.5f} | train_f1_macro={tr_f1:.5f} | val_logloss={va_ll:.5f}"
        )

        del model, X_tr, y_tr, X_va, y_va, tr_probs, va_probs
        gc.collect()

    used_full_train_refit = False
    if cv_strategy == "purged_rolling_temporal":
        used_full_train_refit = True
        refit_seeds = [SEED, SEED + 101, SEED + 202]
        refit_n_estimators = max(int(np.mean(best_iterations)), 100)
        refit_params = dict(lgb_params)
        refit_params["n_estimators"] = refit_n_estimators
        test_preds = np.zeros((n_test, num_classes), dtype=np.float64)
        for rs in refit_seeds:
            m_full = LGBMClassifier(**refit_params, random_state=rs)
            m_full.fit(X, y)
            te_probs = align_proba(m_full.predict_proba(X_test), m_full.classes_, cls_to_col, num_classes)
            test_preds += te_probs / len(refit_seeds)
            del m_full, te_probs
            gc.collect()

    y_cov = y[oof_mask]
    oof_cov = oof_preds[oof_mask]
    oof_pred_cls = np.argmax(oof_cov, axis=1)

    overall_f1 = float(f1_score(y_cov, oof_pred_cls, average="macro", zero_division=0))
    overall_ll = float(log_loss(y_cov, np.clip(oof_cov, 1e-15, 1.0), labels=all_classes))
    per_class_f1_arr = f1_score(y_cov, oof_pred_cls, labels=all_classes, average=None, zero_division=0)
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(num_classes)}

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    train_mean = float(np.mean(train_scores))

    print("\n============================================================")
    print(f"Overall OOF f1_macro (raw argmax) : {overall_f1:.5f} (coverage={oof_mask.mean():.1%})")
    print(f"Overall OOF multi_logloss         : {overall_ll:.5f}")
    print(f"CV Fold f1_macro Mean +/- Std     : {cv_mean:.5f} +/- {cv_std:.5f}")
    print(f"Train Fold f1_macro Mean          : {train_mean:.5f}")
    print("Per-Class OOF F1 (raw argmax):")
    for cname, f1_val in per_class_f1.items():
        print(f"  {cname:<22s}: {f1_val:.5f}")

    # Top-25 feature importances (gain)
    imp_df = pd.DataFrame({"feature": features, "gain": gain_importances}).sort_values(
        "gain", ascending=False
    )
    print("\nTop-25 Feature Importances (Mean Fold Gain):")
    for rank, row in enumerate(imp_df.head(25).itertuples(index=False), start=1):
        print(f"  {rank:2d}. {row.feature:<35s} {row.gain:14.2f}")

    # OOF vs Test distribution & probability sanity check
    test_pred_cls = np.argmax(test_preds, axis=1)
    oof_prop = pd.Series(oof_pred_cls).value_counts(normalize=True).reindex(all_classes, fill_value=0.0)
    tst_prop = pd.Series(test_pred_cls).value_counts(normalize=True).reindex(all_classes, fill_value=0.0)
    oof_mean_p = oof_cov.mean(axis=0)
    tst_mean_p = test_preds.mean(axis=0)

    print("\nOOF vs Test Predicted Class Proportions & Mean Probabilities:")
    print(f"  {'Class':<22s} | {'OOF Prop':>9s} | {'Test Prop':>9s} | {'OOF MeanP':>9s} | {'Test MeanP':>10s}")
    for i, cname in enumerate(class_names):
        print(
            f"  {cname:<22s} | {oof_prop.iloc[i]:9.4f} | {tst_prop.iloc[i]:9.4f} | "
            f"{oof_mean_p[i]:9.4f} | {tst_mean_p[i]:10.4f}"
        )

    # Save required experiment artifacts
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EXP_DIR / "oof_preds.npy", oof_preds)
    np.save(EXP_DIR / "test_preds.npy", test_preds)
    np.save(EXP_DIR / "oof_mask.npy", oof_mask)

    run_meta = {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "train_mean": train_mean,
        "fold_scores": fold_scores,
        "used_full_train_refit": used_full_train_refit,
        "overall_oof_f1_macro": overall_f1,
        "overall_oof_multi_logloss": overall_ll,
        "fold_logloss": fold_logloss,
        "per_class_f1": per_class_f1,
        "best_iterations": best_iterations,
        "classes": class_names,
        "n_features": len(features),
        "top25_gain_features": [
            {"feature": str(r.feature), "gain": float(r.gain)}
            for r in imp_df.head(25).itertuples(index=False)
        ],
    }
    with open(EXP_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Save validated submission CSV matching sample_submission.csv schema
    submissions_dir.mkdir(parents=True, exist_ok=True)
    sub_out_path = submissions_dir / f"{EXP_DIR.name}.csv"
    final_labels = (
        label_encoder.inverse_transform(test_pred_cls)
        if label_encoder is not None
        else test_pred_cls
    )
    if sub_df is not None and ID_COL in sub_df.columns and len(sub_df) == n_test:
        submission = sub_df[[ID_COL]].copy()
        submission[TARGET_COL] = final_labels
    else:
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv(sub_out_path, index=False)

    print(f"\nSaved artifacts to {EXP_DIR}")
    print(f"Saved submission to {sub_out_path} (shape={submission.shape})")


if __name__ == "__main__":
    main()
