"""
EXP-001_baseline_calibrated_lgbm/train.py
Unweighted Multiclass Log-Loss LightGBM Baseline on Locked 5-Fold Stratified CV.
Optimized for low peak RAM and fast execution within the 900s budget.
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

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


def load_csv_compact(
    csv_path: Path,
    usecols: list[str],
    dtype_map: dict[str, object],
    chunksize: int = 400_000,
) -> pd.DataFrame:
    """Load CSV in chunks with compact dtypes to minimize peak memory usage."""
    chunks = list(
        pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype=dtype_map,
            chunksize=chunksize,
        )
    )
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def main() -> None:
    t_start = time.time()
    set_seed(SEED)

    train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
    sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
    splits_path = PROJECT_ROOT / "data" / "processed" / "cv_splits.json"
    submissions_dir = PROJECT_ROOT / "submissions"

    print("=== EXP-001: Calibrated Log-Loss LightGBM Baseline ===")

    # Inspect schema from tiny sample before full read
    tr_hdr = pd.read_csv(train_path, nrows=200)
    te_hdr = pd.read_csv(test_path, nrows=200)

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
    for col in te_hdr.columns:
        if col.lower().endswith("_id"):
            excluded.add(col)

    features = [c for c in te_hdr.columns if c not in excluded and c in tr_hdr.columns]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(te_hdr[c])]
    num_cols = [c for c in features if c not in cat_cols]
    print(
        f"Selected {len(features)} features "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical; "
        f"excluded {len(te_hdr.columns) - len(features)} columns)."
    )

    with open(splits_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    folds = cv_data["folds"]
    cv_strategy = cv_data.get("cv_strategy", "stratified")
    print(f"Loaded locked CV splits: strategy={cv_strategy}, n_folds={len(folds)}")

    # Build unified train+test category vocabularies by reading only cat_cols from test first
    test_cat_vocabs: dict[str, set[str]] = {c: set() for c in cat_cols}
    if cat_cols:
        te_cat_dtype = {c: "category" for c in cat_cols}
        te_cats_df = load_csv_compact(test_path, usecols=cat_cols, dtype_map=te_cat_dtype)
        for c in cat_cols:
            vals = te_cats_df[c].astype(object).where(te_cats_df[c].notna(), "Missing").astype(str)
            test_cat_vocabs[c] = set(vals.unique())
        del te_cats_df
        gc.collect()

    # Load train features + target in compact dtypes
    tr_dtype: dict[str, object] = {c: np.float32 for c in num_cols}
    for c in cat_cols:
        tr_dtype[c] = "category"
    tr_dtype[TARGET_COL] = "object"

    train_df = load_csv_compact(
        train_path,
        usecols=features + [TARGET_COL],
        dtype_map=tr_dtype,
    )
    n_train = len(train_df)
    print(f"Loaded train: ({n_train}, {len(features) + 1}) in {time.time() - t_start:.1f}s")

    raw_target = train_df.pop(TARGET_COL)
    if not pd.api.types.is_numeric_dtype(raw_target):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(raw_target.astype(str))
        class_names = list(label_encoder.classes_)
    else:
        label_encoder = None
        y = raw_target.astype(int).to_numpy()
        class_names = [str(c) for c in np.sort(np.unique(y))]
    del raw_target

    all_classes = np.arange(len(class_names), dtype=int)
    num_classes = len(all_classes)
    cls_to_col = {int(c): i for i, c in enumerate(all_classes)}
    print(f"Classes ({num_classes}): {class_names}")

    # Encode categorical columns deterministically into float32 codes using unified train+test vocab
    cat_mappings: dict[str, dict[str, int]] = {}
    for c in cat_cols:
        tr_s = train_df[c].astype(object).where(train_df[c].notna(), "Missing").astype(str)
        cats = sorted(set(tr_s.unique()) | test_cat_vocabs[c])
        mapping = {cat: idx for idx, cat in enumerate(cats)}
        cat_mappings[c] = mapping
        train_df[c] = tr_s.map(mapping).fillna(-1).astype(np.float32)

    X = train_df[features].to_numpy(dtype=np.float32)
    del train_df, test_cat_vocabs
    gc.collect()

    oof_preds = np.zeros((n_train, num_classes), dtype=np.float64)
    oof_mask = np.zeros(n_train, dtype=bool)

    fold_scores: list[float] = []
    train_scores: list[float] = []
    fold_logloss: list[float] = []
    best_iterations: list[int] = []
    gain_importances = np.zeros(len(features), dtype=np.float64)
    trained_models: list[LGBMClassifier] = []

    lgb_params = dict(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        n_estimators=55,
        learning_rate=0.16,
        num_leaves=23,
        max_bin=63,
        min_child_samples=150,
        colsample_bytree=0.35,
        subsample=0.4,
        subsample_freq=1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        force_col_wise=True,
        n_jobs=N_JOBS,
        verbosity=-1,
    )

    for fold, fd in enumerate(folds):
        t_fold = time.time()
        idx_tr = np.asarray(fd["train_idx"], dtype=int)
        idx_va = np.asarray(fd["val_idx"], dtype=int)

        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_va], y[idx_va]

        model = LGBMClassifier(**lgb_params, random_state=SEED + fold)
        model.fit(X_tr, y_tr)

        best_iter = int(getattr(model, "best_iteration_", None) or lgb_params["n_estimators"])
        best_iterations.append(best_iter)

        va_probs = align_proba(model.predict_proba(X_va), model.classes_, cls_to_col, num_classes)
        # Compute unbiased train score on deterministic 25% stride to avoid excess predict overhead
        tr_probs_sub = align_proba(
            model.predict_proba(X_tr[::4]), model.classes_, cls_to_col, num_classes
        )

        oof_preds[idx_va] = va_probs
        oof_mask[idx_va] = True

        gain_importances += model.booster_.feature_importance(importance_type="gain") / len(folds)
        if cv_strategy != "purged_rolling_temporal":
            trained_models.append(model)

        tr_f1 = float(
            f1_score(y_tr[::4], np.argmax(tr_probs_sub, axis=1), average="macro", zero_division=0)
        )
        va_f1 = float(f1_score(y_va, np.argmax(va_probs, axis=1), average="macro", zero_division=0))
        va_ll = float(log_loss(y_va, np.clip(va_probs, 1e-15, 1.0), labels=all_classes))

        train_scores.append(tr_f1)
        fold_scores.append(va_f1)
        fold_logloss.append(va_ll)
        print(
            f"Fold {fold + 1}/{len(folds)} ({time.time() - t_fold:.1f}s) | "
            f"val_f1_macro={va_f1:.5f} | train_f1_macro={tr_f1:.5f} | val_logloss={va_ll:.5f}"
        )

        del X_tr, y_tr, X_va, y_va, va_probs, tr_probs_sub
        if cv_strategy == "purged_rolling_temporal":
            del model
        gc.collect()

    # Release full train matrix before loading test set
    if cv_strategy != "purged_rolling_temporal":
        del X
        gc.collect()

    # Load test set now with minimal memory footprint
    t_test_load = time.time()
    te_dtype: dict[str, object] = {c: np.float32 for c in num_cols}
    for c in cat_cols:
        te_dtype[c] = "category"
    te_dtype[ID_COL] = "object"

    test_df = load_csv_compact(
        test_path,
        usecols=features + [ID_COL],
        dtype_map=te_dtype,
    )
    n_test = len(test_df)
    test_ids = test_df.pop(ID_COL).copy()
    for c in cat_cols:
        te_s = test_df[c].astype(object).where(test_df[c].notna(), "Missing").astype(str)
        test_df[c] = te_s.map(cat_mappings[c]).fillna(-1).astype(np.float32)

    X_test = test_df[features].to_numpy(dtype=np.float32)
    del test_df
    gc.collect()
    print(f"Loaded test: ({n_test}, {len(features)}) in {time.time() - t_test_load:.1f}s")

    used_full_train_refit = False
    test_preds = np.zeros((n_test, num_classes), dtype=np.float64)
    if cv_strategy == "purged_rolling_temporal":
        used_full_train_refit = True
        refit_seeds = [SEED, SEED + 101, SEED + 202]
        for rs in refit_seeds:
            m_full = LGBMClassifier(**lgb_params, random_state=rs)
            m_full.fit(X, y)
            te_probs = align_proba(
                m_full.predict_proba(X_test), m_full.classes_, cls_to_col, num_classes
            )
            test_preds += te_probs / len(refit_seeds)
            del m_full, te_probs
            gc.collect()
        del X
        gc.collect()
    else:
        for model in trained_models:
            te_probs = align_proba(
                model.predict_proba(X_test), model.classes_, cls_to_col, num_classes
            )
            test_preds += te_probs / len(trained_models)
        del trained_models
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
        "runtime_seconds": float(time.time() - t_start),
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
    sub_df = pd.read_csv(sub_path) if sub_path.exists() else None
    if sub_df is not None and ID_COL in sub_df.columns and len(sub_df) == n_test:
        submission = sub_df[[ID_COL]].copy()
        submission[TARGET_COL] = final_labels
    else:
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_labels})
    submission.to_csv(sub_out_path, index=False)

    print(f"\nSaved artifacts to {EXP_DIR}")
    print(f"Saved submission to {sub_out_path} (shape={submission.shape})")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
