"""
src/dataset.py - Tabular Data Loading & Locked CV Splits Loader
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Iterate through all columns and downcast numeric types to save RAM."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory reduced: {start_mem:.2f} MB -> {end_mem:.2f} MB ({(start_mem - end_mem) / max(start_mem, 1e-6) * 100:.1f}% savings)")
    return df


def load_data(raw_data_dir: str = "data/raw", optimize_mem: bool = True):
    raw_dir = Path(raw_data_dir)
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    sub_path = raw_dir / "sample_submission.csv"

    if not train_path.exists():
        for p in raw_dir.glob("*.csv"):
            if "train" in p.name.lower():
                train_path = p
            elif "test" in p.name.lower():
                test_path = p
            elif "sub" in p.name.lower() or "sample" in p.name.lower():
                sub_path = p

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path) if sub_path.exists() else None

    if optimize_mem:
        train_df = reduce_mem_usage(train_df)
        test_df = reduce_mem_usage(test_df)

    return train_df, test_df, sub_df


def load_cv_splits(splits_path: str = "data/processed/cv_splits.json") -> Optional[Dict]:
    p = Path(splits_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cv_fold_indices(
    train_df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    cv_type: str = "stratified",
    seed: int = 42,
    group_col: Optional[str] = None,
    splits_path: str = "data/processed/cv_splits.json"
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns (cv_strategy_used, list of (train_idx, val_idx)).
    Prefers locked splits from data/processed/cv_splits.json whenever available.
    """
    locked = load_cv_splits(splits_path)
    if locked and locked.get("folds"):
        folds = [
            (np.asarray(fd["train_idx"], dtype=int), np.asarray(fd["val_idx"], dtype=int))
            for fd in locked["folds"]
        ]
        return locked.get("cv_strategy", cv_type), folds

    if cv_type == "group" and group_col and group_col in train_df.columns:
        gkf = GroupKFold(n_splits=n_splits)
        return "group", list(gkf.split(train_df, train_df[target_col], groups=train_df[group_col]))

    if cv_type == "stratified":
        class_counts = train_df[target_col].value_counts()
        if not (class_counts < n_splits).any():
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return "stratified", list(skf.split(train_df, train_df[target_col]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return "kfold", list(kf.split(train_df))
