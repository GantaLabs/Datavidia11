"""
src/build_dataset.py - Consolidate all 9 relational tables with transit-hub joins,
sequential/ratio domain features, and locked OOF 7-class target encodings.
"""

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CV_SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "cv_splits.json"

TARGET_COL = "status_penanganan"
ID_COL = "shipment_id"

CLASSES = [
    "gagal_antar_awal",
    "klaim_hilang",
    "konfirmasi_alamat",
    "normal",
    "retur_pengirim",
    "rusak_perjalanan",
    "tertunda_operasional",
]

NOISE_COLS = [
    "skor_prioritas_kirim",
    "indeks_rute_sekunder",
    "faktor_muatan_internal",
    "varian_cuaca_rute",
    "indeks_waktu_proses",
    "skor_kualitas_cek",
]

DIM_TABLES = [
    ("customers.csv", "customer_id", "cust_"),
    ("couriers.csv", "courier_id", "cour_"),
    ("hubs.csv", "hub_id", "hub_"),
    ("service_packages.csv", "package_id", "pkg_"),
    ("promotions.csv", "promo_id", "promo_"),
]

SCAN_EVENTS = ["keluar_hub", "pickup", "selesai", "sortasi", "transit", "upaya_antar"]
SCAN_DEVICES = ["gerbang_otomatis", "manual", "pda", "scanner"]
ROUTE_MODAS = ["darat", "kereta", "laut", "udara"]
INV_TYPES = ["klaim", "langganan", "pengiriman"]
INV_STATUSES = ["gagal", "lunas", "tertunda"]
INV_METHODS = ["cod", "ewallet", "kartu", "transfer"]

TE_SPECS = [
    ("hub_id", "te_hub_", "freq_hub", 20.0),
    ("scan_last_hub_id", "te_last_hub_", "freq_last_hub", 20.0),
    ("seg_last_hub_id", "te_seg_hub_", "freq_seg_hub", 20.0),
    ("courier_id", "te_cour_", "freq_cour", 15.0),
    ("customer_id", "te_cust_", "freq_cust", 10.0),
    ("route_pair", "te_route_", "freq_route", 20.0),
]

TEMP_HELPER_COLS = ["route_pair", "scan_last_hub_id", "seg_last_hub_id"]


def _count_categories(
    df: pd.DataFrame, key: str, col: str, categories: list[str], prefix: str
) -> pd.DataFrame:
    """Compute fixed-schema per-key value counts for a categorical column."""
    counts = (
        df.groupby([key, col], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    counts.columns = [f"{prefix}{c}" for c in categories]
    return counts


def _load_hub_stats(split_dir: Path, join_key: str, out_prefix: str) -> pd.DataFrame:
    """Load hub capacity, density, and automation attributes for transit joins."""
    hubs = pd.read_csv(
        split_dir / "hubs.csv",
        usecols=["hub_id", "capacity_index", "density_index", "automation_level"],
    )
    hubs = hubs.rename(
        columns={
            "hub_id": join_key,
            "capacity_index": f"{out_prefix}capacity",
            "density_index": f"{out_prefix}density",
            "automation_level": f"{out_prefix}auto",
        }
    )
    return hubs


def _aggregate_scan_events(split_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate 1-to-many scan_events.csv with transit hub stats and sequential signals."""
    usecols = [
        ID_COL,
        "hub_id",
        "event_type",
        "gap_hours",
        "device_type",
        "scan_quality",
        "is_manual",
    ]
    scans = pd.read_csv(split_dir / "scan_events.csv", usecols=usecols)
    hubs = _load_hub_stats(split_dir, join_key="hub_id", out_prefix="_sh_")
    scans = scans.merge(hubs, on="hub_id", how="left", sort=False)

    rev_idx = scans.groupby(ID_COL, sort=False).cumcount(ascending=False)
    second_last = (
        scans.loc[rev_idx == 1, [ID_COL, "event_type", "gap_hours", "scan_quality"]]
        .rename(
            columns={
                "event_type": "scan_second_last_event",
                "gap_hours": "scan_second_last_gap_hours",
                "scan_quality": "scan_second_last_quality",
            }
        )
        .set_index(ID_COL)
    )

    grp = scans.groupby(ID_COL, sort=False)
    agg = grp.agg(
        scan_count=("event_type", "size"),
        scan_nunique_hub=("hub_id", "nunique"),
        scan_gap_hours_sum=("gap_hours", "sum"),
        scan_gap_hours_mean=("gap_hours", "mean"),
        scan_gap_hours_max=("gap_hours", "max"),
        scan_gap_hours_min=("gap_hours", "min"),
        scan_gap_hours_std=("gap_hours", "std"),
        scan_quality_mean=("scan_quality", "mean"),
        scan_quality_min=("scan_quality", "min"),
        scan_quality_max=("scan_quality", "max"),
        scan_is_manual_sum=("is_manual", "sum"),
        scan_is_manual_mean=("is_manual", "mean"),
        scan_first_event=("event_type", "first"),
        scan_last_event=("event_type", "last"),
        scan_last_hub_id=("hub_id", "last"),
        scan_last_gap_hours=("gap_hours", "last"),
        scan_last_quality=("scan_quality", "last"),
        scan_last_is_manual=("is_manual", "last"),
        scan_hub_density_mean=("_sh_density", "mean"),
        scan_hub_density_max=("_sh_density", "max"),
        scan_hub_capacity_mean=("_sh_capacity", "mean"),
        scan_hub_capacity_min=("_sh_capacity", "min"),
        scan_hub_auto_mean=("_sh_auto", "mean"),
        scan_hub_auto_min=("_sh_auto", "min"),
    )
    evt_counts = _count_categories(scans, ID_COL, "event_type", SCAN_EVENTS, "scan_evt_")
    dev_counts = _count_categories(scans, ID_COL, "device_type", SCAN_DEVICES, "scan_dev_")

    res = pd.concat([agg, second_last, evt_counts, dev_counts], axis=1)
    denom_cnt = np.maximum(res["scan_count"].values, 1)
    res["scan_upaya_antar_ratio"] = res["scan_evt_upaya_antar"].values / denom_cnt
    res["scan_selesai_ratio"] = res["scan_evt_selesai"].values / denom_cnt
    res["scan_transit_ratio"] = res["scan_evt_transit"].values / denom_cnt
    res["scan_hub_revisit_ratio"] = res["scan_count"].values / np.maximum(res["scan_nunique_hub"].values, 1)
    res["scan_gap_max_to_sum"] = res["scan_gap_hours_max"].values / np.maximum(
        res["scan_gap_hours_sum"].values, 1e-3
    )

    res = res.reset_index()
    zero_fill_cols = ["scan_count", *evt_counts.columns, *dev_counts.columns]

    del scans, hubs, second_last, grp, agg, evt_counts, dev_counts
    gc.collect()
    return res, zero_fill_cols


def _aggregate_route_segments(split_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate 1-to-many route_segments.csv with transit hub stats and speed signals."""
    usecols = [
        ID_COL,
        "urutan",
        "moda",
        "jarak_segmen_km",
        "durasi_jam",
        "segmen_hub_id",
        "biaya_segmen",
    ]
    segs = pd.read_csv(split_dir / "route_segments.csv", usecols=usecols)
    segs = segs.sort_values([ID_COL, "urutan"], kind="mergesort")
    hubs = _load_hub_stats(split_dir, join_key="segmen_hub_id", out_prefix="_rh_")
    segs = segs.merge(hubs, on="segmen_hub_id", how="left", sort=False)
    segs["_seg_speed"] = segs["jarak_segmen_km"] / np.maximum(segs["durasi_jam"], 0.1)

    grp = segs.groupby(ID_COL, sort=False)
    agg = grp.agg(
        seg_count=("urutan", "size"),
        seg_max_urutan=("urutan", "max"),
        seg_nunique_hub=("segmen_hub_id", "nunique"),
        seg_nunique_moda=("moda", "nunique"),
        seg_last_hub_id=("segmen_hub_id", "last"),
        seg_jarak_sum=("jarak_segmen_km", "sum"),
        seg_jarak_mean=("jarak_segmen_km", "mean"),
        seg_jarak_max=("jarak_segmen_km", "max"),
        seg_jarak_std=("jarak_segmen_km", "std"),
        seg_durasi_sum=("durasi_jam", "sum"),
        seg_durasi_mean=("durasi_jam", "mean"),
        seg_durasi_max=("durasi_jam", "max"),
        seg_durasi_std=("durasi_jam", "std"),
        seg_biaya_sum=("biaya_segmen", "sum"),
        seg_biaya_mean=("biaya_segmen", "mean"),
        seg_biaya_max=("biaya_segmen", "max"),
        seg_speed_mean=("_seg_speed", "mean"),
        seg_speed_min=("_seg_speed", "min"),
        seg_hub_density_mean=("_rh_density", "mean"),
        seg_hub_density_max=("_rh_density", "max"),
        seg_hub_capacity_mean=("_rh_capacity", "mean"),
        seg_hub_capacity_min=("_rh_capacity", "min"),
        seg_hub_auto_min=("_rh_auto", "min"),
    )
    moda_counts = _count_categories(segs, ID_COL, "moda", ROUTE_MODAS, "seg_moda_")

    res = pd.concat([agg, moda_counts], axis=1).reset_index()
    zero_fill_cols = ["seg_count", *moda_counts.columns]

    del segs, hubs, grp, agg, moda_counts
    gc.collect()
    return res, zero_fill_cols


def _aggregate_invoices(split_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate 1-to-many invoices.csv with temporal billing and ratio features."""
    usecols = [
        ID_COL,
        "invoice_type",
        "amount",
        "discount",
        "tax",
        "payment_method",
        "payment_status",
        "issued_month",
        "days_to_pay",
        "is_penalty",
    ]
    invs = pd.read_csv(split_dir / "invoices.csv", usecols=usecols)

    grp = invs.groupby(ID_COL, sort=False)
    agg = grp.agg(
        inv_count=("invoice_type", "size"),
        inv_amount_sum=("amount", "sum"),
        inv_amount_mean=("amount", "mean"),
        inv_amount_max=("amount", "max"),
        inv_amount_min=("amount", "min"),
        inv_discount_sum=("discount", "sum"),
        inv_discount_max=("discount", "max"),
        inv_tax_sum=("tax", "sum"),
        inv_issued_month_min=("issued_month", "min"),
        inv_issued_month_max=("issued_month", "max"),
        inv_days_to_pay_mean=("days_to_pay", "mean"),
        inv_days_to_pay_max=("days_to_pay", "max"),
        inv_is_penalty_sum=("is_penalty", "sum"),
        inv_is_penalty_max=("is_penalty", "max"),
    )
    type_counts = _count_categories(invs, ID_COL, "invoice_type", INV_TYPES, "inv_type_")
    status_counts = _count_categories(invs, ID_COL, "payment_status", INV_STATUSES, "inv_status_")
    pay_counts = _count_categories(invs, ID_COL, "payment_method", INV_METHODS, "inv_pay_")

    res = pd.concat([agg, type_counts, status_counts, pay_counts], axis=1)
    denom_cnt = np.maximum(res["inv_count"].values, 1)
    res["inv_klaim_ratio"] = res["inv_type_klaim"].values / denom_cnt
    res["inv_gagal_ratio"] = res["inv_status_gagal"].values / denom_cnt
    res["inv_tertunda_ratio"] = res["inv_status_tertunda"].values / denom_cnt

    res = res.reset_index()
    zero_fill_cols = [
        "inv_count",
        *type_counts.columns,
        *status_counts.columns,
        *pay_counts.columns,
    ]

    del invs, grp, agg, type_counts, status_counts, pay_counts
    gc.collect()
    return res, zero_fill_cols


def _merge_dimension_tables(df: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    """Left-join all 5 dimension tables onto shipments with column prefixes."""
    for filename, key_col, prefix in DIM_TABLES:
        dim = pd.read_csv(split_dir / filename)
        rename_map = {c: f"{prefix}{c}" for c in dim.columns if c != key_col}
        dim = dim.rename(columns=rename_map)
        df = df.merge(dim, on=key_col, how="left", sort=False)
        del dim
    gc.collect()
    return df


def _add_domain_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-table physical, duration, billing, and matching interaction features."""
    new_cols = {
        "ratio_volumetric_to_weight": df["berat_volumetrik"] / np.maximum(df["weight_kg"], 0.01),
        "density_kg_per_l": df["weight_kg"] / np.maximum(df["volume_l"], 0.01),
        "weight_to_pkg_max": df["weight_kg"] / np.maximum(df["pkg_max_weight_kg"], 0.1),
        "weight_to_cour_cap": df["weight_kg"] / np.maximum(df["cour_kapasitas_muatan_kg"], 0.1),
        "seg_vs_ship_distance_ratio": df["seg_jarak_sum"] / np.maximum(df["distance_km"], 0.1),
        "seg_durasi_vs_estimasi_ratio": df["seg_durasi_sum"] / np.maximum(
            df["pkg_estimasi_hari"] * 24.0, 1.0
        ),
        "scan_gap_vs_seg_durasi_ratio": df["scan_gap_hours_sum"] / np.maximum(
            df["seg_durasi_sum"], 0.1
        ),
        "total_time_hours": (
            df["order_to_ship_hours"]
            + df["scan_gap_hours_sum"].fillna(0)
            + df["seg_durasi_sum"].fillna(0)
        ),
        "inv_amount_vs_declared_ratio": df["inv_amount_sum"] / np.maximum(df["declared_value"], 1.0),
        "inv_discount_vs_amount_ratio": df["inv_discount_sum"] / np.maximum(
            df["inv_amount_sum"], 1.0
        ),
        "inv_month_diff_max": df["inv_issued_month_max"] - df["ship_month"],
        "same_origin_dest_city": (df["kota_asal"] == df["kota_tujuan"]).astype(np.int8),
        "same_cust_dest_region": (df["cust_region"] == df["dest_region"]).astype(np.int8),
        "same_ship_cour_hub": (df["hub_id"] == df["cour_hub_id"]).astype(np.int8),
        "promo_active_flag": (
            (df["ship_month"] >= df["promo_bulan_mulai"])
            & (df["ship_month"] <= df["promo_bulan_akhir"])
        ).astype(np.int8),
        "route_pair": (
            df["kota_asal"].fillna("Missing").astype(str)
            + "_"
            + df["kota_tujuan"].fillna("Missing").astype(str)
        ),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _fit_te_mapping(
    keys: pd.Series, y_bin: np.ndarray, prior: np.ndarray, m: float
) -> pd.DataFrame:
    """Compute smoothed 7-class target encoding mapping for a single training split."""
    df_tmp = pd.DataFrame(y_bin, columns=CLASSES)
    df_tmp["_key"] = keys.values
    grp = df_tmp.groupby("_key", sort=False)
    sums = grp.sum()
    cnts = grp.size().values[:, None].astype(np.float32)
    smoothed = (sums.values + m * prior[None, :]) / (cnts + m)
    return pd.DataFrame(smoothed.astype(np.float32), index=sums.index, columns=CLASSES)


def _apply_te_mapping(
    mapping: pd.DataFrame, keys: pd.Series, prior: np.ndarray
) -> np.ndarray:
    """Lookup smoothed 7-class encodings with fallback to prior for unseen keys."""
    arr = mapping.reindex(keys.values).values.astype(np.float32, copy=True)
    nan_mask = np.isnan(arr[:, 0])
    if nan_mask.any():
        arr[nan_mask] = prior
    return arr


def _fit_oof_target_encoding(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute distribution-matched OOF 7-class target encodings and frequencies."""
    with open(CV_SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    folds = [
        (np.asarray(fd["train_idx"], dtype=int), np.asarray(fd["val_idx"], dtype=int))
        for fd in splits_data["folds"]
    ]

    y_str = train_df[TARGET_COL].astype(str).values
    y_bin = np.column_stack([(y_str == c).astype(np.float32) for c in CLASSES])

    te_artifacts = {"n_folds": len(folds), "entities": {}}
    new_cols: dict[str, np.ndarray] = {}

    for col, prefix, freq_col, m in TE_SPECS:
        keys = train_df[col].fillna("__MISSING__").astype(str)
        oof_freq = np.zeros(len(train_df), dtype=np.float32)
        oof_te = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
        fold_maps = []

        for tr_idx, va_idx in folds:
            tr_keys = keys.iloc[tr_idx]
            va_keys = keys.iloc[va_idx]

            fold_freq_map = np.log1p(tr_keys.value_counts()).astype(np.float32)
            oof_freq[va_idx] = va_keys.map(fold_freq_map).fillna(0.0).values.astype(np.float32)

            fold_prior = y_bin[tr_idx].mean(axis=0).astype(np.float32)
            fold_te_map = _fit_te_mapping(tr_keys, y_bin[tr_idx], fold_prior, m)
            oof_te[va_idx] = _apply_te_mapping(fold_te_map, va_keys, fold_prior)

            fold_maps.append({
                "freq_map": fold_freq_map,
                "te_map": fold_te_map,
                "prior": fold_prior,
            })

        te_artifacts["entities"][col] = {
            "prefix": prefix,
            "freq_col": freq_col,
            "fold_maps": fold_maps,
        }
        new_cols[freq_col] = oof_freq
        for j, cls_name in enumerate(CLASSES):
            new_cols[f"{prefix}{cls_name}"] = oof_te[:, j]

    train_df = pd.concat([train_df, pd.DataFrame(new_cols, index=train_df.index)], axis=1)
    print(f"Computed locked OOF 7-class Target Encoding & Frequency for {len(TE_SPECS)} entity keys.")
    return train_df, te_artifacts


def _apply_test_target_encoding(test_df: pd.DataFrame, te_artifacts: dict) -> pd.DataFrame:
    """Apply matched per-fold encoders onto equal test slices so train/test distributions match."""
    n_folds = te_artifacts["n_folds"]
    rng = np.random.default_rng(42)
    test_fold_id = rng.integers(0, n_folds, size=len(test_df))
    fold_masks = [test_fold_id == k for k in range(n_folds)]

    new_cols: dict[str, np.ndarray] = {}
    for col, info in te_artifacts["entities"].items():
        keys = test_df[col].fillna("__MISSING__").astype(str)
        test_freq = np.zeros(len(test_df), dtype=np.float32)
        test_te = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)

        for k, mask in enumerate(fold_masks):
            f_info = info["fold_maps"][k]
            slice_keys = keys.iloc[mask]
            test_freq[mask] = (
                slice_keys.map(f_info["freq_map"]).fillna(0.0).values.astype(np.float32)
            )
            test_te[mask] = _apply_te_mapping(f_info["te_map"], slice_keys, f_info["prior"])

        new_cols[info["freq_col"]] = test_freq
        prefix = info["prefix"]
        for j, cls_name in enumerate(CLASSES):
            new_cols[f"{prefix}{cls_name}"] = test_te[:, j]

    return pd.concat([test_df, pd.DataFrame(new_cols, index=test_df.index)], axis=1)


def _finalize_frame(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Drop helper columns, downcast float64 to float32, and place target column last."""
    drop_helpers = [c for c in TEMP_HELPER_COLS if c in df.columns]
    if drop_helpers:
        df = df.drop(columns=drop_helpers)

    float64_cols = df.select_dtypes(include=[np.float64]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)

    if is_train and TARGET_COL in df.columns:
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        df = df[feature_cols + [TARGET_COL]]
    return df


def build_split(raw_split_dir: Path, is_train: bool) -> pd.DataFrame:
    """Load shipments.csv, drop noise, join 5 dimension tables, and merge aggregated event tables."""
    print(f"\n--- Building {'train' if is_train else 'test'} dataset from {raw_split_dir} ---")
    df = pd.read_csv(raw_split_dir / "shipments.csv")
    df = df.drop(columns=[c for c in NOISE_COLS if c in df.columns])
    base_rows = len(df)
    print(f"Loaded shipments.csv (after dropping {len(NOISE_COLS)} noise cols): {df.shape}")

    df = _merge_dimension_tables(df, raw_split_dir)
    assert len(df) == base_rows, f"Row count changed after dimension join: {len(df)} != {base_rows}"
    print(f"After dimension joins: {df.shape}")

    zero_fill_all: list[str] = []
    for name, agg_fn in [
        ("scan_events", _aggregate_scan_events),
        ("route_segments", _aggregate_route_segments),
        ("invoices", _aggregate_invoices),
    ]:
        agg_df, zero_cols = agg_fn(raw_split_dir)
        df = df.merge(agg_df, on=ID_COL, how="left", sort=False)
        assert len(df) == base_rows, f"Row count changed after {name} join: {len(df)} != {base_rows}"
        zero_fill_all.extend(zero_cols)
        del agg_df
        gc.collect()
        print(f"After {name} merge: {df.shape}")

    for col in zero_fill_all:
        df[col] = df[col].fillna(0).astype(np.int32)

    df = _add_domain_interactions(df)
    print(f"After cross-table domain interactions: {df.shape}")
    return df


def main():
    train_df = build_split(RAW_DIR / "train", is_train=True)
    assert len(train_df) == 1243617, f"Unexpected train rows: {len(train_df)}"

    train_df, te_artifacts = _fit_oof_target_encoding(train_df)
    train_df = _finalize_frame(train_df, is_train=True)
    train_cols = [c for c in train_df.columns if c != TARGET_COL]

    train_out = RAW_DIR / "train.csv"
    print(f"Writing {train_out} ({train_df.shape})...")
    train_df.to_csv(train_out, index=False)
    del train_df
    gc.collect()

    test_df = build_split(RAW_DIR / "test", is_train=False)
    assert len(test_df) == 362245, f"Unexpected test rows: {len(test_df)}"

    test_df = _apply_test_target_encoding(test_df, te_artifacts)
    test_df = _finalize_frame(test_df, is_train=False)
    assert train_cols == list(test_df.columns), (
        f"Column mismatch! train_only={set(train_cols) - set(test_df.columns)}, "
        f"test_only={set(test_df.columns) - set(train_cols)}"
    )

    sample_sub = pd.read_csv(RAW_DIR / "sample_submission.csv", usecols=[ID_COL])
    assert (test_df[ID_COL].values == sample_sub[ID_COL].values).all(), (
        "test.csv shipment_id order does not match sample_submission.csv!"
    )

    test_out = RAW_DIR / "test.csv"
    print(f"Writing {test_out} ({test_df.shape})...")
    test_df.to_csv(test_out, index=False)
    del test_df, te_artifacts
    gc.collect()

    print("\nSUCCESS: Upgraded train.csv and test.csv verified and saved.")


if __name__ == "__main__":
    main()
