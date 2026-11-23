"""
EXP-002_nested_te_seq_lgbm/train.py
Nested (Double) OOF 7-Class Target Encoding + Vectorized Sequential Trajectory
Features + Native LightGBM Categorical Handling on Locked 5-Fold Stratified CV.
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
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
HELPER_KEYS = [
    "hub_id",
    "scan_last_hub_id",
    "seg_last_hub_id",
    "courier_id",
    "customer_id",
    "route_pair",
    "cour_hub_id",
    "hub_goods_pair",
]
TE_SPECS = [
    ("hub_id", "nte_hub_", "rel_freq_hub", 20.0),
    ("scan_last_hub_id", "nte_last_hub_", "rel_freq_last_hub", 20.0),
    ("seg_last_hub_id", "nte_seg_hub_", "rel_freq_seg_hub", 20.0),
    ("courier_id", "nte_cour_", "rel_freq_cour", 15.0),
    ("customer_id", "nte_cust_", "rel_freq_cust", 10.0),
    ("route_pair", "nte_route_", "rel_freq_route", 20.0),
    ("cour_hub_id", "nte_cour_hub_", "rel_freq_cour_hub", 15.0),
    ("hub_goods_pair", "nte_hub_goods_", "rel_freq_hub_goods", 20.0),
]


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


def _calc_group_entropy(
    sid_codes: np.ndarray,
    cat_codes: np.ndarray,
    n_cats: int,
    n_ship: int,
    grp_len: np.ndarray,
) -> np.ndarray:
    """Compute vectorized per-shipment Shannon entropy over a categorical event column."""
    ent = np.zeros(n_ship, dtype=np.float32)
    denom = np.maximum(grp_len.astype(np.float32), 1.0)
    for k in range(n_cats):
        ck = np.bincount(sid_codes[cat_codes == k], minlength=n_ship).astype(np.float32)
        pk = ck / denom
        mask = pk > 0
        ent[mask] -= pk[mask] * np.log(pk[mask])
    return ent


def _extract_scan_features(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract vectorized sequential scan features and unique (shipment_id, hub_id) pairs."""
    scans = pd.read_csv(
        split_dir / "scan_events.csv",
        usecols=[
            ID_COL,
            "hub_id",
            "event_type",
            "gap_hours",
            "device_type",
            "scan_quality",
            "is_manual",
        ],
        dtype={
            ID_COL: "str",
            "hub_id": "str",
            "event_type": "str",
            "gap_hours": np.float32,
            "device_type": "str",
            "scan_quality": np.float32,
            "is_manual": np.float32,
        },
    )

    sid_codes, sid_uniques = pd.factorize(scans[ID_COL], sort=False)
    if (np.diff(sid_codes) < 0).any():
        order = np.argsort(sid_codes, kind="mergesort")
        scans = scans.iloc[order].reset_index(drop=True)
        sid_codes = sid_codes[order]

    n_ship = len(sid_uniques)
    n_rows = len(scans)
    is_new = np.empty(n_rows, dtype=bool)
    is_new[0] = True
    is_new[1:] = sid_codes[1:] != sid_codes[:-1]
    grp_start = np.flatnonzero(is_new)
    grp_end = np.empty(n_ship, dtype=np.int64)
    grp_end[:-1] = grp_start[1:]
    grp_end[-1] = n_rows
    grp_len = grp_end - grp_start
    last_idx = grp_end - 1

    qual = scans["scan_quality"].to_numpy(dtype=np.float32)
    gap = scans["gap_hours"].to_numpy(dtype=np.float32)
    isman = scans["is_manual"].to_numpy(dtype=np.float32)
    hub_vals = scans["hub_id"].fillna("Missing").to_numpy()
    hub_codes, _ = pd.factorize(hub_vals, sort=False)
    evt_codes, evt_uniques = pd.factorize(scans["event_type"].fillna("Missing"), sort=False)
    dev_codes, dev_uniques = pd.factorize(scans["device_type"].fillna("Missing"), sort=False)

    scan_delta_quality = (qual[last_idx] - qual[grp_start]).astype(np.float32)
    scan_delta_gap = (gap[last_idx] - gap[grp_start]).astype(np.float32)

    pos_in_grp = np.arange(n_rows, dtype=np.int64) - np.repeat(grp_start, grp_len)
    half_pt = np.repeat(grp_len // 2, grp_len)
    len_rep = np.repeat(grp_len, grp_len)
    in_first = (pos_in_grp < half_pt) & (len_rep > 1)
    in_second = (pos_in_grp >= half_pt) & (len_rep > 1)

    c1 = np.maximum(np.bincount(sid_codes[in_first], minlength=n_ship).astype(np.float32), 1.0)
    c2 = np.maximum(np.bincount(sid_codes[in_second], minlength=n_ship).astype(np.float32), 1.0)
    has2 = grp_len > 1

    scan_half_delta_quality = np.where(
        has2,
        np.bincount(sid_codes[in_second], weights=qual[in_second], minlength=n_ship) / c2
        - np.bincount(sid_codes[in_first], weights=qual[in_first], minlength=n_ship) / c1,
        0.0,
    ).astype(np.float32)
    scan_half_delta_gap = np.where(
        has2,
        np.bincount(sid_codes[in_second], weights=gap[in_second], minlength=n_ship) / c2
        - np.bincount(sid_codes[in_first], weights=gap[in_first], minlength=n_ship) / c1,
        0.0,
    ).astype(np.float32)
    scan_half_delta_is_manual = np.where(
        has2,
        np.bincount(sid_codes[in_second], weights=isman[in_second], minlength=n_ship) / c2
        - np.bincount(sid_codes[in_first], weights=isman[in_first], minlength=n_ship) / c1,
        0.0,
    ).astype(np.float32)

    gap_sum = np.bincount(sid_codes, weights=gap, minlength=n_ship).astype(np.float32)
    second_last_idx = np.maximum(last_idx - 1, grp_start)
    last2_gap = np.where(has2, gap[last_idx] + gap[second_last_idx], gap[last_idx])
    scan_last2_gap_share = (last2_gap / np.maximum(gap_sum, 1e-3)).astype(np.float32)

    same_ship_adj = sid_codes[1:] == sid_codes[:-1]
    adj_sids = sid_codes[1:]
    n_trans = np.maximum((grp_len - 1).astype(np.float32), 1.0)

    hub_sw = same_ship_adj & (hub_codes[1:] != hub_codes[:-1])
    scan_hub_switches = np.bincount(adj_sids[hub_sw], minlength=n_ship).astype(np.float32)
    scan_hub_switch_rate = (scan_hub_switches / n_trans).astype(np.float32)

    evt_map = {str(name): i for i, name in enumerate(evt_uniques)}
    u_id = evt_map.get("upaya_antar", -99)
    t_id = evt_map.get("transit", -99)
    s_id = evt_map.get("sortasi", -99)

    scan_trans_upaya_repeat_cnt = np.bincount(
        adj_sids[same_ship_adj & (evt_codes[:-1] == u_id) & (evt_codes[1:] == u_id)],
        minlength=n_ship,
    ).astype(np.float32)
    scan_trans_transit_sortasi_cnt = np.bincount(
        adj_sids[same_ship_adj & (evt_codes[:-1] == t_id) & (evt_codes[1:] == s_id)],
        minlength=n_ship,
    ).astype(np.float32)
    scan_trans_sortasi_transit_cnt = np.bincount(
        adj_sids[same_ship_adj & (evt_codes[:-1] == s_id) & (evt_codes[1:] == t_id)],
        minlength=n_ship,
    ).astype(np.float32)

    scan_event_entropy = _calc_group_entropy(sid_codes, evt_codes, len(evt_uniques), n_ship, grp_len)
    scan_device_entropy = _calc_group_entropy(sid_codes, dev_codes, len(dev_uniques), n_ship, grp_len)

    scan_df = pd.DataFrame(
        {
            ID_COL: sid_uniques.to_numpy(),
            "scan_first_hub_id": hub_vals[grp_start],
            "scan_last_hub_id": hub_vals[last_idx],
            "scan_delta_quality": scan_delta_quality,
            "scan_delta_gap": scan_delta_gap,
            "scan_half_delta_quality": scan_half_delta_quality,
            "scan_half_delta_gap": scan_half_delta_gap,
            "scan_half_delta_is_manual": scan_half_delta_is_manual,
            "scan_last2_gap_share": scan_last2_gap_share,
            "scan_hub_switches": scan_hub_switches,
            "scan_hub_switch_rate": scan_hub_switch_rate,
            "scan_event_entropy": scan_event_entropy,
            "scan_device_entropy": scan_device_entropy,
            "scan_trans_upaya_repeat_cnt": scan_trans_upaya_repeat_cnt,
            "scan_trans_upaya_repeat_rate": (scan_trans_upaya_repeat_cnt / n_trans).astype(np.float32),
            "scan_trans_transit_sortasi_cnt": scan_trans_transit_sortasi_cnt,
            "scan_trans_transit_sortasi_rate": (scan_trans_transit_sortasi_cnt / n_trans).astype(
                np.float32
            ),
            "scan_trans_sortasi_transit_cnt": scan_trans_sortasi_transit_cnt,
            "scan_trans_sortasi_transit_rate": (scan_trans_sortasi_transit_cnt / n_trans).astype(
                np.float32
            ),
        }
    )

    scan_pairs = scans[[ID_COL, "hub_id"]].drop_duplicates()
    del scans
    gc.collect()
    return scan_df, scan_pairs


def _extract_route_features(split_dir: Path, scan_pairs: pd.DataFrame) -> pd.DataFrame:
    """Extract first/last route-segment features and cross-table scan-vs-route hub overlap."""
    segs = pd.read_csv(
        split_dir / "route_segments.csv",
        usecols=[ID_COL, "urutan", "jarak_segmen_km", "durasi_jam", "segmen_hub_id", "biaya_segmen"],
        dtype={
            ID_COL: "str",
            "urutan": np.int32,
            "jarak_segmen_km": np.float32,
            "durasi_jam": np.float32,
            "segmen_hub_id": "str",
            "biaya_segmen": np.float32,
        },
    )
    segs = segs.sort_values([ID_COL, "urutan"], kind="mergesort").reset_index(drop=True)

    sid_codes, sid_uniques = pd.factorize(segs[ID_COL], sort=False)
    n_ship = len(sid_uniques)
    n_rows = len(segs)

    is_new = np.empty(n_rows, dtype=bool)
    is_new[0] = True
    is_new[1:] = sid_codes[1:] != sid_codes[:-1]
    grp_start = np.flatnonzero(is_new)
    grp_end = np.empty(n_ship, dtype=np.int64)
    grp_end[:-1] = grp_start[1:]
    grp_end[-1] = n_rows
    last_idx = grp_end - 1

    durasi = segs["durasi_jam"].to_numpy(dtype=np.float32)
    jarak = segs["jarak_segmen_km"].to_numpy(dtype=np.float32)
    biaya = segs["biaya_segmen"].to_numpy(dtype=np.float32)
    speed = (jarak / np.maximum(durasi, 0.1)).astype(np.float32)
    hub_vals = segs["segmen_hub_id"].fillna("Missing").to_numpy()

    durasi_sum = np.bincount(sid_codes, weights=durasi, minlength=n_ship).astype(np.float32)
    biaya_sum = np.bincount(sid_codes, weights=biaya, minlength=n_ship).astype(np.float32)

    seg_first_speed = speed[grp_start]
    seg_last_speed = speed[last_idx]
    seg_first_durasi = durasi[grp_start]
    seg_last_durasi = durasi[last_idx]
    seg_first_biaya = biaya[grp_start]
    seg_last_biaya = biaya[last_idx]

    route_pairs = (
        segs[[ID_COL, "segmen_hub_id"]]
        .rename(columns={"segmen_hub_id": "hub_id"})
        .drop_duplicates()
    )
    overlap = scan_pairs.merge(route_pairs, on=[ID_COL, "hub_id"], how="inner")
    overlap_cnts = overlap.groupby(ID_COL, sort=False).size()

    route_df = pd.DataFrame(
        {
            ID_COL: sid_uniques.to_numpy(),
            "seg_last_hub_id": hub_vals[last_idx],
            "seg_first_speed": seg_first_speed,
            "seg_last_speed": seg_last_speed,
            "seg_speed_last_minus_first": (seg_last_speed - seg_first_speed).astype(np.float32),
            "seg_speed_last_to_first_ratio": (
                seg_last_speed / np.maximum(seg_first_speed, 0.1)
            ).astype(np.float32),
            "seg_first_durasi": seg_first_durasi,
            "seg_last_durasi": seg_last_durasi,
            "seg_last_durasi_share": (seg_last_durasi / np.maximum(durasi_sum, 1e-3)).astype(
                np.float32
            ),
            "seg_first_biaya": seg_first_biaya,
            "seg_last_biaya": seg_last_biaya,
            "seg_last_biaya_share": (seg_last_biaya / np.maximum(biaya_sum, 1e-3)).astype(
                np.float32
            ),
            "seg_biaya_last_minus_first": (seg_last_biaya - seg_first_biaya).astype(np.float32),
        }
    )
    route_df["scan_route_hub_overlap_cnt"] = (
        route_df[ID_COL].map(overlap_cnts).fillna(0).astype(np.float32)
    )

    del segs, route_pairs, overlap, overlap_cnts
    gc.collect()
    return route_df


def _build_split_cache(
    raw_csv_path: Path,
    split_aux_dir: Path,
    out_parquet_path: Path,
    is_train: bool,
) -> None:
    """Build compact Parquet feature table for train or test split."""
    t0 = time.time()
    hdr = pd.read_csv(raw_csv_path, nrows=2000)
    drop_cols = {c for c in hdr.columns if c.startswith(("te_", "freq_"))} | NOISE_COLS
    use_cols = [c for c in hdr.columns if c not in drop_cols]

    dtype_map: dict[str, object] = {
        c: (np.float32 if pd.api.types.is_numeric_dtype(hdr[c]) else "str") for c in use_cols
    }

    chunks = list(pd.read_csv(raw_csv_path, usecols=use_cols, dtype=dtype_map, chunksize=400_000))
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    scan_df, scan_pairs = _extract_scan_features(split_aux_dir)
    route_df = _extract_route_features(split_aux_dir, scan_pairs)
    del scan_pairs
    gc.collect()

    df = df.merge(scan_df, on=ID_COL, how="left", sort=False)
    del scan_df
    df = df.merge(route_df, on=ID_COL, how="left", sort=False)
    del route_df
    gc.collect()

    ship_hr = df["ship_hour"]
    buka = df["hub_jam_buka"]
    tutup = df["hub_jam_tutup"]
    log_val = np.log1p(np.maximum(df["declared_value"], 0.0))

    new_cols = {
        "scan_route_hub_overlap_frac": (
            df["scan_route_hub_overlap_cnt"].fillna(0.0)
            / np.maximum(df["scan_nunique_hub"].fillna(1.0), 1.0)
        ).astype(np.float32),
        "route_scan_hub_overlap_frac": (
            df["scan_route_hub_overlap_cnt"].fillna(0.0)
            / np.maximum(df["seg_nunique_hub"].fillna(1.0), 1.0)
        ).astype(np.float32),
        "scan_last_eq_seg_last_hub": (
            df["scan_last_hub_id"].fillna("__na1__") == df["seg_last_hub_id"].fillna("__na2__")
        ).astype(np.float32),
        "scan_first_eq_origin_hub": (
            df["scan_first_hub_id"].fillna("__na1__") == df["hub_id"].fillna("__na2__")
        ).astype(np.float32),
        "ship_inside_hub_hours": ((ship_hr >= buka) & (ship_hr <= tutup)).astype(np.float32),
        "ship_hours_to_close": (tutup - ship_hr).astype(np.float32),
        "ship_hours_after_open": (ship_hr - buka).astype(np.float32),
        "hub_load_proxy": (
            df["hub_density_index"]
            * df["hub_capacity_index"]
            / np.maximum(df["hub_luas_gudang_m2"], 1.0)
        ).astype(np.float32),
        "fragility_x_pkg_score": (df["fragility_flag"] * df["pkg_fragile_score"]).astype(np.float32),
        "fragility_x_no_haz_cert": (
            df["fragility_flag"] * (1.0 - df["cour_sertifikasi_hazardous"])
        ).astype(np.float32),
        "new_addr_x_bad_addr_qual": (
            df["flag_alamat_baru"] * (1.0 - df["cust_address_quality"] / 5.0)
        ).astype(np.float32),
        "cod_x_not_allowed": (df["is_cod"] * (1.0 - df["pkg_cod_allowed"])).astype(np.float32),
        "ins_mismatch_x_val": (
            (df["insurance_opt"] != df["pkg_insurance_default"]).astype(np.float32) * log_val
        ).astype(np.float32),
        "ins_diff_x_val": (
            (df["insurance_opt"] - df["pkg_insurance_default"]).astype(np.float32) * log_val
        ).astype(np.float32),
        "route_pair": (
            df["kota_asal"].fillna("Missing").astype(str)
            + "_"
            + df["kota_tujuan"].fillna("Missing").astype(str)
        ),
        "hub_goods_pair": (
            df["hub_id"].fillna("Missing").astype(str)
            + "_"
            + df["goods_category"].fillna("Missing").astype(str)
        ),
    }
    df.drop(columns=["scan_first_hub_id", "scan_route_hub_overlap_cnt"], inplace=True)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    for k in HELPER_KEYS:
        df[k] = df[k].fillna("Missing").astype(str)

    if not is_train and TARGET_COL in df.columns:
        df.drop(columns=[TARGET_COL], inplace=True)

    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet_path, index=False, compression="snappy")
    print(f"Built cache {out_parquet_path.name}: {df.shape} in {time.time() - t0:.1f}s")
    del df
    gc.collect()


def ensure_cache_exists() -> tuple[Path, Path]:
    """Build cached Parquet feature tables once from data/raw if missing."""
    cache_dir = PROJECT_ROOT / "data" / "processed" / "exp002_cache"
    tr_pq = cache_dir / "train_features.parquet"
    te_pq = cache_dir / "test_features.parquet"

    if not tr_pq.exists():
        _build_split_cache(
            PROJECT_ROOT / "data" / "raw" / "train.csv",
            PROJECT_ROOT / "data" / "raw" / "train",
            tr_pq,
            is_train=True,
        )
    if not te_pq.exists():
        _build_split_cache(
            PROJECT_ROOT / "data" / "raw" / "test.csv",
            PROJECT_ROOT / "data" / "raw" / "test",
            te_pq,
            is_train=False,
        )
    return tr_pq, te_pq


def compute_nested_te_fold(
    tr_key_codes: list[np.ndarray],
    n_uniques_list: list[int],
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    num_classes: int,
    fold_seed: int,
    n_inner: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray, list[str], np.ndarray]:
    """
    Compute strictly intra-fold 4-split Nested OOF 7-class smoothed TE + relative
    frequency shares for X_tr and X_va using pre-factorized integer key codes.
    Also returns per-key full-idx_tr (te_full, rf_full) tables + prior_full for test transform.
    """
    y_tr = y[idx_tr]
    n_tr = len(idx_tr)
    n_va = len(idx_va)
    n_cols_per_key = num_classes + 1
    total_te_cols = len(TE_SPECS) * n_cols_per_key

    tr_out = np.zeros((n_tr, total_te_cols), dtype=np.float32)
    va_out = np.zeros((n_va, total_te_cols), dtype=np.float32)
    full_tables: list[tuple[np.ndarray, np.ndarray]] = []
    col_names: list[str] = []

    skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=fold_seed)
    inner_splits = list(skf_inner.split(np.zeros(n_tr, dtype=np.int8), y_tr))
    prior_full = np.bincount(y_tr, minlength=num_classes).astype(np.float64) / float(n_tr)
    va_cust_rel_freq = np.zeros(n_va, dtype=np.float32)

    col_offset = 0
    for k_idx, (key_col, te_prefix, freq_name, smooth_m) in enumerate(TE_SPECS):
        for c_idx in range(num_classes):
            col_names.append(f"{te_prefix}c{c_idx}")
        col_names.append(freq_name)

        codes_all = tr_key_codes[k_idx]
        n_u = n_uniques_list[k_idx]
        tr_codes = codes_all[idx_tr]
        va_codes = codes_all[idx_va]

        for in_tr, in_va in inner_splits:
            c_in_tr = tr_codes[in_tr]
            y_in_tr = y_tr[in_tr]
            c_in_va = tr_codes[in_va]

            prior_in = np.bincount(y_in_tr, minlength=num_classes).astype(np.float64) / float(len(in_tr))
            cnt_in = np.bincount(c_in_tr, minlength=n_u).astype(np.float64)
            flat_in = c_in_tr.astype(np.int64) * num_classes + y_in_tr
            cls_cnt_in = (
                np.bincount(flat_in, minlength=n_u * num_classes)
                .reshape(n_u, num_classes)
                .astype(np.float64)
            )

            te_tbl = ((cls_cnt_in + smooth_m * prior_in[None, :]) / (cnt_in[:, None] + smooth_m)).astype(
                np.float32
            )
            rf_tbl = (cnt_in / float(len(in_tr))).astype(np.float32)

            tr_out[in_va, col_offset : col_offset + num_classes] = te_tbl[c_in_va]
            tr_out[in_va, col_offset + num_classes] = rf_tbl[c_in_va]

        # Full idx_tr fit
        cnt_full = np.bincount(tr_codes, minlength=n_u).astype(np.float64)
        flat_full = tr_codes.astype(np.int64) * num_classes + y_tr
        cls_cnt_full = (
            np.bincount(flat_full, minlength=n_u * num_classes)
            .reshape(n_u, num_classes)
            .astype(np.float64)
        )
        te_full = ((cls_cnt_full + smooth_m * prior_full[None, :]) / (cnt_full[:, None] + smooth_m)).astype(
            np.float32
        )
        rf_full = (cnt_full / float(n_tr)).astype(np.float32)
        full_tables.append((te_full, rf_full))

        va_out[:, col_offset : col_offset + num_classes] = te_full[va_codes]
        va_out[:, col_offset + num_classes] = rf_full[va_codes]

        if key_col == "customer_id":
            va_cust_rel_freq = va_out[:, col_offset + num_classes].copy()

        col_offset += n_cols_per_key

    return tr_out, va_out, full_tables, prior_full.astype(np.float32), col_names, va_cust_rel_freq


def transform_test_te(
    te_key_codes: list[np.ndarray],
    full_tables: list[tuple[np.ndarray, np.ndarray]],
    prior_full: np.ndarray,
    num_classes: int,
    n_test: int,
) -> np.ndarray:
    """Apply outer-fold full-idx_tr TE + rel_freq tables to test integer key codes."""
    n_cols_per_key = num_classes + 1
    out = np.zeros((n_test, len(TE_SPECS) * n_cols_per_key), dtype=np.float32)
    col_offset = 0
    for k_idx, (te_full, rf_full) in enumerate(full_tables):
        codes = te_key_codes[k_idx]
        known = codes >= 0
        out[known, col_offset : col_offset + num_classes] = te_full[codes[known]]
        out[~known, col_offset : col_offset + num_classes] = prior_full
        out[known, col_offset + num_classes] = rf_full[codes[known]]
        col_offset += n_cols_per_key
    return out


def main() -> None:
    t_start = time.time()
    set_seed(SEED)

    sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
    splits_path = PROJECT_ROOT / "data" / "processed" / "cv_splits.json"
    submissions_dir = PROJECT_ROOT / "submissions"

    print("=== EXP-002: Nested TE + Sequential Trajectory + Native Categorical LightGBM ===")
    tr_pq, te_pq = ensure_cache_exists()

    t_load = time.time()
    train_df = pd.read_parquet(tr_pq)
    print(f"Loaded train Parquet in {time.time() - t_load:.2f}s | shape={train_df.shape}")

    with open(splits_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    folds = cv_data["folds"]
    cv_strategy = cv_data.get("cv_strategy", "stratified")
    print(f"Loaded locked CV splits: strategy={cv_strategy}, n_folds={len(folds)}")

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
    n_train = len(train_df)

    excluded = {ID_COL, TARGET_COL} | DRIFTING_IDS | NOISE_COLS | set(HELPER_KEYS)
    base_features = [c for c in train_df.columns if c not in excluded]
    cat_cols = [c for c in base_features if not pd.api.types.is_numeric_dtype(train_df[c])]
    num_cols = [c for c in base_features if c not in cat_cols]
    ordered_base = num_cols + cat_cols
    cat_indices = list(range(len(num_cols), len(ordered_base)))
    print(f"Base features: {len(ordered_base)} ({len(num_cols)} numeric, {len(cat_cols)} native categorical)")

    # Compact integer factorization of helper keys & nominal categoricals (avoids holding GBs of Python str)
    tr_key_codes: list[np.ndarray] = []
    key_uniques: list[pd.Index] = []
    n_uniques_list: list[int] = []
    for k in HELPER_KEYS:
        codes, uniq = pd.factorize(train_df[k].fillna("Missing").to_numpy(), sort=False)
        tr_key_codes.append(codes.astype(np.int32))
        key_uniques.append(pd.Index(uniq))
        n_uniques_list.append(len(uniq))
    train_df.drop(columns=HELPER_KEYS + [c for c in [ID_COL, "package_id", "promo_id"] if c in train_df.columns], inplace=True)

    tr_cat_codes = np.empty((n_train, len(cat_cols)), dtype=np.int32)
    cat_uniques: list[pd.Index] = []
    for j, c in enumerate(cat_cols):
        codes, uniq = pd.factorize(train_df[c].fillna("Missing").astype(str).to_numpy(), sort=True)
        tr_cat_codes[:, j] = codes.astype(np.int32)
        cat_uniques.append(pd.Index(uniq))
    train_df.drop(columns=cat_cols, inplace=True)

    X_num_tr = train_df[num_cols].to_numpy(dtype=np.float32)
    del train_df
    gc.collect()

    oof_preds = np.zeros((n_train, num_classes), dtype=np.float64)
    oof_mask = np.zeros(n_train, dtype=bool)
    oof_cust_rel_freq = np.zeros(n_train, dtype=np.float32)

    fold_scores: list[float] = []
    train_scores: list[float] = []
    fold_logloss: list[float] = []
    best_iterations: list[int] = []
    gain_importances: np.ndarray | None = None
    all_feature_names: list[str] = []

    trained_models: list[LGBMClassifier] = []
    fold_cat_masks: list[list[np.ndarray]] = []
    fold_te_artifacts: list[tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]] = []

    lgb_params = dict(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        n_estimators=52,
        learning_rate=0.195,
        num_leaves=31,
        max_bin=63,
        min_child_samples=100,
        colsample_bytree=0.30,
        subsample=0.42,
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

        te_tr, te_va, full_tables, prior_full, te_col_names, va_cust_rf = compute_nested_te_fold(
            tr_key_codes=tr_key_codes,
            n_uniques_list=n_uniques_list,
            y=y,
            idx_tr=idx_tr,
            idx_va=idx_va,
            num_classes=num_classes,
            fold_seed=SEED + fold * 100,
            n_inner=4,
        )
        oof_cust_rel_freq[idx_va] = va_cust_rf

        # Enforce strict idx_tr-only category visibility (unseen categories in idx_va mapped to -1)
        cat_tr = tr_cat_codes[idx_tr].astype(np.float32)
        per_col_seen: list[np.ndarray] = []
        for j in range(len(cat_cols)):
            n_c = len(cat_uniques[j])
            per_col_seen.append(np.bincount(tr_cat_codes[idx_tr, j], minlength=n_c) > 0)

        X_tr = np.hstack([X_num_tr[idx_tr], cat_tr, te_tr])
        y_tr, y_va = y[idx_tr], y[idx_va]
        del cat_tr, te_tr
        gc.collect()

        if not all_feature_names:
            all_feature_names = ordered_base + te_col_names
            gain_importances = np.zeros(len(all_feature_names), dtype=np.float64)

        model = LGBMClassifier(**lgb_params, random_state=SEED + fold)
        model.fit(X_tr, y_tr)

        best_iter = int(getattr(model, "best_iteration_", None) or lgb_params["n_estimators"])
        best_iterations.append(best_iter)

        tr_probs_sub = align_proba(
            model.predict_proba(X_tr[::10]), model.classes_, cls_to_col, num_classes
        )
        tr_f1 = float(
            f1_score(y_tr[::10], np.argmax(tr_probs_sub, axis=1), average="macro", zero_division=0)
        )
        del X_tr, y_tr, tr_probs_sub
        gc.collect()

        cat_va_int = tr_cat_codes[idx_va].copy()
        for j in range(len(cat_cols)):
            va_col = cat_va_int[:, j]
            va_col[~per_col_seen[j][va_col]] = -1
        X_va = np.hstack([X_num_tr[idx_va], cat_va_int.astype(np.float32), te_va])
        del cat_va_int, te_va

        va_probs = align_proba(model.predict_proba(X_va), model.classes_, cls_to_col, num_classes)
        oof_preds[idx_va] = va_probs
        oof_mask[idx_va] = True

        gain_importances += model.booster_.feature_importance(importance_type="gain") / len(folds)
        if cv_strategy != "purged_rolling_temporal":
            trained_models.append(model)
            fold_cat_masks.append(per_col_seen)
            fold_te_artifacts.append((full_tables, prior_full))

        va_f1 = float(f1_score(y_va, np.argmax(va_probs, axis=1), average="macro", zero_division=0))
        va_ll = float(log_loss(y_va, np.clip(va_probs, 1e-15, 1.0), labels=all_classes))

        train_scores.append(tr_f1)
        fold_scores.append(va_f1)
        fold_logloss.append(va_ll)
        print(
            f"Fold {fold + 1}/{len(folds)} ({time.time() - t_fold:.1f}s) | best_iter={best_iter:4d} | "
            f"val_f1_macro={va_f1:.5f} | train_f1_macro={tr_f1:.5f} | val_logloss={va_ll:.5f}"
        )

        del X_va, y_va, va_probs
        if cv_strategy == "purged_rolling_temporal":
            del model
        gc.collect()

    # Release train matrices before loading test Parquet
    if cv_strategy != "purged_rolling_temporal":
        del X_num_tr, tr_cat_codes, tr_key_codes
        gc.collect()

    t_te = time.time()
    test_df = pd.read_parquet(te_pq)
    n_test = len(test_df)
    test_ids = test_df[ID_COL].copy()

    te_key_codes: list[np.ndarray] = []
    for k_idx, k in enumerate(HELPER_KEYS):
        te_key_codes.append(
            key_uniques[k_idx].get_indexer(test_df[k].fillna("Missing").to_numpy()).astype(np.int32)
        )

    te_cat_codes = np.empty((n_test, len(cat_cols)), dtype=np.int32)
    for j, c in enumerate(cat_cols):
        te_cat_codes[:, j] = (
            cat_uniques[j]
            .get_indexer(test_df[c].fillna("Missing").astype(str).to_numpy())
            .astype(np.int32)
        )

    X_num_te = test_df[num_cols].to_numpy(dtype=np.float32)
    del test_df, key_uniques, cat_uniques
    gc.collect()
    print(f"Loaded & prepared test Parquet in {time.time() - t_te:.2f}s | n_test={n_test}")

    used_full_train_refit = False
    test_preds = np.zeros((n_test, num_classes), dtype=np.float64)
    if cv_strategy == "purged_rolling_temporal":
        used_full_train_refit = True
        full_idx = np.arange(n_train, dtype=int)
        te_full_tr, _, full_tables, prior_full, _, _ = compute_nested_te_fold(
            tr_key_codes=tr_key_codes,
            n_uniques_list=n_uniques_list,
            y=y,
            idx_tr=full_idx,
            idx_va=full_idx[:1],
            num_classes=num_classes,
            fold_seed=SEED,
            n_inner=4,
        )
        te_full_te = transform_test_te(te_key_codes, full_tables, prior_full, num_classes, n_test)
        X_full = np.hstack([X_num_tr, tr_cat_codes.astype(np.float32), te_full_tr])
        X_te_full = np.hstack([X_num_te, te_cat_codes.astype(np.float32), te_full_te])
        refit_params = dict(lgb_params)
        refit_params["n_estimators"] = max(int(np.mean(best_iterations)), 70)
        for rs in [SEED, SEED + 101, SEED + 202]:
            m_full = LGBMClassifier(**refit_params, random_state=rs)
            m_full.fit(X_full, y, categorical_feature=cat_indices)
            test_preds += (
                align_proba(m_full.predict_proba(X_te_full), m_full.classes_, cls_to_col, num_classes)
                / 3.0
            )
            del m_full
            gc.collect()
        del X_full, X_te_full, X_num_tr, tr_cat_codes, tr_key_codes
    else:
        n_num = len(num_cols)
        n_cat = len(cat_cols)
        n_te_cols = len(TE_SPECS) * (num_classes + 1)
        X_te = np.empty((n_test, n_num + n_cat + n_te_cols), dtype=np.float32)
        X_te[:, :n_num] = X_num_te
        for fold_i, model in enumerate(trained_models):
            full_tables, prior_full = fold_te_artifacts[fold_i]
            per_col_seen = fold_cat_masks[fold_i]
            X_te[:, n_num + n_cat :] = transform_test_te(
                te_key_codes, full_tables, prior_full, num_classes, n_test
            )
            cat_te_slice = X_te[:, n_num : n_num + n_cat]
            cat_te_slice[:] = te_cat_codes
            for j in range(n_cat):
                col_c = te_cat_codes[:, j]
                known = col_c >= 0
                unseen_in_fold = np.zeros(n_test, dtype=bool)
                unseen_in_fold[known] = ~per_col_seen[j][col_c[known]]
                cat_te_slice[unseen_in_fold, j] = -1.0
            te_probs = align_proba(model.predict_proba(X_te), model.classes_, cls_to_col, num_classes)
            test_preds += te_probs / len(trained_models)
            del te_probs
        del X_te, trained_models, fold_cat_masks, fold_te_artifacts
        gc.collect()

    del X_num_te, te_cat_codes, te_key_codes
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

    # Diagnostic: Unseen (cust_rel_freq == 0) vs Repeat customers
    rf_cov = oof_cust_rel_freq[oof_mask]
    unseen_m = rf_cov == 0.0
    repeat_m = ~unseen_m
    unseen_diag: dict[str, float | int] = {}
    if unseen_m.any() and repeat_m.any():
        unseen_f1 = float(f1_score(y_cov[unseen_m], oof_pred_cls[unseen_m], average="macro", zero_division=0))
        repeat_f1 = float(f1_score(y_cov[repeat_m], oof_pred_cls[repeat_m], average="macro", zero_division=0))
        unseen_ll = float(log_loss(y_cov[unseen_m], np.clip(oof_cov[unseen_m], 1e-15, 1.0), labels=all_classes))
        repeat_ll = float(log_loss(y_cov[repeat_m], np.clip(oof_cov[repeat_m], 1e-15, 1.0), labels=all_classes))
        unseen_diag = {
            "unseen_n": int(unseen_m.sum()),
            "unseen_f1_macro": unseen_f1,
            "unseen_logloss": unseen_ll,
            "repeat_n": int(repeat_m.sum()),
            "repeat_f1_macro": repeat_f1,
            "repeat_logloss": repeat_ll,
        }
        print("\nCustomer Fold-Overlap Diagnostic:")
        print(
            f"  Unseen Customers (n={unseen_m.sum():,}, {unseen_m.mean():.1%}): "
            f"f1_macro={unseen_f1:.5f} | logloss={unseen_ll:.5f}"
        )
        print(
            f"  Repeat Customers (n={repeat_m.sum():,}, {repeat_m.mean():.1%}): "
            f"f1_macro={repeat_f1:.5f} | logloss={repeat_ll:.5f}"
        )

    # Top-25 feature importances (gain)
    imp_df = pd.DataFrame({"feature": all_feature_names, "gain": gain_importances}).sort_values(
        "gain", ascending=False
    )
    print("\nTop-25 Feature Importances (Mean Fold Gain):")
    for rank, row in enumerate(imp_df.head(25).itertuples(index=False), start=1):
        print(f"  {rank:2d}. {row.feature:<35s} {row.gain:14.2f}")

    # OOF vs Test distribution & TVD sanity check
    test_pred_cls = np.argmax(test_preds, axis=1)
    oof_prop = pd.Series(oof_pred_cls).value_counts(normalize=True).reindex(all_classes, fill_value=0.0)
    tst_prop = pd.Series(test_pred_cls).value_counts(normalize=True).reindex(all_classes, fill_value=0.0)
    oof_mean_p = oof_cov.mean(axis=0)
    tst_mean_p = test_preds.mean(axis=0)
    tvd = float(0.5 * np.abs(oof_prop.to_numpy() - tst_prop.to_numpy()).sum())

    print(f"\nOOF vs Test Predicted Class Proportions (TVD={tvd:.5f}):")
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
        "n_features": len(all_feature_names),
        "oof_vs_test_tvd": tvd,
        "customer_overlap_diag": unseen_diag,
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
