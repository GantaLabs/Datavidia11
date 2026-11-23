"""
notebooks/01_eda.py - Fast Exploratory Data Analysis + Covariate Shift Audit
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import load_data


def run_eda():
    print("=== Loading Raw Data for EDA ===")
    train_df, test_df, sub_df = load_data(optimize_mem=False)

    print("\n--- Shape Overview ---")
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape:  {test_df.shape}")
    if sub_df is not None:
        print(f"Sample Sub:  {sub_df.shape}")

    profile_path = Path("configs/data_profile.json")
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            prof = json.load(f)
        print("\n--- Automated Data Profile & Leakage Audit ---")
        print(f"CV Strategy       : {prof.get('cv_strategy')} (OOF coverage={prof.get('cv_oof_coverage')})")
        print(f"Temporal Info     : {prof.get('temporal')}")
        print(f"Group Info        : {prof.get('group')}")
        print(f"Adversarial Valid : {prof.get('adversarial_validation')}")
        print(f"Target Dist       : {prof.get('target_distribution')}")
        print(f"Train-Only Cols   : {prof.get('train_only_columns')}")
        print(f"Auxiliary Tables  : {prof.get('auxiliary_tables')}")

    print("\n--- Missing Values (Train) ---")
    missing_train = train_df.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    if len(missing_train) == 0:
        print("No missing values in train!")
    else:
        for col, cnt in missing_train.items():
            print(f"  {col}: {cnt} ({cnt / len(train_df) * 100:.2f}%)")

    print("\n--- Missing Values (Test) ---")
    missing_test = test_df.isnull().sum()
    missing_test = missing_test[missing_test > 0]
    if len(missing_test) == 0:
        print("No missing values in test!")
    else:
        for col, cnt in missing_test.items():
            print(f"  {col}: {cnt} ({cnt / len(test_df) * 100:.2f}%)")


if __name__ == "__main__":
    run_eda()
