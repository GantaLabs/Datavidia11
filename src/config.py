"""
src/config.py - Project Configuration Loader
"""
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class CompetitionConfig:
    competition_name: str
    modality: str
    task_type: str
    target_column: str
    id_column: str
    metric_name: str
    higher_is_better: bool = True
    cv_strategy: str = "stratified"
    seed: int = 42
    n_splits: int = 5
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    submissions_dir: str = "submissions"

def load_competition_config(config_path: str = "configs/competition.yaml") -> CompetitionConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    metric_name = data.get("metric_name", "roc_auc")
    higher_is_better = data.get("higher_is_better", metric_name.lower() not in {"rmse", "mae", "rmsle", "log_loss", "logloss", "mse", "mape"})
    return CompetitionConfig(
        competition_name=data.get("competition_name", "competition"),
        modality=data.get("modality", "tabular"),
        task_type=data.get("task_type", "classification"),
        target_column=data.get("target_column", "target"),
        id_column=data.get("id_column", "id"),
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        cv_strategy=data.get("cv_strategy", "stratified"),
        seed=data.get("seed", 42),
        n_splits=data.get("n_splits", 5),
        raw_data_dir=data.get("raw_data_dir", "data/raw"),
        processed_data_dir=data.get("processed_data_dir", "data/processed"),
        submissions_dir=data.get("submissions_dir", "submissions"),
    )
