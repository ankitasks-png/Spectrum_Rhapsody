"""Load and validate project configuration from configs/config.yaml."""

import yaml

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_path": "data/processed/features.csv",
        "labels": {},
    },
    "audio": {
        "sample_rate": 22050,
        "top_db": 25,
    },
    "model": {
        "type": "random_forest",
        "n_estimators": 200,
        "test_size": 0.2,
        "random_state": 42,
        "save_path": "models/saved_models/baseline.pkl",
    },
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config, falling back to defaults for any missing keys.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        A config dict merged with DEFAULT_CONFIG (file values take
        precedence).
    """
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        user_config = {}

    config = _deep_merge(DEFAULT_CONFIG, user_config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
