import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_cached_config = None


def load_config(path=None):
    global _cached_config
    if _cached_config is not None and path is None:
        return _cached_config
    config_path = Path(path) if path else _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        _cached_config = yaml.safe_load(f)
    return _cached_config


def reload_config():
    global _cached_config
    _cached_config = None
    return load_config()
