"""Shared JSON configuration loader for dynamic tracker modules."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config() -> SimpleNamespace:
    config_path = Path(
        os.environ.get("DYNAMIC_TRACKER_CONFIG", Path(__file__).with_name("config.json"))
    )
    with config_path.open("r", encoding="utf-8") as f:
        return _to_namespace(json.load(f))


cfg = load_config()
