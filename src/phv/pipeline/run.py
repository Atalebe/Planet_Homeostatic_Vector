# src/phv/pipeline/run.py
from __future__ import annotations

from typing import Any, Dict, Optional

from phv.pipeline.tier1 import run_tier1


def run_pipeline(cfg: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Backward/CLI-friendly alias for Tier 1.
    Keeps `phv run --config ...` stable even if Tier 1 implementation lives in tier1.py.
    """
    run_tier1(cfg, config_path=config_path)
