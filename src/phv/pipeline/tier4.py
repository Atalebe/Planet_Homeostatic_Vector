import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd


def _utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def _coerce_num(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _pick_best_mag_row(row: pd.Series, candidates: List[Tuple[str, str]]) -> Tuple[Optional[str], float]:
    best_val = np.nan
    best_band = None
    for col, band in candidates:
        if col in row.index:
            v = _safe_float(row[col])
            if np.isfinite(v):
                if (best_band is None) or (v < best_val):
                    best_val = v
                    best_band = band
    return best_band, best_val


def _minmax_norm(s: pd.Series, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return pd.Series(np.nan, index=s.index)

    xmin = np.nanmin(x[finite])
    xmax = np.nanmax(x[finite])
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        out = pd.Series(np.nan, index=s.index)
        out.loc[finite] = 0.5
        return out

    y = (x - xmin) / (xmax - xmin)
    if invert:
        y = 1.0 - y
    return y


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_tier4(cfg: Dict[str, Any], config_path: Optional[str] = None) -> None:
    run_cfg = cfg.get("run", {})
    inputs = cfg.get("inputs", {})
    tier4_cfg = cfg.get("tier4", {})

    run_id = run_cfg.get("run_id", "PHV-TIER4-v1")
    out_dir = run_cfg.get("out_dir", "data/derived/tier4")
    _ensure_dir(out_dir)

    print(f"[TIER4] Run: {run_id}")

    tier3_path = inputs.get("tier3_enriched_csv")
    if not tier3_path:
        raise KeyError("configs: inputs.tier3_enriched_csv is required")
    if not os.path.exists(tier3_path):
        raise FileNotFoundError(f"Missing tier3_enriched_csv: {tier3_path}")

    print(f"[TIER4] Reading: {tier3_path}")
    df = pd.read_csv(tier3_path)

    require_prev = bool(tier4_cfg.get("require_previous_pass", True))
    n0 = len(df)
    if require_prev and "tier2_pass" in df.columns:
        df = df[pd.to_numeric(df["tier2_pass"], errors="coerce").fillna(0).astype(int) == 1].copy()
    print(f"[TIER4] require_previous_pass: {1 if require_prev else 0} -> {len(df)}")

    # ---- optional merge with NASA PS ----
    ps_csv = inputs.get("ps_csv")
    merge_log = {"did_merge": False, "added_cols": [], "ps_csv": ps_csv}

    if ps_csv and os.path.exists(ps_csv):
        ps_keep = tier4_cfg.get("ps_keep_columns", [])
        if not isinstance(ps_keep, list):
            ps_keep = []

        left_key = tier4_cfg.get("merge_key_left", "planet_id")
        right_key = tier4_cfg.get("merge_key_right", "pl_name")
        how = tier4_cfg.get("merge_how", "left")

        ps = pd.read_csv(ps_csv, comment="#", low_memory=False)

        if right_key not in ps.columns:
            print(f"[TIER4] PS merge skipped: right key '{right_key}' not in ps_csv columns")
            merge_log["skip_reason"] = f"missing_right_key:{right_key}"
        elif left_key not in df.columns:
            print(f"[TIER4] PS merge skipped: left key '{left_key}' not in tier3 columns")
            merge_log["skip_reason"] = f"missing_left_key:{left_key}"
        else:
            keep_cols = [c for c in ps_keep if c in ps.columns]
            if right_key not in keep_cols:
                keep_cols = [right_key] + keep_cols

            ps = ps[keep_cols].copy()
            df[left_key] = df[left_key].astype(str).str.strip()
            ps[right_key] = ps[right_key].astype(str).str.strip()

            # important: the PS file can contain multiple rows per planet
            # keep last (row order is fine because archive rows already have rowupdate, but we don't rely on it here)
            ps = ps.drop_duplicates(subset=[right_key], keep="last")

            before = set(df.columns)
            df = df.merge(ps, left_on=left_key, right_on=right_key, how=how, suffixes=("", "_ps"))
            added = sorted(list(set(df.columns) - before))
            merge_log = {"did_merge": True, "added_cols": added, "ps_csv": ps_csv, "left_key": left_key, "right_key": right_key, "how": how}
            print(f"[TIER4] PS merge OK. Added cols: {added}")
    else:
        if ps_csv:
            print(f"[TIER4] ps_csv not found: {ps_csv} (skipping merge)")
            merge_log["skip_reason"] = "ps_csv_not_found"
        else:
            merge_log["skip_reason"] = "ps_csv_not_provided"

    _coerce_num(df, [
        "pl_rade", "st_rad", "pl_bmasse", "st_age", "pl_eqt", "pl_insol",
        "pl_orbeccen", "rho_bulk_g_cm3", "g_surface_m_s2", "vesc_rel_earth",
        "Ripeness_T3", "Ripeness_T2"
    ])

    if "Ripeness_T3" not in df.columns:
        df["Ripeness_T3"] = pd.to_numeric(df.get("Ripeness_T2", np.nan), errors="coerce")

    # ---- transit depth proxy ----
    RE_OVER_RS = 1.0 / 109.1
    rp_rs = (df["pl_rade"] * RE_OVER_RS) / df["st_rad"]
    df["tran_depth_proxy"] = np.where(np.isfinite(rp_rs), rp_rs ** 2, np.nan)

    # ---- best magnitude selection ----
    mag_candidates = [
        ("sy_gaiamag", "G"),
        ("sy_tmag", "T"),
        ("sy_kmag", "K"),
        ("sy_hmag", "H"),
        ("sy_jmag", "J"),
        ("sy_vmag", "V"),
        ("sy_gmag", "g"),
        ("sy_rmag", "r"),
        ("sy_imag", "i"),
        ("sy_zmag", "z"),
        ("sy_w1mag", "W1"),
        ("sy_w2mag", "W2"),
    ]

    # CRITICAL: make these non-float so string assignment is legal
    df["best_mag_band"] = pd.Series([None] * len(df), index=df.index, dtype="object")
    df["best_mag"] = np.nan

    for idx in df.index:
        band, val = _pick_best_mag_row(df.loc[idx], mag_candidates)
        if band is not None and np.isfinite(val):
            df.at[idx, "best_mag_band"] = str(band)
            df.at[idx, "best_mag"] = float(val)

    df["brightness_proxy"] = df["best_mag"]

    # ---- follow-up score ----
    depth_norm = _minmax_norm(df["tran_depth_proxy"], invert=False)
    bright_norm = _minmax_norm(df["brightness_proxy"], invert=True)

    df["followup_depth_norm"] = depth_norm
    df["followup_brightness_norm"] = bright_norm

    w_depth = float(tier4_cfg.get("w_depth", 0.5))
    w_bright = float(tier4_cfg.get("w_brightness", 0.5))

    depth_score = df["followup_depth_norm"].where(np.isfinite(df["tran_depth_proxy"]), np.nan)
    bright_score = df["followup_brightness_norm"].where(np.isfinite(df["brightness_proxy"]), np.nan)

    d_part = depth_score.fillna(0.0)
    b_part = bright_score.fillna(0.0)

    df["followup_score"] = (w_depth * d_part + w_bright * b_part) / (w_depth + w_bright)
    df["followup_score"] = df["followup_score"].clip(0.0, 1.0)

    # ---- heuristic flags ----
    rho_min = float(tier4_cfg.get("rho_min_g_cm3", 3.0))
    df["flag_density_rocky"] = np.where(
        np.isfinite(df.get("rho_bulk_g_cm3", np.nan)) & (df["rho_bulk_g_cm3"] >= rho_min),
        True,
        False,
    )

    vesc_min = float(tier4_cfg.get("vesc_min_rel_earth", 1.0))
    df["flag_retention_ok"] = np.where(
        np.isfinite(df.get("vesc_rel_earth", np.nan)) & (df["vesc_rel_earth"] >= vesc_min),
        True,
        False,
    )

    teq_min = float(tier4_cfg.get("Teq_min_K", 180.0))
    teq_max = float(tier4_cfg.get("Teq_max_K", 300.0))
    df["flag_teq_in_band"] = np.where(
        np.isfinite(df.get("pl_eqt", np.nan)) & (df["pl_eqt"] >= teq_min) & (df["pl_eqt"] <= teq_max),
        True,
        False,
    )

    # ---- Ripeness_T4 ----
    w_followup = float(tier4_cfg.get("w_followup", 0.5))
    max_factor = float(tier4_cfg.get("followup_factor_max", 1.5))

    df["followup_factor_T4"] = (1.0 + w_followup * df["followup_score"]).clip(1.0, max_factor)

    base = pd.to_numeric(df["Ripeness_T3"], errors="coerce").fillna(0.0)
    df["Ripeness_T4"] = base * df["followup_factor_T4"]

    df = df.sort_values("Ripeness_T4", ascending=False).reset_index(drop=True)
    df["rank_tier4"] = np.arange(1, len(df) + 1)

    # ---- outputs ----
    enriched_csv = os.path.join(out_dir, "tier4_enriched.csv")
    scorecard_csv = os.path.join(out_dir, "tier4_scorecard.csv")
    summary_path = os.path.join(out_dir, "summary.json")
    meta_path = os.path.join(out_dir, "run_meta.json")

    df.to_csv(enriched_csv, index=False)

    score_cols = [
        "rank_tier4", "planet_id", "hostname",
        "best_mag_band", "best_mag",
        "tran_depth_proxy", "followup_score",
        "pl_rade", "pl_bmasse", "st_age", "pl_insol", "pl_eqt", "pl_orbeccen",
        "Phi_p",
        "rho_bulk_g_cm3", "g_surface_m_s2", "vesc_rel_earth",
        "Ripeness_T3", "Ripeness_T4",
        "flag_density_rocky", "flag_retention_ok", "flag_teq_in_band",
    ]
    score_cols = [c for c in score_cols if c in df.columns]
    df[score_cols].to_csv(scorecard_csv, index=False)

    availability = {
        "sy_dist": int(pd.to_numeric(df.get("sy_dist", pd.Series([], dtype=float)), errors="coerce").notna().sum()) if "sy_dist" in df.columns else 0,
        "best_mag": int(pd.to_numeric(df["best_mag"], errors="coerce").notna().sum()) if "best_mag" in df.columns else 0,
        "tran_depth_proxy": int(pd.to_numeric(df["tran_depth_proxy"], errors="coerce").notna().sum()) if "tran_depth_proxy" in df.columns else 0,
    }

    summary = {
        "run_id": run_id,
        "tier3_enriched_csv": tier3_path,
        "n_input": int(n0),
        "n_output": int(len(df)),
        "require_previous_pass": require_prev,
        "ps_merge": merge_log,
        "computed": {
            "transit_depth_proxy": True,
            "best_mag": True,
            "followup_score": True,
            "heuristic_flags": True,
        },
        "availability_counts": availability,
        "outputs": {
            "tier4_enriched_csv": enriched_csv,
            "tier4_scorecard_csv": scorecard_csv,
        },
        "timestamp_utc": _utc_now_str(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "run_id": run_id,
        "config_path": config_path,
        "config_sha256": _sha256_file(config_path) if config_path and os.path.exists(config_path) else None,
        "tier3_enriched_csv": tier3_path,
        "ps_csv": inputs.get("ps_csv"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[TIER4] Wrote: {enriched_csv}")
    print(f"[TIER4] Wrote: {scorecard_csv}")
    print(f"[TIER4] Wrote: {summary_path}")
    print(f"[TIER4] Wrote: {meta_path}")
    print("[TIER4] Done.")
