# src/phv/pipeline/tier5.py
from __future__ import annotations

import json
import math
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _to_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def _sigma_from_err(err1, err2) -> float:
    """
    NASA archive convention: err1 is +, err2 is - (often negative).
    We take sigma ~ mean(|err1|, |err2|) if either exists.
    """
    e1 = _to_float(err1)
    e2 = _to_float(err2)
    vals = []
    if np.isfinite(e1):
        vals.append(abs(e1))
    if np.isfinite(e2):
        vals.append(abs(e2))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _mc_sample_normal(rng: np.random.Generator, mu: float, sigma: float, n: int,
                      lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    if (not np.isfinite(mu)) or (not np.isfinite(sigma)) or sigma <= 0:
        # Degenerate: return constant samples
        out = np.full(n, mu, dtype=float)
    else:
        out = rng.normal(mu, sigma, size=n).astype(float)

    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out


@dataclass
class Tier5Config:
    run_id: str
    tier4_enriched_csv: str
    ps_csv: Optional[str]
    require_previous_pass: bool
    require_rocky: bool
    rocky_rmax_rearth: float

    n_mc: int
    seed: int
    age_scale_gyr: float

    # Tier2 gates (copied for sampling)
    S_min: float
    S_max: float
    Teq_min_K: float
    Teq_max_K: float
    e_max: float
    require_age: bool

    # Tier3 gates
    rho_min_g_cm3: float
    rho_max_g_cm3: float
    vesc_min_rel_earth: float

    # Tier4 scoring knobs
    depth_ref: float
    mag_ref_min: float
    mag_ref_max: float
    alpha_factor: float

    # Uncertainty strategy
    use_ps_errors: bool
    use_floors_when_missing: bool

    # Floors (used only when err1/err2 missing)
    floor_frac_rade: float
    floor_frac_bmasse: float
    floor_frac_teq: float
    floor_frac_insol: float
    floor_abs_ecc: float
    floor_abs_age_gyr: float

    # PS row selection behavior
    ps_select_method: str  # "closest_to_tier4" or "median"


def _load_cfg(cfg: dict) -> Tier5Config:
    t5 = cfg.get("tier5", {})
    t2 = cfg.get("tier2", {})
    t3 = cfg.get("tier3", {})
    t4 = cfg.get("tier4", {})

    # Defaults chosen to be conservative and reviewer-friendly
    return Tier5Config(
        run_id=t5.get("run_id", "PHV-TIER5-v1"),
        tier4_enriched_csv=t5.get("tier4_enriched_csv", "data/derived/tier4/tier4_enriched.csv"),
        ps_csv=t5.get("ps_csv", cfg.get("paths", {}).get("ps_csv")),
        require_previous_pass=bool(t5.get("require_previous_pass", True)),
        require_rocky=bool(t5.get("require_rocky", True)),
        rocky_rmax_rearth=float(t5.get("rocky_rmax_rearth", 1.8)),
        n_mc=int(t5.get("n_mc", 20000)),
        seed=int(t5.get("seed", 12345)),
        age_scale_gyr=float(t5.get("age_scale_gyr", 4.5)),

        S_min=float(t2.get("S_min", 0.2)),
        S_max=float(t2.get("S_max", 2.0)),
        Teq_min_K=float(t2.get("Teq_min_K", 180.0)),
        Teq_max_K=float(t2.get("Teq_max_K", 300.0)),
        e_max=float(t2.get("e_max", 0.2)),
        require_age=bool(t2.get("require_age", True)),

        rho_min_g_cm3=float(t3.get("rho_min_g_cm3", 3.0)),
        rho_max_g_cm3=float(t3.get("rho_max_g_cm3", 10.0)),
        vesc_min_rel_earth=float(t3.get("vesc_min_rel_earth", 1.0)),

        depth_ref=float(t4.get("depth_ref", 0.005)),
        mag_ref_min=float(t4.get("mag_ref_min", 7.0)),
        mag_ref_max=float(t4.get("mag_ref_max", 13.0)),
        alpha_factor=float(t4.get("alpha_factor", 0.5)),

        use_ps_errors=bool(t5.get("use_ps_errors", True)),
        use_floors_when_missing=bool(t5.get("use_floors_when_missing", True)),

        floor_frac_rade=float(t5.get("floor_frac_rade", 0.03)),     # 3%
        floor_frac_bmasse=float(t5.get("floor_frac_bmasse", 0.10)), # 10%
        floor_frac_teq=float(t5.get("floor_frac_teq", 0.02)),       # 2%
        floor_frac_insol=float(t5.get("floor_frac_insol", 0.10)),   # 10%
        floor_abs_ecc=float(t5.get("floor_abs_ecc", 0.02)),         # abs ecc
        floor_abs_age_gyr=float(t5.get("floor_abs_age_gyr", 0.5)),  # 0.5 Gyr

        ps_select_method=str(t5.get("ps_select_method", "closest_to_tier4")),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_best_ps_row(ps_sub: pd.DataFrame, tier4_row: pd.Series) -> pd.Series:
    """
    Choose the PS row that best matches Tier4 nominal values.
    This avoids catalog duplicates randomly shifting your Monte Carlo center.
    """
    # Features to compare (only if present)
    keys = ["pl_rade", "pl_bmasse", "pl_eqt", "pl_insol", "pl_orbeccen"]
    scales = {
        "pl_rade": 0.2,      # R_earth scale
        "pl_bmasse": 2.0,    # M_earth scale
        "pl_eqt": 30.0,      # Kelvin scale
        "pl_insol": 0.5,     # S_earth scale
        "pl_orbeccen": 0.1,  # eccentricity
    }

    best_idx = None
    best_score = None

    for idx, r in ps_sub.iterrows():
        # completeness bonus: count non-null among keys
        comp = 0
        score = 0.0
        for k in keys:
            tv = _to_float(tier4_row.get(k))
            pv = _to_float(r.get(k))
            if np.isfinite(pv):
                comp += 1
            if np.isfinite(tv) and np.isfinite(pv):
                score += abs(pv - tv) / scales[k]

        # Prefer rows that match well AND have more fields filled.
        # Penalize missingness softly, but enough to break ties.
        score = score + (len(keys) - comp) * 2.0

        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx

    return ps_sub.loc[best_idx] if best_idx is not None else ps_sub.iloc[0]


def _aggregate_ps_median(ps_sub: pd.DataFrame) -> pd.Series:
    """
    Alternative: robust median aggregation across duplicates.
    """
    out = {}
    cols = list(ps_sub.columns)
    for c in cols:
        if c == "pl_name":
            out[c] = str(ps_sub[c].iloc[0])
            continue
        v = pd.to_numeric(ps_sub[c], errors="coerce")
        out[c] = float(np.nanmedian(v)) if np.isfinite(np.nanmedian(v)) else np.nan
    return pd.Series(out)


def _compute_rho_g_surface_vesc(rade_rearth: np.ndarray, bmasse_mearth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bulk density [g/cm^3], surface gravity [m/s^2], and escape speed relative to Earth.
    Using simple scalings:
      rho ~ M/R^3
      g ~ M/R^2
      vesc ~ sqrt(M/R)
    with Earth: rho=5.51 g/cm3, g=9.80665 m/s2, vesc_rel=1.
    """
    R = np.maximum(rade_rearth, 1e-9)
    M = np.maximum(bmasse_mearth, 1e-9)

    rho = 5.51 * (M / (R ** 3))
    g = 9.80665 * (M / (R ** 2))
    vesc_rel = np.sqrt(M / R)
    return rho, g, vesc_rel


def _followup_score(depth_proxy: np.ndarray,
                    best_mag: np.ndarray,
                    depth_ref: float,
                    mag_ref_min: float,
                    mag_ref_max: float,
                    alpha_factor: float) -> np.ndarray:
    """
    Follow-up score in [0,1] from transit depth proxy and brightness.
    - depth proxy: (Rp/Rs)^2 if possible
    - brightness: a magnitude where smaller is brighter
    """
    # Depth normalization: map depth_ref -> 0.5-ish by logistic-ish scaling
    d = np.maximum(depth_proxy, 0.0)
    depth_norm = np.clip(d / max(depth_ref, 1e-9), 0.0, 2.0)
    depth_norm = np.clip(depth_norm / 2.0, 0.0, 1.0)

    # Brightness normalization: map [mag_ref_min, mag_ref_max] -> [1,0]
    m = best_mag.copy()
    m = np.where(np.isfinite(m), m, np.nan)
    bright_norm = (mag_ref_max - m) / max(mag_ref_max - mag_ref_min, 1e-9)
    bright_norm = np.clip(bright_norm, 0.0, 1.0)

    # Combine
    score = (1.0 - alpha_factor) * depth_norm + alpha_factor * bright_norm
    return np.clip(score, 0.0, 1.0)


def run_tier5(cfg: dict, config_path: Optional[str] = None) -> None:
    t5 = _load_cfg(cfg)
    rng = np.random.default_rng(t5.seed)

    out_dir = "data/derived/tier5"
    _ensure_dir(out_dir)

    print(f"[TIER5] Run: {t5.run_id}")
    print(f"[TIER5] Reading: {t5.tier4_enriched_csv}")

    df = pd.read_csv(t5.tier4_enriched_csv, low_memory=False)

    # optional: require previous pass
    if t5.require_previous_pass:
        for col in ["tier2_pass", "rank_tier4", "rank_tier3"]:
            if col in df.columns:
                df = df[df[col].notna()]
        if "tier2_pass" in df.columns:
            df = df[df["tier2_pass"].astype(int) == 1]
        if "flag_density_rocky" in df.columns:
            df = df[df["flag_density_rocky"].astype(bool)]
    print(f"[TIER5] require_previous_pass: {len(pd.read_csv(t5.tier4_enriched_csv))} -> {len(df)}")

    # ensure numeric core cols
    core_num = ["pl_rade","pl_bmasse","pl_orbeccen","pl_eqt","pl_insol","st_age","Phi_p","C_phi_pow","best_mag","tran_depth_proxy","st_rad"]
    for c in core_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Merge PS uncertainties (and optionally values) per planet
    ps_merge_report = None
    ps_err_counts = {}
    ps_selected_rows = 0
    ps_method = t5.ps_select_method

    if t5.use_ps_errors and t5.ps_csv and os.path.exists(t5.ps_csv):
        ps = pd.read_csv(t5.ps_csv, comment="#", low_memory=False)

        # we only need rows relevant to df planet_id
        if "planet_id" not in df.columns:
            raise RuntimeError("tier4_enriched.csv must include planet_id for PS merge.")
        ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
        needed = set(df["planet_id"].astype(str).str.strip().unique().tolist())
        ps_sub = ps[ps["pl_name"].isin(needed)].copy()

        # columns we want from PS
        want = [
            "pl_name",
            "pl_rade","pl_radeerr1","pl_radeerr2",
            "pl_bmasse","pl_bmasseerr1","pl_bmasseerr2",
            "pl_orbeccen","pl_orbeccenerr1","pl_orbeccenerr2",
            "pl_eqt","pl_eqterr1","pl_eqterr2",
            "pl_insol","pl_insolerr1","pl_insolerr2",
            "st_age","st_ageerr1","st_ageerr2",
        ]
        want = [c for c in want if c in ps_sub.columns]
        ps_sub = ps_sub[want].copy()

        # numeric coercion
        for c in want:
            if c != "pl_name":
                ps_sub[c] = pd.to_numeric(ps_sub[c], errors="coerce")

        # build a per-planet table by selecting or aggregating
        rows = []
        for pid in df["planet_id"].astype(str).str.strip().tolist():
            sub = ps_sub[ps_sub["pl_name"].astype(str).str.strip().eq(pid)]
            if len(sub) == 0:
                continue
            if ps_method == "median":
                pick = _aggregate_ps_median(sub)
            else:
                # closest_to_tier4
                trow = df[df["planet_id"].astype(str).str.strip().eq(pid)].iloc[0]
                pick = _pick_best_ps_row(sub, trow)
            rows.append(pick)
            ps_selected_rows += 1

        ps_best = pd.DataFrame(rows) if rows else pd.DataFrame(columns=want)

        # attach to df
        df = df.merge(ps_best, how="left", left_on="planet_id", right_on="pl_name", suffixes=("", "_ps"))

        # count availability
        for c in ["pl_radeerr1","pl_radeerr2","pl_bmasseerr1","pl_bmasseerr2","pl_eqterr1","pl_eqterr2","pl_insolerr1","pl_insolerr2","pl_orbeccenerr1","pl_orbeccenerr2","st_ageerr1","st_ageerr2"]:
            if c in df.columns:
                ps_err_counts[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            else:
                ps_err_counts[c] = 0

        ps_merge_report = {
            "did_merge": True,
            "ps_csv": t5.ps_csv,
            "selection_method": ps_method,
            "n_ps_rows_for_planets": int(len(ps_sub)),
            "n_planets_selected": int(ps_selected_rows),
            "err_nonnull_counts": ps_err_counts,
        }
        print(f"[TIER5] PS merge OK. selection={ps_method}. planets_selected={ps_selected_rows}")
    else:
        ps_merge_report = {"did_merge": False, "reason": "ps_csv missing or use_ps_errors=false"}

    # Monte Carlo per planet
    out_rows = []

    for _, row in df.iterrows():
        pid = str(row.get("planet_id", ""))
        host = str(row.get("hostname", ""))
        best_mag_band = str(row.get("best_mag_band", "")) if "best_mag_band" in row else ""
        best_mag_nom = _to_float(row.get("best_mag")) if "best_mag" in row else float("nan")

        # Nominal centers come from Tier4 pipeline columns
        rade0 = _to_float(row.get("pl_rade"))
        m0 = _to_float(row.get("pl_bmasse"))
        e0 = _to_float(row.get("pl_orbeccen"))
        teq0 = _to_float(row.get("pl_eqt"))
        insol0 = _to_float(row.get("pl_insol"))
        age0 = _to_float(row.get("st_age"))

        phi = _to_float(row.get("Phi_p"))
        cphi = _to_float(row.get("C_phi_pow", row.get("C_phi", float("nan"))))

        # Pull PS sigmas (if present)
        s_rade = _sigma_from_err(row.get("pl_radeerr1"), row.get("pl_radeerr2"))
        s_m = _sigma_from_err(row.get("pl_bmasseerr1"), row.get("pl_bmasseerr2"))
        s_teq = _sigma_from_err(row.get("pl_eqterr1"), row.get("pl_eqterr2"))
        s_ins = _sigma_from_err(row.get("pl_insolerr1"), row.get("pl_insolerr2"))
        s_e = _sigma_from_err(row.get("pl_orbeccenerr1"), row.get("pl_orbeccenerr2"))
        s_age = _sigma_from_err(row.get("st_ageerr1"), row.get("st_ageerr2"))

        # Apply floors if missing
        floors_used = {}
        if t5.use_floors_when_missing:
            if not np.isfinite(s_rade) and np.isfinite(rade0):
                s_rade = abs(rade0) * t5.floor_frac_rade
                floors_used["pl_rade"] = s_rade
            if not np.isfinite(s_m) and np.isfinite(m0):
                s_m = abs(m0) * t5.floor_frac_bmasse
                floors_used["pl_bmasse"] = s_m
            if not np.isfinite(s_teq) and np.isfinite(teq0):
                s_teq = abs(teq0) * t5.floor_frac_teq
                floors_used["pl_eqt"] = s_teq
            if not np.isfinite(s_ins) and np.isfinite(insol0):
                s_ins = abs(insol0) * t5.floor_frac_insol
                floors_used["pl_insol"] = s_ins
            if not np.isfinite(s_e) and np.isfinite(e0):
                s_e = t5.floor_abs_ecc
                floors_used["pl_orbeccen"] = s_e
            if not np.isfinite(s_age) and np.isfinite(age0):
                s_age = t5.floor_abs_age_gyr
                floors_used["st_age"] = s_age

        # Sample
        n = t5.n_mc
        rade = _mc_sample_normal(rng, rade0, s_rade, n, lo=0.01, hi=50.0)
        m = _mc_sample_normal(rng, m0, s_m, n, lo=0.01, hi=1e5)
        ecc = _mc_sample_normal(rng, e0, s_e, n, lo=0.0, hi=1.0)
        teq = _mc_sample_normal(rng, teq0, s_teq, n, lo=1.0, hi=5000.0)
        insol = _mc_sample_normal(rng, insol0, s_ins, n, lo=1e-9, hi=1e6)
        age = _mc_sample_normal(rng, age0, s_age, n, lo=0.0, hi=20.0)

        # Tier2 gate pass per sample
        pass_t2 = np.ones(n, dtype=bool)
        pass_t2 &= np.isfinite(teq) & (teq >= t5.Teq_min_K) & (teq <= t5.Teq_max_K)
        pass_t2 &= np.isfinite(insol) & (insol >= t5.S_min) & (insol <= t5.S_max)
        pass_t2 &= np.isfinite(ecc) & (ecc <= t5.e_max)
        if t5.require_age:
            pass_t2 &= np.isfinite(age)

        # Tier3 derived
        rho, g, vesc_rel = _compute_rho_g_surface_vesc(rade, m)
        pass_t3 = pass_t2.copy()
        pass_t3 &= np.isfinite(rho) & (rho >= t5.rho_min_g_cm3) & (rho <= t5.rho_max_g_cm3)
        pass_t3 &= np.isfinite(vesc_rel) & (vesc_rel >= t5.vesc_min_rel_earth)

        # Tier4 depth proxy: if Rp and st_rad exist (st_rad in R_sun). If missing, use existing proxy if present.
        # Convert star radius to Earth radii: R_sun ~ 109.1 R_earth
        if "st_rad" in row and np.isfinite(_to_float(row.get("st_rad"))):
            st_rad_rsun = _to_float(row.get("st_rad"))
            st_rad_rearth = st_rad_rsun * 109.1
            depth_proxy = (rade / max(st_rad_rearth, 1e-9)) ** 2
        else:
            dp0 = _to_float(row.get("tran_depth_proxy"))
            depth_proxy = np.full(n, dp0 if np.isfinite(dp0) else float("nan"), dtype=float)

        # best_mag: deterministic from tier4 (we don't have mag uncertainties)
        best_mag = np.full(n, best_mag_nom if np.isfinite(best_mag_nom) else float("nan"), dtype=float)

        fscore = _followup_score(depth_proxy, best_mag, t5.depth_ref, t5.mag_ref_min, t5.mag_ref_max, t5.alpha_factor)
        pass_t4 = pass_t3.copy()
        # we do not add a hard tier4 gate by default; tier4 is scoring.

        # Ripeness scores (these mirror your deterministic definition)
        # T2: structural*age factor (age anchor) -> C_phi_pow * (age/age_scale)
        # If age missing, penalize by 0.
        age_factor = np.where(np.isfinite(age), age / max(t5.age_scale_gyr, 1e-9), 0.0)
        rip_t2 = (cphi if np.isfinite(cphi) else 0.0) * age_factor

        # T3: keep same unless you explicitly want to modulate by rho/vesc; here: identical to your current output style
        rip_t3 = rip_t2.copy()

        # T4: modulate by follow-up score factor as you’ve been doing
        # Use a soft multiplier: 1 + alpha*score (or whatever you want). Here we keep simple: 1 + 0.5*score
        followup_factor = 1.0 + 0.5 * fscore
        rip_t4 = rip_t3 * followup_factor

        # Summaries
        def pct(x, q):
            return float(np.nanpercentile(x, q)) if np.isfinite(np.nanmedian(x)) else float("nan")

        row_out = {
            "planet_id": pid,
            "hostname": host,
            "best_mag_band": best_mag_band if best_mag_band != "nan" else "",
            "best_mag": float(best_mag_nom) if np.isfinite(best_mag_nom) else float("nan"),
            "Phi_p": float(phi) if np.isfinite(phi) else float("nan"),
            "C_phi_pow": float(cphi) if np.isfinite(cphi) else float("nan"),
            "n_mc": int(n),
            "age_scale_gyr": float(t5.age_scale_gyr),

            "P_pass_T2": float(np.mean(pass_t2)),
            "P_pass_T3": float(np.mean(pass_t3)),
            "P_pass_T4": float(np.mean(pass_t4)),

            "Ripeness_T2_med": pct(rip_t2, 50),
            "Ripeness_T2_p16": pct(rip_t2, 16),
            "Ripeness_T2_p84": pct(rip_t2, 84),

            "Ripeness_T3_med": pct(rip_t3, 50),
            "Ripeness_T3_p16": pct(rip_t3, 16),
            "Ripeness_T3_p84": pct(rip_t3, 84),

            "Ripeness_T4_med": pct(rip_t4, 50),
            "Ripeness_T4_p16": pct(rip_t4, 16),
            "Ripeness_T4_p84": pct(rip_t4, 84),

            "depth_proxy_med": pct(depth_proxy, 50),
            "depth_proxy_p16": pct(depth_proxy, 16),
            "depth_proxy_p84": pct(depth_proxy, 84),

            "followup_score_med": pct(fscore, 50),
            "followup_factor_med": pct(followup_factor, 50),

            "rho_med_g_cm3": pct(rho, 50),
            "g_med_m_s2": pct(g, 50),
            "vesc_med_rel_earth": pct(vesc_rel, 50),
        }

        # record whether we used floors (helps reviewers)
        row_out["floors_used_json"] = json.dumps(floors_used, sort_keys=True)

        out_rows.append(row_out)

    out = pd.DataFrame(out_rows)

    # Rank by Ripeness_T4_med (or whatever you want)
    if len(out):
        out = out.sort_values("Ripeness_T4_med", ascending=False).reset_index(drop=True)
        out["rank_tier5"] = np.arange(1, len(out) + 1)

    scorecard_csv = os.path.join(out_dir, "tier5_scorecard.csv")
    out.to_csv(scorecard_csv, index=False)

    summary = {
        "run_id": t5.run_id,
        "tier4_enriched_csv": t5.tier4_enriched_csv,
        "n_input": int(len(df)),
        "n_output": int(len(out)),
        "n_mc": int(t5.n_mc),
        "seed": int(t5.seed),
        "age_scale_gyr": float(t5.age_scale_gyr),
        "ps_merge": ps_merge_report,
        "ps_select_method": t5.ps_select_method,
        "use_ps_errors": bool(t5.use_ps_errors),
        "use_floors_when_missing": bool(t5.use_floors_when_missing),
        "floors": {
            "floor_frac_rade": t5.floor_frac_rade,
            "floor_frac_bmasse": t5.floor_frac_bmasse,
            "floor_frac_teq": t5.floor_frac_teq,
            "floor_frac_insol": t5.floor_frac_insol,
            "floor_abs_ecc": t5.floor_abs_ecc,
            "floor_abs_age_gyr": t5.floor_abs_age_gyr,
        },
        "gates": {
            "S_min": t5.S_min,
            "S_max": t5.S_max,
            "Teq_min_K": t5.Teq_min_K,
            "Teq_max_K": t5.Teq_max_K,
            "e_max": t5.e_max,
            "require_age": bool(t5.require_age),
            "require_rocky": bool(t5.require_rocky),
            "rocky_rmax_rearth": float(t5.rocky_rmax_rearth),
        },
        "tier3": {
            "rho_min_g_cm3": t5.rho_min_g_cm3,
            "rho_max_g_cm3": t5.rho_max_g_cm3,
            "vesc_min_rel_earth": t5.vesc_min_rel_earth,
        },
        "tier4": {
            "depth_ref": t5.depth_ref,
            "mag_ref_min": t5.mag_ref_min,
            "mag_ref_max": t5.mag_ref_max,
            "alpha_factor": t5.alpha_factor,
        },
        "outputs": {
            "tier5_scorecard_csv": scorecard_csv,
            "tier5_mc_samples_csv": None,
        },
        "timestamp_utc": _utc_now_str(),
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    run_meta = {
        "run_id": t5.run_id,
        "config_path": config_path,
        "tier4_enriched_csv": t5.tier4_enriched_csv,
        "tier4_sha256": _sha256_file(t5.tier4_enriched_csv) if os.path.exists(t5.tier4_enriched_csv) else None,
        "ps_csv": t5.ps_csv,
        "ps_sha256": _sha256_file(t5.ps_csv) if (t5.ps_csv and os.path.exists(t5.ps_csv)) else None,
    }
    run_meta_path = os.path.join(out_dir, "run_meta.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[TIER5] Wrote: {scorecard_csv}")
    print(f"[TIER5] Wrote: {summary_path}")
    print(f"[TIER5] Wrote: {run_meta_path}")
    print("[TIER5] Done.")
