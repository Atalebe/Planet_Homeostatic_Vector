from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from phv.io.exo_csv import read_exoplanet_csv
from phv.metrics.proxies import (
    stellar_luminosity_Lsun,
    log10_flux_rel_earth,
    proxy_H,
    proxy_M,
    proxy_S,
    proxy_R,
)
from phv.metrics.normalize import mad_z
from phv.metrics.window import in_window
from phv.metrics.ripeness import static_ripeness
from phv.utils.hashing import sha256_of_file
from phv.utils.logging import log


# ---------------------------
# Helpers
# ---------------------------

def enforce_required(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for c in required:
        out = out[out[c].notna()]
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_default_flag_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {"default_flag_present": False, "n_after_default_flag": int(df.shape[0])}
    if "default_flag" in df.columns:
        info["default_flag_present"] = True
        d = pd.to_numeric(df["default_flag"], errors="coerce")
        df2 = df.loc[d == 1].copy()
        info["n_after_default_flag"] = int(df2.shape[0])
        return df2, info
    return df, info


def _deduplicate_planets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Dedup by pl_name. Prefer rowupdate newest if available.
    """
    info = {"dedup_key": None, "rowupdate_present": False, "n_unique": int(df.shape[0])}
    if "pl_name" not in df.columns:
        return df, info

    info["dedup_key"] = "pl_name"

    if "rowupdate" in df.columns:
        info["rowupdate_present"] = True
        tmp = df.copy()
        tmp["rowupdate_parsed"] = pd.to_datetime(tmp["rowupdate"], errors="coerce")
        tmp.sort_values(["pl_name", "rowupdate_parsed"], ascending=[True, False], inplace=True)
        out = tmp.drop_duplicates(subset=["pl_name"], keep="first").drop(columns=["rowupdate_parsed"])
    else:
        out = df.drop_duplicates(subset=["pl_name"], keep="first").copy()

    info["n_unique"] = int(out.shape[0])
    return out, info


def classify_radius(df: pd.DataFrame, rocky_rmax: float, subnep_rmax: float) -> pd.Series:
    if "pl_rade" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)

    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    cls = pd.Series(["unknown"] * len(df), index=df.index)

    cls.loc[(r.notna()) & (r > 0) & (r <= rocky_rmax)] = "rocky"
    cls.loc[(r.notna()) & (r > rocky_rmax) & (r <= subnep_rmax)] = "subnep"
    cls.loc[(r.notna()) & (r > subnep_rmax)] = "giant"
    return cls


def mad_z_by_group(series: pd.Series, group: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for gval, idx in group.groupby(group).groups.items():
        x = pd.to_numeric(series.loc[idx], errors="coerce").astype(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if (not np.isfinite(mad)) or mad == 0 or len(idx) < 5:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (x - med) / mad
    return out


def quantile_bounds(phi: pd.Series, qlo: float, qhi: float) -> tuple[float, float]:
    x = pd.to_numeric(phi, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return (float("nan"), float("nan"))
    return (float(np.nanquantile(x, qlo)), float(np.nanquantile(x, qhi)))


# ---------------------------
# Tier1 runner
# ---------------------------

def run_tier1(cfg: dict, config_path: str) -> None:
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    csv_path = cfg["inputs"]["exoplanet_csv"]
    df = read_exoplanet_csv(csv_path)
    n_rows_initial = int(df.shape[0])

    # numeric coercion for common columns
    df = _coerce_numeric(
        df,
        cols=[
            "pl_orbper","pl_orbsmax","st_teff","st_rad","st_lum","st_mass","st_met","st_age",
            "pl_bmasse","pl_rade","pl_orbeccen","pl_insol","pl_eqt","default_flag"
        ],
    )

    # Required columns
    required = cfg["tier"]["require_columns"]
    df_req = enforce_required(df, required)
    n_rows_after_required = int(df_req.shape[0])

    # Default set and dedup
    df_def, def_info = _apply_default_flag_filter(df_req)
    n_rows_after_default = int(df_def.shape[0])

    df_uni, dedup_info = _deduplicate_planets(df_def)
    n_unique = int(df_uni.shape[0])

    # Luminosity source
    if "st_lum" in df_uni.columns and df_uni["st_lum"].notna().any():
        L = np.power(10.0, df_uni["st_lum"].astype(float))
        lum_source = "st_lum"
    else:
        L = stellar_luminosity_Lsun(df_uni["st_rad"], df_uni["st_teff"])
        lum_source = "R^2T^4"

    logF = log10_flux_rel_earth(L, df_uni["pl_orbsmax"])

    # Proxies
    H = proxy_H(logF)
    M = proxy_M(
        df_uni["pl_bmasse"] if "pl_bmasse" in df_uni.columns else None,
        df_uni["pl_rade"] if "pl_rade" in df_uni.columns else None,
    )
    S = proxy_S(
        df_uni["pl_orbeccen"] if "pl_orbeccen" in df_uni.columns else None,
        df_uni["st_age"] if "st_age" in df_uni.columns else None,
    )
    R = proxy_R(
        logF,
        F_mid_log10=float(cfg["proxies"]["F_mid_log10"]),
        pl_bmasse=(df_uni["pl_bmasse"] if "pl_bmasse" in df_uni.columns else None),
    )

    # Normalization mode
    window_cfg = cfg.get("window", {})
    mode = window_cfg.get("mode", "fixed")

    if mode == "fixed":
        # global MAD-z
        Rh = mad_z(R)
        Hh = mad_z(H)
        Mh = mad_z(M)
        Sh = mad_z(S)

    elif mode in ("quantile_global", "quantile_by_class"):
        # still compute hats, but choice of group differs
        if mode == "quantile_global":
            Rh = mad_z(R)
            Hh = mad_z(H)
            Mh = mad_z(M)
            Sh = mad_z(S)
        else:
            rocky_rmax = float(cfg.get("diagnostics", {}).get("rocky_rmax_rearth", 1.8))
            subnep_rmax = float(cfg.get("diagnostics", {}).get("subnep_rmax_rearth", 4.0))
            pclass = classify_radius(df_uni, rocky_rmax=rocky_rmax, subnep_rmax=subnep_rmax)

            Rh = mad_z_by_group(R, pclass)
            Hh = mad_z_by_group(H, pclass)
            Mh = mad_z_by_group(M, pclass)
            Sh = mad_z_by_group(S, pclass)
    else:
        raise ValueError(f"Unknown window.mode: {mode}")

    # Phi
    w = cfg["weights"]
    Phi = (
        float(w["w_R"]) * Rh
        + float(w["w_H"]) * Hh
        + float(w["w_M"]) * Mh
        + float(w["w_S"]) * Sh
    )

    # Window membership + bounds record
    bounds_record: dict = {}
    win = np.zeros(len(df_uni), dtype=int)

    if mode == "fixed":
        Phi_min = float(window_cfg["Phi_min"])
        Phi_max = float(window_cfg["Phi_max"])
        win = in_window(Phi.to_numpy(dtype=float), Phi_min, Phi_max).astype(int)
        bounds_record["global"] = {"Phi_min": Phi_min, "Phi_max": Phi_max}

    elif mode == "quantile_global":
        qlo = float(window_cfg.get("qlo", 0.40))
        qhi = float(window_cfg.get("qhi", 0.60))
        lo, hi = quantile_bounds(Phi, qlo, qhi)
        win = in_window(Phi.to_numpy(dtype=float), lo, hi).astype(int)
        bounds_record["global"] = {"qlo": qlo, "qhi": qhi, "Phi_min": lo, "Phi_max": hi}

    elif mode == "quantile_by_class":
        qlo = float(window_cfg.get("qlo", 0.40))
        qhi = float(window_cfg.get("qhi", 0.60))
        rocky_rmax = float(cfg.get("diagnostics", {}).get("rocky_rmax_rearth", 1.8))
        subnep_rmax = float(cfg.get("diagnostics", {}).get("subnep_rmax_rearth", 4.0))
        pclass = classify_radius(df_uni, rocky_rmax=rocky_rmax, subnep_rmax=subnep_rmax)

        for cls in ["rocky", "subnep", "giant", "unknown"]:
            mask = (pclass == cls).to_numpy()
            n = int(mask.sum())
            if n < 20:
                bounds_record[cls] = {"n": n, "qlo": qlo, "qhi": qhi, "Phi_min": None, "Phi_max": None}
                continue
            lo, hi = quantile_bounds(Phi.loc[mask], qlo, qhi)
            bounds_record[cls] = {"n": n, "qlo": qlo, "qhi": qhi, "Phi_min": lo, "Phi_max": hi}
            win[mask] = in_window(Phi.loc[mask].to_numpy(dtype=float), lo, hi).astype(int)

    # Ripeness static (now possible if st_age exists)
    st_age = df_uni["st_age"] if "st_age" in df_uni.columns else None
    rip = static_ripeness(pd.Series(win, index=df_uni.index), st_age)

    # Output table
    out = pd.DataFrame(
        {
            "planet_id": df_uni["pl_name"].astype(str) if "pl_name" in df_uni.columns else df_uni.index.astype(str),
            "hostname": df_uni["hostname"].astype(str) if "hostname" in df_uni.columns else "",
            "default_flag": df_uni["default_flag"] if "default_flag" in df_uni.columns else np.nan,
            "rowupdate": df_uni["rowupdate"] if "rowupdate" in df_uni.columns else "",
            "disc_year": df_uni["disc_year"] if "disc_year" in df_uni.columns else np.nan,
            "discoverymethod": df_uni["discoverymethod"] if "discoverymethod" in df_uni.columns else "",
            "pl_orbper": df_uni["pl_orbper"].astype(float),
            "pl_orbsmax": df_uni["pl_orbsmax"].astype(float),
            "pl_orbeccen": df_uni["pl_orbeccen"] if "pl_orbeccen" in df_uni.columns else np.nan,
            "pl_rade": df_uni["pl_rade"] if "pl_rade" in df_uni.columns else np.nan,
            "pl_bmasse": df_uni["pl_bmasse"] if "pl_bmasse" in df_uni.columns else np.nan,
            "pl_insol": df_uni["pl_insol"] if "pl_insol" in df_uni.columns else np.nan,
            "pl_eqt": df_uni["pl_eqt"] if "pl_eqt" in df_uni.columns else np.nan,
            "st_teff": df_uni["st_teff"].astype(float),
            "st_rad": df_uni["st_rad"].astype(float),
            "st_mass": df_uni["st_mass"] if "st_mass" in df_uni.columns else np.nan,
            "st_age": df_uni["st_age"] if "st_age" in df_uni.columns else np.nan,
            "st_met": df_uni["st_met"] if "st_met" in df_uni.columns else np.nan,
            "st_lum_log10": df_uni["st_lum"] if "st_lum" in df_uni.columns else np.nan,
            "lum_source": lum_source,
            "logF": logF.astype(float),
            "R_raw": R.astype(float),
            "H_raw": H.astype(float),
            "M_raw": M.astype(float),
            "S_raw": S.astype(float),
            "Rhat": Rh.astype(float),
            "Hhat": Hh.astype(float),
            "Mhat": Mh.astype(float),
            "Shat": Sh.astype(float),
            "Phi_p": Phi.astype(float),
            "in_window": win.astype(int),
            "ripeness_static": rip.astype(float),
        }
    )

    per_planet_path = os.path.join(out_dir, "per_planet_table.csv")
    out.to_csv(per_planet_path, index=False)

    # Summary
    window_fraction = float(np.mean(out["in_window"].to_numpy(dtype=float))) if out.shape[0] else float("nan")
    in_window_count = int(out["in_window"].sum())

    age_mask = out["st_age"].notna().to_numpy()
    age_n = int(np.sum(age_mask))
    age_win = float(np.mean(out.loc[age_mask, "in_window"].to_numpy(dtype=float))) if age_n > 0 else float("nan")

    summary = {
        "run_id": cfg["run"]["run_id"],
        "input_csv": csv_path,
        "n_rows_initial": n_rows_initial,
        "n_rows_after_required": n_rows_after_required,
        "default_flag_present": def_info["default_flag_present"],
        "n_rows_after_default_flag": n_rows_after_default,
        "dedup_key": dedup_info["dedup_key"],
        "rowupdate_present": dedup_info["rowupdate_present"],
        "n_unique_planets": n_unique,
        "required_columns": required,
        "lum_source": lum_source,
        "window_mode": mode,
        "window_bounds": bounds_record,
        "in_window_count": in_window_count,
        "window_fraction": window_fraction,
        "age_known_n": age_n,
        "age_known_window_fraction": age_win,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "run_id": cfg["run"]["run_id"],
        "config_path": config_path,
        "config_sha256": sha256_of_file(config_path),
        "input_csv": csv_path,
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"Wrote: {per_planet_path}")
    log(f"Wrote: {summary_path}")
    log(f"Wrote: {meta_path}")
