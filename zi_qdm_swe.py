# -*- coding: utf-8 -*-
"""
ZI-QDM for SWE (daily, 0.25°, 25–90°N) —— 历史=ZI-QM，未来=ZI-QDM
- 训练只用一次（基于 ssp126 历史），对 126/245/585 整段(1981–2100)应用
- 只校正 ERA 和 CMIP 首日同时非NaN的格点；其余保持 NaN
- log1p 域、DOY±15，阈值 tau>0
- 多进程按“行”并行；主进程串行写 netCDF；打印行进度与 ETA
"""

import argparse
import os, sys, time
import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# File paths are read from config.yml. Scientific settings below retain the
# values used in the supplied research code.
ERA_PATH = None
SSP_LIST = []
Q_OCC_TMP = None

ERA_VAR   = "sd"
CMIP_VAR  = "snw"
TAU = 0.0
WIN_HALF = 15
CLIP_Q = (0.01, 0.99)
USE_LOG1P = True
PROCESSES = 4
PRINT_EVERY_TRAIN = 4
PRINT_EVERY_APPLY = 4


def load_config(config_path):
    """Load paths and execution settings without changing the ZI-QDM method."""
    global ERA_PATH, SSP_LIST, Q_OCC_TMP, ERA_VAR, CMIP_VAR, PROCESSES
    global PRINT_EVERY_TRAIN, PRINT_EVERY_APPLY
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    ERA_PATH = str(Path(cfg["era_path"]).expanduser().resolve())
    SSP_LIST = [str(Path(path).expanduser().resolve()) for path in cfg["scenario_paths"]]
    Q_OCC_TMP = str(Path(cfg.get("q_occ_tmp", "./_q_occ_all_tmp.npy")).expanduser().resolve())
    ERA_VAR = cfg.get("era_variable", "sd")
    CMIP_VAR = cfg.get("cmip_variable", "snw")
    PROCESSES = int(cfg.get("processes", 4))
    PRINT_EVERY_TRAIN = int(cfg.get("print_every_train", 4))
    PRINT_EVERY_APPLY = int(cfg.get("print_every_apply", 4))
    if not SSP_LIST:
        raise ValueError("scenario_paths must contain at least one NetCDF file")
    if not os.path.isfile(ERA_PATH):
        raise FileNotFoundError(f"ERA5-Land file not found: {ERA_PATH}")
    for path in SSP_LIST:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def make_dates(start, n): return pd.date_range(pd.Timestamp(start), periods=n, freq="D")

def build_doy_windows(dates, win_half=15):
    doy = dates.dayofyear.values.astype(int)
    is_229 = (dates.month.values == 2) & (dates.day.values == 29)
    doy[is_229] = 59
    groups = {}
    for d in range(1, 366):
        dist = np.minimum((doy - d) % 365, (d - doy) % 365)
        groups[d] = np.where(dist <= win_half)[0]
    return doy, groups

def ecdf_ppf(sorted_samples, u):
    n = sorted_samples.size
    if n == 0: return np.full_like(u, np.nan, dtype=np.float32)
    if n == 1: return np.full_like(u, sorted_samples[0], dtype=np.float32)
    u = np.clip(u, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.interp(u, grid, sorted_samples).astype(np.float32)

def ecdf_cdf(sorted_samples, x):
    n = sorted_samples.size
    if n == 0: return np.full_like(x, np.nan, dtype=np.float32)
    idx = np.searchsorted(sorted_samples, x, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    if n == 1: return np.ones_like(x, dtype=np.float32)
    return (idx / (n - 1)).astype(np.float32)

def qm_log1p(x_pos, obs_sorted, mod_sorted, clip=CLIP_Q):
    z = np.log1p(x_pos)
    u = ecdf_cdf(mod_sorted, z)
    u = np.clip(u, clip[0], clip[1])
    z_out = ecdf_ppf(obs_sorted, u)
    return np.expm1(z_out)

def qdm_log1p(x_pos, obs_tr_sorted, mod_tr_sorted, mod_fu_sorted, clip=CLIP_Q):
    z = np.log1p(x_pos)
    u_fu = ecdf_cdf(mod_fu_sorted, z)
    u_fu = np.clip(u_fu, clip[0], clip[1])
    q_obs = ecdf_ppf(obs_tr_sorted, u_fu)
    q_mod = ecdf_ppf(mod_tr_sorted, u_fu)
    z_out = z + (q_obs - q_mod)
    return np.expm1(z_out)

_SKIP_ATTS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

def _get_fill_value(vin):
    try: return vin.getncattr("_FillValue")
    except Exception: return None

def _copy_attrs(vin, vout):
    for att in vin.ncattrs():
        if att in _SKIP_ATTS: continue
        try: setattr(vout, att, getattr(vin, att))
        except Exception: pass

def train_q_occ():
    era_ds  = Dataset(ERA_PATH, "r")
    T_obs = era_ds.dimensions["time"].size
    H     = era_ds.dimensions["lat"].size
    W     = era_ds.dimensions["lon"].size
    lat   = era_ds.variables["lat"][:]
    lon   = era_ds.variables["lon"][:]
    dates_obs = make_dates("1981-01-01", T_obs)
    assert dates_obs[0] == pd.Timestamp("1981-01-01") and dates_obs[-1] == pd.Timestamp("2020-12-31")

    ref_ds = Dataset(SSP_LIST[0], "r")
    T_mod  = ref_ds.dimensions["time"].size
    lat2   = ref_ds.variables["lat"][:]
    lon2   = ref_ds.variables["lon"][:]
    assert H==lat2.size and W==lon2.size and np.allclose(lat, lat2) and np.allclose(lon, lon2), "经纬网格不一致"
    dates_mod = make_dates("1981-01-01", T_mod)
    idx_mod_hist = np.where((dates_mod >= pd.Timestamp("1981-01-01")) & (dates_mod <= pd.Timestamp("2020-12-31")))[0]
    _, groups_obs = build_doy_windows(dates_obs, WIN_HALF)
    _, groups_mod = build_doy_windows(dates_mod, WIN_HALF)

    print("[Train] 计算 q_occ(d)（逐像元）", flush=True)
    era_first = era_ds.variables[ERA_VAR][0, :, :]
    cm0_first = ref_ds.variables[CMIP_VAR][0, :, :]
    try: era_first = era_first.filled(np.nan)
    except: pass
    try: cm0_first = cm0_first.filled(np.nan)
    except: pass
    valid_mask = np.isfinite(era_first) & np.isfinite(cm0_first)
    q_occ_all = np.full((H, W, 366), np.nan, dtype=np.float32)

    t0 = time.time()
    for i in range(H):
        if (i % PRINT_EVERY_TRAIN == 0) or (i == H-1):
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / max(done,1) * (H - done)
            print(f"[Train] row {done}/{H} | elapsed {elapsed/3600:.2f} h | ETA ~{eta/3600:.2f} h", flush=True)
        era_row = era_ds.variables[ERA_VAR][:, i, :].astype(np.float32)
        mod_row = ref_ds.variables[CMIP_VAR][:, i, :].astype(np.float32)
        for j in range(W):
            if not valid_mask[i, j]: 
                continue
            obs_ij = era_row[:, j]
            mod_ij = mod_row[:, j]
            for d in range(1, 366):
                win_obs_idx = groups_obs[d]
                if win_obs_idx.size < 10: continue
                p_obs = float(np.mean(obs_ij[win_obs_idx] > TAU))
                win_mod_idx = groups_mod[d]
                win_mod_idx = win_mod_idx[(win_mod_idx >= idx_mod_hist[0]) & (win_mod_idx <= idx_mod_hist[-1])]
                if win_mod_idx.size < 10: continue
                x_tr = mod_ij[win_mod_idx]
                x_tr = x_tr[np.isfinite(x_tr)]
                if x_tr.size < 10: continue
                x_tr_sorted = np.sort(x_tr)
                u = np.array([np.clip(1.0 - p_obs, 0.0, 1.0)], dtype=np.float32)
                q_occ_all[i, j, d] = ecdf_ppf(x_tr_sorted, u)[0]

    ref_ds.close(); era_ds.close()
    np.save(Q_OCC_TMP, q_occ_all.astype(np.float32))
    return valid_mask, q_occ_all.shape

def process_one_row(i, scen_path, hist_start, hist_end, fut_start,
                    CLIP_Q, USE_LOG1P, TAU, WIN_HALF, ERA_PATH, CMIP_VAR, ERA_VAR, q_occ_path):
    in_ds  = Dataset(scen_path, "r")
    era_ds = Dataset(ERA_PATH, "r")
    in_row  = in_ds.variables[CMIP_VAR][:, i, :].astype(np.float32)
    obs_row = era_ds.variables[ERA_VAR][:, i, :].astype(np.float32)

    T_mod = in_ds.dimensions["time"].size
    T_obs = era_ds.dimensions["time"].size
    dates_mod = make_dates("1981-01-01", T_mod)
    dates_obs = make_dates("1981-01-01", T_obs)
    _, groups_mod = build_doy_windows(dates_mod, WIN_HALF)
    _, groups_obs = build_doy_windows(dates_obs, WIN_HALF)

    idx_mod_hist = np.arange(hist_start, hist_end+1) if (hist_end >= hist_start) else np.array([], dtype=int)
    idx_mod_fut  = np.arange(fut_start, T_mod) if (fut_start < T_mod) else np.array([], dtype=int)

    q_occ_all = np.load(q_occ_path, mmap_mode="r")
    q_occ_line = q_occ_all[i]  # (W,366)
    W = in_row.shape[1]
    out_row = np.full_like(in_row, np.nan, dtype=np.float32)

    era_first = era_ds.variables[ERA_VAR][0, i, :]
    cm0_first = in_ds.variables[CMIP_VAR][0, i, :]
    try: era_first = era_first.filled(np.nan)
    except: pass
    try: cm0_first = cm0_first.filled(np.nan)
    except: pass
    valid_line = np.isfinite(era_first) & np.isfinite(cm0_first)

    for j in range(W):
        if not valid_line[j]: continue
        obs_ij = obs_row[:, j]
        mod_ij = in_row[:, j]
        q_occ  = q_occ_line[j, :]

        if idx_mod_hist.size:
            for d in range(1, 366):
                kk = groups_mod[d]
                kk = kk[(kk >= idx_mod_hist[0]) & (kk <= idx_mod_hist[-1])]
                if kk.size == 0: continue
                xb = mod_ij[kk]; has = xb > q_occ[d]; xout = xb.copy()
                win_obs_idx = groups_obs[d]
                pos_obs = obs_ij[win_obs_idx]; pos_obs = pos_obs[pos_obs > TAU]
                win_mod_hist = groups_mod[d]
                win_mod_hist = win_mod_hist[(win_mod_hist >= idx_mod_hist[0]) & (win_mod_hist <= idx_mod_hist[-1])]
                pos_mod_tr = mod_ij[win_mod_hist]; pos_mod_tr = pos_mod_tr[pos_mod_tr > TAU]
                if pos_obs.size >= 10 and pos_mod_tr.size >= 10:
                    if USE_LOG1P:
                        obs_sorted = np.sort(np.log1p(pos_obs))
                        mod_sorted = np.sort(np.log1p(pos_mod_tr))
                        pm = (xb > TAU)
                        if np.any(pm):
                            xin = xb[pm]
                            xout[pm] = qm_log1p(xin, obs_sorted, mod_sorted, CLIP_Q)
                    else:
                        pm = (xb > TAU)
                        if np.any(pm):
                            xin = xb[pm]
                            u = ecdf_cdf(np.sort(pos_mod_tr), xin)
                            u = np.clip(u, CLIP_Q[0], CLIP_Q[1])
                            xout[pm] = ecdf_ppf(np.sort(pos_obs), u)
                xout[~has] = 0.0
                out_row[kk, j] = xout.astype(np.float32)

        if idx_mod_fut.size:
            for d in range(1, 366):
                kk = groups_mod[d]; kk = kk[(kk >= idx_mod_fut[0])]
                if kk.size == 0: continue
                xb = mod_ij[kk]; has = xb > q_occ[d]; xout = xb.copy()
                win_obs_idx = groups_obs[d]
                pos_obs = obs_ij[win_obs_idx]; pos_obs = pos_obs[pos_obs > TAU]
                win_mod_hist = groups_mod[d]
                win_mod_hist = win_mod_hist[(win_mod_hist >= idx_mod_hist[0]) & (win_mod_hist <= idx_mod_hist[-1])]
                pos_mod_hist = mod_ij[win_mod_hist]; pos_mod_hist = pos_mod_hist[pos_mod_hist > TAU]
                pos_mod_fu = xb[xb > TAU]
                if pos_obs.size >= 10 and pos_mod_hist.size >= 10 and pos_mod_fu.size >= 10:
                    if USE_LOG1P:
                        obs_tr_sorted = np.sort(np.log1p(pos_obs))
                        mod_tr_sorted = np.sort(np.log1p(pos_mod_hist))
                        mod_fu_sorted = np.sort(np.log1p(pos_mod_fu))
                        pm = (xb > TAU)
                        if np.any(pm):
                            xin = xb[pm]
                            xout[pm] = qdm_log1p(xin, obs_tr_sorted, mod_tr_sorted, mod_fu_sorted, CLIP_Q)
                    else:
                        pm = (xb > TAU)
                        if np.any(pm):
                            xin = xb[pm]
                            u_fu = ecdf_cdf(np.sort(pos_mod_fu), xin)
                            u_fu = np.clip(u_fu, CLIP_Q[0], CLIP_Q[1])
                            q_obs = ecdf_ppf(np.sort(pos_obs), u_fu)
                            q_mod = ecdf_ppf(np.sort(pos_mod_hist), u_fu)
                            xout[pm] = xin + (q_obs - q_mod)
                xout[~has] = 0.0
                out_row[kk, j] = xout.astype(np.float32)

    era_ds.close(); in_ds.close()
    return i, out_row

def main():
    print("[Paths]")
    print(" ERA :", ERA_PATH)
    for p in SSP_LIST: print(" CMIP:", p)

    valid_mask, q_shape = train_q_occ()
    H, W, _ = q_shape
    print("[Train] 完成，q_occ_all 保存到:", Q_OCC_TMP, flush=True)

    for scen_path in SSP_LIST:
        print(f"\n[Apply] {os.path.basename(scen_path)}", flush=True)
        in_ds  = Dataset(scen_path, "r")
        out_path = scen_path[:-3] + "_corrected.nc"
        out_ds = Dataset(out_path, "w", format="NETCDF4")

        for name, dim in in_ds.dimensions.items():
            out_ds.createDimension(name, (len(dim) if not dim.isunlimited() else None))

        for vname in ["time", "lat", "lon"]:
            vin = in_ds.variables[vname]
            fv  = _get_fill_value(vin)
            if fv is None:
                vout = out_ds.createVariable(vname, vin.datatype, vin.dimensions, zlib=True, complevel=4)
            else:
                vout = out_ds.createVariable(vname, vin.datatype, vin.dimensions, zlib=True, complevel=4, fill_value=fv)
            vout[:] = vin[:]; _copy_attrs(vin, vout)

        vin  = in_ds.variables[CMIP_VAR]
        fv   = _get_fill_value(vin)
        if fv is None:
            vout = out_ds.createVariable(CMIP_VAR, "f4", vin.dimensions, zlib=True, complevel=4, fill_value=np.nan)
        else:
            vout = out_ds.createVariable(CMIP_VAR, "f4", vin.dimensions, zlib=True, complevel=4, fill_value=fv)
        _copy_attrs(vin, vout)
        try: vout.units = vin.units
        except Exception: vout.units = "mm"

        T_mod  = in_ds.dimensions["time"].size
        dates_mod = make_dates("1981-01-01", T_mod)
        idx_mod_hist = np.where((dates_mod >= pd.Timestamp("1981-01-01")) & (dates_mod <= pd.Timestamp("2020-12-31")))[0]
        idx_mod_fut  = np.where(dates_mod > pd.Timestamp("2020-12-31"))[0]
        hist_start, hist_end = (int(idx_mod_hist[0]), int(idx_mod_hist[-1])) if idx_mod_hist.size else (0, -1)
        fut_start = int(idx_mod_fut[0]) if idx_mod_fut.size else T_mod

        t1 = time.time()
        with ProcessPoolExecutor(max_workers=PROCESSES) as ex:
            futures = [
                ex.submit(process_one_row, i, scen_path,
                          hist_start, hist_end, fut_start,
                          CLIP_Q, USE_LOG1P, TAU, WIN_HALF,
                          ERA_PATH, CMIP_VAR, ERA_VAR, Q_OCC_TMP)
                for i in range(H)
            ]
            done = 0
            for fut in as_completed(futures):
                i, out_row = fut.result()
                vout[:, i, :] = out_row
                done += 1
                if (done % PRINT_EVERY_APPLY == 0) or (done == H):
                    el = time.time() - t1
                    eta = el / max(done,1) * (H - done)
                    print(f"[Apply] {os.path.basename(scen_path)} row {done}/{H} | elapsed {el/3600:.2f} h | ETA ~{eta/3600:.2f} h", flush=True)

        for att in in_ds.ncattrs():
            try: setattr(out_ds, att, getattr(in_ds, att))
            except Exception: pass
        out_ds.history = (getattr(in_ds, "history", "") + " | ZI-QDM corrected (history=ZI-QM, future=ZI-QDM, tau>0, log1p, DOY±15)").strip()

        out_ds.close(); in_ds.close()
        print(f"[Saved] {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZI-QM/ZI-QDM correction for daily SWE")
    parser.add_argument("--config", default="config.yml", help="YAML configuration file")
    args = parser.parse_args()
    load_config(args.config)
    main()
