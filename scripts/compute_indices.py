"""
Compute tambon-level drought indicators, two composite indices and alert levels
from the Earth Engine CSV exports, and write the dashboard data files.

    python scripts/compute_indices.py --raw raw/ --boundaries data/sukhothai_tambon_web.geojson --out data/
    python scripts/compute_indices.py --mock --boundaries ... --out data/     # synthetic demo data (flagged as MOCK)

Indicators (all per tambon, per month)
  spi1/3/6     SPI from CHIRPS (gamma fit per calendar month, baseline 1991-2020)
  spei3        rain - PET (ERA5-Land), empirical percentile -> z
  sm_z         soil moisture layer 1 (ERA5-Land), percentile -> z
  smdeep_z     soil moisture layer 2 (ERA5-Land), proxy for shallow storage
  vci/tci/vhi  MODIS NDVI (cropland) + LST, Kogan method per calendar month
  water_z      Sentinel-2 open-water fraction anomaly per calendar month

Composite (0-1, higher = worse), PROVISIONAL weights to be calibrated on 2019/20 DDPM declarations
  agri_index   0.25 SPI-3 + 0.30 soil moisture + 0.30 VHI + 0.15 SPEI-3
  water_index  0.35 surface-water anomaly + 0.25 SPI-6 + 0.15 deep soil moisture + 0.25 DDPM history
Alert
  3 red    water_index >= 0.60  -> prepare water delivery
  2 orange agri_index  >= 0.50 and cropland >= 10% of area -> warn farmers
  1 yellow either index >= 0.30 -> watch
  0 normal
"""
import argparse, json, os, sys, datetime
import numpy as np, pandas as pd
from scipy import stats

BASE = (1991, 2020)
ANALOG_YEARS = [2015, 2019, 2023]      # El Niño years used for the December outlook


# ---------- standardisation helpers ----------
def spi_series(rain, k):
    """rain: monthly Series indexed by Timestamp. Returns SPI-k Series."""
    acc = rain.rolling(k, min_periods=k).sum()
    out = pd.Series(np.nan, index=acc.index)
    for m in range(1, 13):
        sel = acc[acc.index.month == m].dropna()
        base = sel[(sel.index.year >= BASE[0]) & (sel.index.year <= BASE[1])]
        if len(base) < 15:
            continue
        q0 = (base == 0).mean()
        nz = base[base > 0]
        a, loc, sc = stats.gamma.fit(nz, floc=0)
        p = q0 + (1 - q0) * stats.gamma.cdf(sel.clip(lower=0), a, loc=0, scale=sc)
        out.loc[sel.index] = stats.norm.ppf(np.clip(p, 0.001, 0.999))
    return out


def empirical_z(s):
    """Percentile within the same calendar month -> standard normal z."""
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        sel = s[s.index.month == m].dropna()
        if len(sel) < 5:
            continue
        r = sel.rank(pct=True) * len(sel) / (len(sel) + 1)
        out.loc[sel.index] = stats.norm.ppf(r)
    return out


def kogan(s, invert=False):
    """VCI / TCI style 0-100 condition index per calendar month."""
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        sel = s[s.index.month == m].dropna()
        if len(sel) < 5:
            continue
        lo, hi = sel.min(), sel.max()
        v = (hi - sel) / (hi - lo) if invert else (sel - lo) / (hi - lo)
        out.loc[sel.index] = (v * 100).clip(0, 100)
    return out


sev_z = lambda z: np.clip(-z / 2.0, 0, 1)            # z=-2 -> 1, z>=0 -> 0
sev_vhi = lambda v: np.clip((40 - v) / 30.0, 0, 1)   # VHI<=10 -> 1, >=40 -> 0


def wmean(parts):
    """weighted mean ignoring missing parts (renormalises weights)."""
    num = sum(w * v.fillna(0) for w, v in parts)
    den = sum(w * v.notna() for w, v in parts)
    return num / den.replace(0, np.nan)


# ---------- load ----------
SCHEMA = {"chirps_monthly": ["rain_mm"], "era5land_monthly": ["sm_l1", "sm_l2", "era_rain_m", "pet_m"],
          "modis_ndvi_monthly": ["ndvi", "ndvi_crop"], "modis_lst_monthly": ["lst_c"],
          "s2_water_monthly": ["water_km2", "clear_km2"]}


def load_raw(raw, pop_factor=1.0):
    """Reads the EE exports. Missing files or all-empty columns fall back where possible; each fallback is noted."""
    notes = []
    def rd(name):
        path = os.path.join(raw, f"{name}.csv")
        if os.path.exists(os.path.join(raw, f"{name}_v2.csv")):      # re-export from the fixed notebook wins
            path = os.path.join(raw, f"{name}_v2.csv")
        if not os.path.exists(path):
            notes.append(f"{name}: ยังไม่มีไฟล์")
            return pd.DataFrame({"pcode": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                                 **{c: pd.Series(dtype=float) for c in SCHEMA[name]}})
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"] + "-01")
        empty = [c for c in SCHEMA[name] if c not in df or df[c].isna().all()]
        if empty:
            notes.append(f"{name}: คอลัมน์ว่าง {', '.join(empty)}")
        return df
    d = {n: rd(n) for n in SCHEMA}
    if d["chirps_monthly"]["rain_mm"].isna().all() and len(d["era5land_monthly"]):
        e = d["era5land_monthly"]
        d["chirps_monthly"] = e[["pcode", "date"]].assign(rain_mm=e["era_rain_m"] * 1000)
        notes.append("SPI ใช้ฝนจาก ERA5-Land แทน CHIRPS ชั่วคราว")
    if d["modis_lst_monthly"]["lst_c"].isna().all():
        notes.append("VHI ใช้ VCI (NDVI) อย่างเดียว เพราะยังไม่มีอุณหภูมิผิวดิน")
    if not len(d["s2_water_monthly"]):
        notes.append("ดัชนีน้ำกินยังไม่รวมพื้นที่ผิวน้ำ Sentinel-2")
    elif "mask_km2" not in d["s2_water_monthly"]:
        notes.append("พื้นที่ผิวน้ำ Sentinel-2 ยังวัดทั้งตำบล (ปนนาข้าว) จึงแสดงเป็นชั้นข้อมูลแต่ไม่นำเข้าดัชนีน้ำกิน")
    st = pd.read_csv(os.path.join(raw, "static_layers.csv"))
    if pop_factor != 1.0:
        st["pop_2020"] = st["pop_2020"] / pop_factor
        notes.append(f"ประชากรปรับแก้ ÷{pop_factor:g} (export ที่ scale 30 m นับซ้ำ)")
    d["static"] = st
    d["notes"] = notes
    return d


def make_mock(geo, seed=7):
    """Synthetic data with the same schema as the EE exports. FOR LAYOUT DEMO ONLY."""
    rng = np.random.default_rng(seed)
    feats = geo["features"]
    months = pd.date_range("1981-01-01", "2026-09-01", freq="MS")
    clim = np.array([10, 20, 40, 80, 190, 170, 190, 230, 250, 120, 25, 8])   # mm, northern-central TH pattern
    north = {f["properties"]["pcode"]: f["properties"].get("lat", 17.2) for f in feats}
    rows = {k: [] for k in ["chirps_monthly", "era5land_monthly", "modis_ndvi_monthly", "modis_lst_monthly", "s2_water_monthly"]}
    stat = []
    for f in feats:
        p = f["properties"]; pc = p["pcode"]
        hist = 1.0 if p.get("moo_total", 0) > 0 else 0.0
        bias = rng.normal(0, 0.08)
        for d in months:
            dry = 0
            if d.year == 2019 or (d.year == 2020 and d.month <= 5): dry = 0.35 + 0.25 * hist
            if d.year == 2015 or (d.year == 2016 and d.month <= 4): dry = 0.3
            if d.year == 2023 and d.month >= 6: dry = 0.25
            if d.year == 2026 and d.month >= 6: dry = 0.12 + 0.22 * hist + rng.normal(0, 0.1)
            r = max(0, clim[d.month - 1] * (1 - dry + bias) * rng.gamma(4, 0.25))
            base = dict(pcode=pc, tambon=p["tambon"], amphoe=p["amphoe"], date=d.strftime("%Y-%m"))
            rows["chirps_monthly"].append({**base, "rain_mm": r})
            wet = clim[d.month - 1] / 250
            rows["era5land_monthly"].append({**base, "sm_l1": 0.18 + 0.2 * wet * (1 - dry) + rng.normal(0, 0.015),
                "sm_l2": 0.22 + 0.15 * wet * (1 - 0.8 * dry) + rng.normal(0, 0.01), "era_rain_m": r / 1000,
                "pet_m": -(0.12 + 0.04 * dry + rng.normal(0, 0.01)), "et_m": -0.08, "t2m_k": 300 + 3 * dry})
            if d >= pd.Timestamp("2000-03-01"):
                rows["modis_ndvi_monthly"].append({**base, "ndvi": 0.45 + 0.3 * wet * (1 - dry) + rng.normal(0, 0.03),
                    "ndvi_crop": 0.4 + 0.35 * wet * (1 - 1.2 * dry) + rng.normal(0, 0.03)})
                rows["modis_lst_monthly"].append({**base, "lst_c": 30 + 6 * (1 - wet) + 5 * dry + rng.normal(0, 1)})
            if d >= pd.Timestamp("2017-04-01"):
                a = p.get("area_km2", 50)
                rows["s2_water_monthly"].append({**base, "water_km2": a * 0.02 * (0.5 + wet) * (1 - 1.3 * dry) + abs(rng.normal(0, 0.02)),
                    "clear_km2": a * (0.6 if d.month in (6, 7, 8, 9) else 0.95)})
        a = p.get("area_km2", 50)
        stat.append(dict(pcode=pc, pop_2020=a * rng.uniform(60, 250), crop_km2=a * rng.uniform(0.2, 0.7),
                         builtup_km2=a * 0.03, perm_water_km2=a * 0.005, any_water_km2=a * 0.03))
    d = {k: pd.DataFrame(v).assign(date=lambda x: pd.to_datetime(x["date"] + "-01")) for k, v in rows.items()}
    d["static"] = pd.DataFrame(stat)
    return d


# ---------- compute ----------
def compute(d, geo):
    hist = {f["properties"]["pcode"]: f["properties"].get("moo_total", 0) for f in geo["features"]}
    v2_water = "mask_km2" in d["s2_water_monthly"] and d["s2_water_monthly"]["mask_km2"].notna().any()
    area_by_pc = {f["properties"]["pcode"]: f["properties"].get("area_km2") or np.nan for f in geo["features"]}
    st = d["static"].set_index("pcode")
    out = []
    for pc in sorted(hist):
        g = lambda name, col: d[name][d[name].pcode == pc].set_index("date")[col].sort_index()
        rain = g("chirps_monthly", "rain_mm")
        df = pd.DataFrame(index=rain.index)
        df["rain_mm"] = rain
        for k in (1, 3, 6):
            df[f"spi{k}"] = spi_series(rain, k)
        era = d["era5land_monthly"][d["era5land_monthly"].pcode == pc].set_index("date").sort_index()
        bal = (era["era_rain_m"] + era["pet_m"]) * 1000              # pet is negative in ERA5 (downward +)
        df["spei3"] = empirical_z(bal.rolling(3, min_periods=3).sum())
        df["sm_z"] = empirical_z(era["sm_l1"])
        df["smdeep_z"] = empirical_z(era["sm_l2"])
        ndvi = g("modis_ndvi_monthly", "ndvi_crop").fillna(g("modis_ndvi_monthly", "ndvi"))
        df["vci"] = kogan(ndvi)
        df["tci"] = kogan(g("modis_lst_monthly", "lst_c"), invert=True)
        df["vhi"] = (0.5 * df["vci"] + 0.5 * df["tci"]).fillna(df["vci"])   # VCI only if LST missing
        w = d["s2_water_monthly"][d["s2_water_monthly"].pcode == pc].set_index("date").sort_index()
        area = area_by_pc.get(pc, np.nan)
        # need >= 50% of the tambon cloud-free that month, else the water fraction is too noisy
        if "mask_km2" in w and w["mask_km2"].notna().any():          # v2: water bodies only
            base_area = w["mask_km2"]
            frac = (w["water_km2"] / w["clear_km2"]).where((w["clear_km2"] >= 0.5 * base_area) & (base_area >= 0.01))
        else:
            frac = (w["water_km2"] / w["clear_km2"]).where(w["clear_km2"] >= 0.5 * area)
        df["water_z"] = empirical_z(frac)
        if not v2_water:      # whole-tambon water is dominated by paddy; show it but keep it out of the index
            water_for_index = pd.Series(np.nan, index=df.index)
        else:
            water_for_index = df["water_z"]
        df["veg_z"] = empirical_z(ndvi)
        df["water_z_idx"] = water_for_index
        ext_agri, ext_water = [], []
        for em, z in d.get("external", []):
            col = f"ext_{em['id']}"
            df[col] = z[pc].reindex(df.index) if pc in z.columns else np.nan
            (ext_agri if em["use_in"] == "agri" else ext_water if em["use_in"] == "water" else []).append(col)
        df["ext_agri"] = df[ext_agri].min(axis=1) if ext_agri else np.nan     # most severe external signal
        df["ext_water"] = df[ext_water].min(axis=1) if ext_water else np.nan
        df["agri_level"] = cdi_levels(df)
        df["water_haz"] = water_hazard_level(df)
        df["pcode"] = pc
        out.append(df.reset_index().rename(columns={"index": "date"}))
    return pd.concat(out, ignore_index=True)


# ---------- classification (literature-based, no arbitrary weights) ----------
# Agricultural drought: cascade of the EDO Combined Drought Indicator (Sepulcre-Canto et al. 2012, NHESS 12:3519)
#   1 Watch   : SPI-3 <= -1 or SPI-1 <= -2                       (rain deficit)
#   2 Warning : soil-moisture anomaly <= -1, with rain deficit or previous level >= 1
#   3 Alert   : vegetation anomaly <= -1, with SM deficit or previous level >= 2
def cdi_step(spi1, spi3, sm, veg, prev, ext=99):
    rain = (spi3 <= -1) or (spi1 <= -2) or (ext <= -1)    # ext: external index with use_in="agri"
    if veg <= -1 and (sm <= -1 or prev >= 2): return 3
    if sm <= -1 and (rain or prev >= 1): return 2
    if rain: return 1
    return 0


def cdi_levels(df):
    lv, prev = [], 0
    for _, r in df.iterrows():
        v = [r.get(k) for k in ("spi1", "spi3", "sm_z", "veg_z")] + [r.get("ext_agri", np.nan)]
        v = [99 if pd.isna(x) else x for x in v]
        prev = cdi_step(*v[:4], prev, v[4])
        lv.append(prev)
    return pd.Series(lv, index=df.index)


# Drinking-water hazard: count of hydrological deficits (SPI-6, deep soil moisture, water-body area when available)
#   0 none, 1 moderate (one deficit), 2 high (two or more)
def water_hazard_level(df):
    flags = (df["spi6"] <= -1).astype(int) + (df["smdeep_z"] <= -1).astype(int) + (df["water_z_idx"] <= -1).astype(int) \
        + (df.get("ext_water", pd.Series(np.nan, index=df.index)) <= -1).astype(int)
    return flags.clip(upper=2)


# Risk matrix (IPCC hazard x vulnerability). Vulnerability from DDPM villages in drought areas, 8 years 2560-2567:
#   0 = no year, 1 = one year, 2 = two or more years
WATER_MATRIX = {   # (vulnerability, hazard) -> 0 none, 1 watch, 3 prepare water delivery
    (0, 0): 0, (0, 1): 0, (0, 2): 1,
    (1, 0): 0, (1, 1): 1, (1, 2): 3,
    (2, 0): 1, (2, 1): 1, (2, 2): 3,
}


IRR_SHARE_MIN = 0.5   # tambon counted as irrigated when >= 50% of its cropland is inside RID irrigation areas


def alert(agri_level, water_haz, vuln, crop_frac, irr_share=None):
    # inside irrigation areas farmers can still get canal water: farmer warning needs one CDI level more
    if irr_share is not None and irr_share >= IRR_SHARE_MIN:
        agri_level = max(0, agri_level - 1)
    w = WATER_MATRIX[(vuln, int(water_haz))]
    if w == 3: return 3
    if agri_level >= 2 and crop_frac >= 0.10: return 2
    if agri_level >= 1 or w == 1: return 1
    return 0


# ---------- write ----------
def write(ts, geo, d, out, mock, vuln, irr=None, ddpm_risk=None):
    irr = irr or {}; ddpm_risk = ddpm_risk or {}
    os.makedirs(out, exist_ok=True)
    st = d["static"].set_index("pcode")
    ts = ts.set_index(["pcode", "date"]).sort_index()
    cols = ["rain_mm", "spi1", "spi3", "spi6", "spei3", "sm_z", "smdeep_z", "veg_z", "vci", "tci", "vhi", "water_z"]
    latest_month = {c: ts[c].dropna().index.get_level_values("date").max() for c in cols}
    now = min(latest_month["spi3"], latest_month["sm_z"], latest_month["veg_z"])
    ext_cols = [f"ext_{em['id']}" for em, _ in d.get("external", [])]
    zc = ["spi1", "spi3", "spi6", "sm_z", "smdeep_z", "veg_z", "water_z_idx", "ext_agri", "ext_water"] + ext_cols
    series, summary = {}, []
    for f in geo["features"]:
        p = f["properties"]; pc = p["pcode"]
        t = ts.loc[pc]
        cur = t.loc[now]
        area = p.get("area_km2") or 1
        crop_frac = float(st.loc[pc, "crop_km2"] / area) if pc in st.index else 0
        v = vuln.get(pc, {"level": 0, "years": 0, "villages": 0})
        # December outlook: each anomaly = half persistence of now + half the mean December anomaly of El Nino analog years
        dec = t[(t.index.month == 12) & (t.index.year.isin(ANALOG_YEARS))]
        zd = {c: (0.5 * cur[c] + 0.5 * dec[c].mean()) if pd.notna(cur[c]) else dec[c].mean() for c in zc}
        zd = {c: (99 if pd.isna(x) else x) for c, x in zd.items()}
        agri_dec = cdi_step(zd["spi1"], zd["spi3"], zd["sm_z"], zd["veg_z"], int(cur["agri_level"]), zd["ext_agri"])
        haz_dec = min(2, int(zd["spi6"] <= -1) + int(zd["smdeep_z"] <= -1) + int(zd["water_z_idx"] <= -1) + int(zd["ext_water"] <= -1))
        # short-range outlook, 1 and 2 months ahead: persistence of the latest anomaly blended with the El Nino analog
        # years' anomaly for the same target month; persistence weight 0.7 (+1 month) and 0.5 (+2 months)
        fc, prev_lv = {}, int(cur["agri_level"])
        for k, w in ((1, 0.7), (2, 0.5)):
            tgt = now + pd.DateOffset(months=k)
            yrs = [y + (tgt.year - now.year) for y in ANALOG_YEARS]
            an = t[(t.index.month == tgt.month) & (t.index.year.isin(yrs))]
            zf = {c: (w * cur[c] + (1 - w) * an[c].mean()) if pd.notna(cur[c]) else an[c].mean() for c in zc}
            zf = {c: (99 if pd.isna(x) else x) for c, x in zf.items()}
            ag = cdi_step(zf["spi1"], zf["spi3"], zf["sm_z"], zf["veg_z"], prev_lv, zf["ext_agri"])
            hz = min(2, int(zf["spi6"] <= -1) + int(zf["smdeep_z"] <= -1) + int(zf["water_z_idx"] <= -1) + int(zf["ext_water"] <= -1))
            fc[k] = {"z": zf, "agri": ag, "haz": hz, "alert": alert(ag, hz, v["level"], crop_frac, irr.get(pc))}
            prev_lv = ag
        a_now = alert(int(cur["agri_level"]), int(cur["water_haz"]), v["level"], crop_frac, irr.get(pc))
        a_dec = alert(agri_dec, haz_dec, v["level"], crop_frac)
        r = lambda x, n=2: None if pd.isna(x) else round(float(x), n)
        p.update({
            "pop_2020": r(st.loc[pc, "pop_2020"], 0) if pc in st.index else None,
            "crop_frac": r(crop_frac), "alert_now": a_now, "alert_dec": a_dec,
            "alert_f1": fc[1]["alert"], "alert_f2": fc[2]["alert"], "agri_f1": fc[1]["agri"], "agri_f2": fc[2]["agri"],
            "haz_f1": fc[1]["haz"], "haz_f2": fc[2]["haz"],
            **{f"{c}_f{k}": (r(fc[k]["z"][c]) if fc[k]["z"][c] != 99 else None) for k in (1, 2) for c in ["spi3", "sm_z", "veg_z"] + ext_cols},
            "agri_now": int(cur["agri_level"]), "agri_dec": agri_dec,
            "haz_now": int(cur["water_haz"]), "haz_dec": haz_dec,
            "vuln": v["level"], "vuln_years": v["years"], "vuln_villages": v["villages"],
            "irr_share": r(irr.get(pc)) if pc in irr else None,
            **ddpm_risk.get(pc, {}),
            **{f"{c}_now": r(cur[c]) for c in ["spi3", "spi6", "sm_z", "veg_z", "vhi", "water_z"]},
            **{f"{c}_dec": r(zd[c]) if zd[c] != 99 else None for c in ["spi3", "spi6", "sm_z", "veg_z"]},
            **{f"{c}_now": r(cur[c]) for c in ext_cols},
            **{f"{c}_dec": r(zd[c]) if zd[c] != 99 else None for c in ext_cols},
        })
        sub = t[t.index >= "2015-01-01"]
        series[pc] = {"dates": [x.strftime("%Y-%m") for x in sub.index],
                      **{c: [r(x) for x in sub[c]] for c in ["rain_mm", "spi3", "spi6", "sm_z", "veg_z", "vhi", "water_z"]},
                      **{c: [r(x) for x in sub[c]] for c in ext_cols},
                      "agri_level": [int(x) for x in sub["agri_level"]], "water_haz": [int(x) for x in sub["water_haz"]]}
        summary.append({k: p.get(k) for k in ["pcode", "amphoe", "tambon", "alert_now", "alert_f1", "alert_f2", "agri_now", "agri_f1",
                                              "haz_now", "haz_f1", "vuln", "vuln_years", "pop_2020"]})
    meta = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "mock": mock,
            "now_month": now.strftime("%Y-%m"),
            "latest_by_indicator": {c: (None if pd.isna(latest_month[c]) else latest_month[c].strftime("%Y-%m")) for c in cols},
            "analog_years": ANALOG_YEARS, "method": "CDI cascade + hazard x vulnerability matrix",
            "alert_counts_now": pd.Series([x["alert_now"] for x in summary]).value_counts().to_dict(),
            "forecast_months": [(now + pd.DateOffset(months=k)).strftime("%Y-%m") for k in (1, 2)],
            "alert_counts_f1": pd.Series([x["alert_f1"] for x in summary]).value_counts().to_dict(),
            "alert_counts_f2": pd.Series([x["alert_f2"] for x in summary]).value_counts().to_dict(),
            "notes": d.get("notes", []),
            "external": [{"id": em["id"], "name": em["name"], "units": em["units"], "kind": em["kind"], "use_in": em["use_in"],
                          "source": em.get("source", ""), "contact": em.get("contact", ""),
                          "first": em["report"]["first"], "last": em["report"]["last"], "tambons": em["report"]["tambons"],
                          "backtest": external_backtest(ts.reset_index(), f"ext_{em['id']}")} for em, _ in d.get("external", [])]}
    meta = json.loads(json.dumps(meta, default=int))
    json.dump(geo, open(os.path.join(out, "tambons.geojson"), "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(series, open(os.path.join(out, "timeseries.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(summary).sort_values(["alert_now", "alert_f1", "alert_f2", "vuln"], ascending=False).to_csv(
        os.path.join(out, "tambon_alerts.csv"), index=False, encoding="utf-8-sig")
    return meta


def external_backtest(ts, col):
    """Share of tambons with the external index <= -1, mean over each Nov-Mar dry season."""
    if col not in ts or ts[col].isna().all():
        return {}
    x = ts[["date", col]].dropna()
    prov = x.groupby("date")[col].apply(lambda v: float((v <= -1).mean()))
    res = {}
    for y in range(prov.index.min().year, prov.index.max().year + 1):
        sel = prov[(prov.index >= f"{y}-11-01") & (prov.index <= f"{y+1}-03-01")]
        if len(sel):
            res[f"{y+543-2500:02d}/{y+544-2500:02d}"] = round(float(sel.mean()), 3)
    return res


def load_irrigation(path):
    """Optional tambon irrigation share. CSV with pcode (or tambon_code) and irr_share 0-1 (or irr_pct 0-100)."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"pcode": str, "tambon_code": str})
    if "pcode" not in df and "tambon_code" in df:
        df["pcode"] = "TH" + df["tambon_code"].str.zfill(6)
    if "irr_share" not in df and "irr_pct" in df:
        df["irr_share"] = df["irr_pct"] / 100
    df["pcode"] = df["pcode"].str.upper().where(df["pcode"].str.upper().str.startswith("TH"), "TH" + df["pcode"])
    print(f"[irrigation] {path}: {len(df)} ตำบล, ในเขตชลประทาน >= {IRR_SHARE_MIN:.0%}: {(df.irr_share >= IRR_SHARE_MIN).sum()}")
    return dict(zip(df["pcode"], df["irr_share"].clip(0, 1)))


def load_ddpm_risk(path):
    """DDPM drought_region village risk (gis-portal.disaster.go.th) summarised per tambon."""
    if not path or not os.path.exists(path):
        return {}
    P = json.load(open(path, encoding="utf-8"))
    out = {}
    for p in P:
        pc = "TH" + str(p["sub_district_code"])
        o = out.setdefault(pc, {"dv_n": 0, "dv_high": 0, "dv_drink": 0, "dv_short": 0, "dv_max": 0})
        o["dv_n"] += 1
        o["dv_high"] += int((p["risk_level_value"] or 0) >= 3)
        o["dv_drink"] += int(p["impactall_drinking_water"] == 1)
        o["dv_short"] += int(p["impactall_hardness"] == "มักจะขาดแคลนน้ำ")
        o["dv_max"] = max(o["dv_max"], p["risk_level_value"] or 0)
    return out


def reservoirs_overlay(path, out):
    """raw/reservoirs.csv (name, lat, lon, capacity_mcm, storage_mcm, date[, type, tambon, amphoe]) -> data/overlays/reservoirs.geojson"""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["lat", "lon"])
    feats = []
    for _, r in df.iterrows():
        pct = None
        if pd.notna(r.get("storage_mcm")) and pd.notna(r.get("capacity_mcm")) and r["capacity_mcm"] > 0:
            pct = round(float(r["storage_mcm"]) / float(r["capacity_mcm"]) * 100, 1)
        props = {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in ["name", "type", "tambon", "amphoe", "capacity_mcm", "storage_mcm", "date",
                                                                          "storage_prev_mcm", "usable_mcm", "source"] if k in r}
        props["pct"] = pct
        if props.get("storage_prev_mcm") is not None and pct is not None:   # same date last year (RID reports it)
            props["pct_prev"] = round(float(r["storage_prev_mcm"]) / float(r["capacity_mcm"]) * 100, 1)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(r["lon"]), 5), round(float(r["lat"]), 5)]},
                      "properties": json.loads(json.dumps(props, default=str))})
    os.makedirs(os.path.join(out, "overlays"), exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(out, "overlays", "reservoirs.geojson"), "w"), ensure_ascii=False)
    print(f"[reservoirs] {len(feats)} แห่ง -> overlays/reservoirs.geojson")
    return len(feats)


def load_vuln(path):
    """DDPM dpm-gd016 tambon summary: years with villages in drought areas, 2560-2567."""
    if not path or not os.path.exists(path):
        return {}
    v = pd.read_csv(path)
    out = {}
    for _, row in v.iterrows():
        pc = "TH" + str(row["รหัสตำบล"])
        yrs = int(row["years_any"])
        out[pc] = {"level": 0 if yrs == 0 else 1 if yrs == 1 else 2, "years": yrs, "villages": int(row["villages"])}
    return out


def validate(ts, geo):
    """Do the levels pick out DDPM-declared tambons of 2019/20? AUC (0.5 = chance) and declared share per level."""
    lab = {f["properties"]["pcode"]: f["properties"].get("moo_total", 0) > 0 for f in geo["features"]}
    res = {}
    for month in ["2019-06", "2019-12", "2020-03"]:
        x = ts[ts.date == pd.Timestamp(month + "-01")].set_index("pcode")
        for c in ["agri_level", "water_haz"] + [k for k in x.columns if k.startswith("ext_") and k not in ("ext_agri", "ext_water")]:
            y = x.index.map(lab); s_ = -x[c] if c.startswith("ext_") else x[c]   # ext: lower = drier
            if s_.isna().all():
                continue
            pos, neg = s_[y], s_[~y]
            if len(pos) and len(neg):
                auc = (pos.values[:, None] > neg.values[None, :]).mean() + 0.5 * (pos.values[:, None] == neg.values[None, :]).mean()
                res[f"{month} {c}"] = {"auc": round(float(auc), 2)} if c.startswith("ext_") else \
                    {"auc": round(float(auc), 2), "levels_all": x[c].value_counts().sort_index().to_dict()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw"); ap.add_argument("--boundaries", required=True)
    ap.add_argument("--out", default="data"); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--vuln", default="data/source/sukhothai_ddpm_risk_villages_2560-2567_tambon.csv")
    ap.add_argument("--irrigation", default="raw/irrigation_tambon.csv", help="optional tambon irrigation share CSV")
    ap.add_argument("--ddpm-risk", default="data/source/ddpm_drought_region_villages_64.json")
    ap.add_argument("--reservoirs", default="raw/reservoirs.csv", help="optional reservoir list with storage (from RID)")
    ap.add_argument("--external", default="raw/external", help="folder of external indices (one sub-folder each)")
    ap.add_argument("--pop-factor", type=float, default=1.0, help="divide population by this (fix for old static export)")
    a = ap.parse_args()
    geo = json.load(open(a.boundaries))
    d = make_mock(geo) if a.mock else load_raw(a.raw, a.pop_factor)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from external_intake import load_external
    d["external"] = load_external(a.external, geo)
    ts = compute(d, geo)
    print("validation AUC vs DDPM 2019/20 (0.5 = no skill):", validate(ts, geo))
    reservoirs_overlay(a.reservoirs, a.out)
    irr = load_irrigation(a.irrigation)
    d.setdefault("notes", [])
    if not irr:
        d["notes"].append("ยังไม่มีขอบเขตพื้นที่ชลประทาน ระดับเตือนเกษตรยังไม่ได้ปรับตามเขตชลประทาน")
    meta = write(ts, geo, d, a.out, a.mock, load_vuln(a.vuln), irr, load_ddpm_risk(a.ddpm_risk))
    print(json.dumps(meta, ensure_ascii=False, indent=1))
