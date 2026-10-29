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
import argparse, json, os, datetime
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
def load_raw(raw):
    def rd(name):
        df = pd.read_csv(os.path.join(raw, f"{name}.csv"))
        df["date"] = pd.to_datetime(df["date"] + "-01")
        return df
    d = {n: rd(n) for n in ["chirps_monthly", "era5land_monthly", "modis_ndvi_monthly",
                             "modis_lst_monthly", "s2_water_monthly"]}
    d["static"] = pd.read_csv(os.path.join(raw, "static_layers.csv"))
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
        df["vhi"] = 0.5 * df["vci"] + 0.5 * df["tci"]
        w = d["s2_water_monthly"][d["s2_water_monthly"].pcode == pc].set_index("date").sort_index()
        frac = (w["water_km2"] / w["clear_km2"]).where(w["clear_km2"] > 0)
        df["water_z"] = empirical_z(frac)
        h = min(hist[pc] / 10.0, 1.0)                                # villages declared, capped at 10
        df["agri_index"] = wmean([(0.25, sev_z(df.spi3)), (0.30, sev_z(df.sm_z)),
                                  (0.30, sev_vhi(df.vhi)), (0.15, sev_z(df.spei3))])
        df["water_index"] = wmean([(0.35, sev_z(df.water_z)), (0.25, sev_z(df.spi6)),
                                   (0.15, sev_z(df.smdeep_z)), (0.25, pd.Series(h, index=df.index))])
        df["pcode"] = pc
        out.append(df.reset_index().rename(columns={"index": "date"}))
    return pd.concat(out, ignore_index=True)


def alert(row, crop_frac):
    if row["water_index"] >= 0.60: return 3
    if row["agri_index"] >= 0.50 and crop_frac >= 0.10: return 2
    if max(row["water_index"], row["agri_index"]) >= 0.30: return 1
    return 0


def latest_valid(s):
    s = s.dropna()
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, np.nan)


# ---------- write ----------
def write(ts, geo, d, out, mock):
    os.makedirs(out, exist_ok=True)
    st = d["static"].set_index("pcode")
    ts = ts.set_index(["pcode", "date"]).sort_index()
    cols = ["rain_mm", "spi1", "spi3", "spi6", "spei3", "sm_z", "smdeep_z", "vci", "tci", "vhi", "water_z",
            "agri_index", "water_index"]
    latest_month = {c: ts[c].dropna().index.get_level_values("date").max() for c in cols}
    # "now" = latest month in which both composites exist
    now = min(latest_month["agri_index"], latest_month["water_index"])
    series, summary = {}, []
    for f in geo["features"]:
        p = f["properties"]; pc = p["pcode"]
        t = ts.loc[pc]
        cur = t.loc[now] if now in t.index else t.iloc[-1]
        area = p.get("area_km2") or 1
        crop_frac = float(st.loc[pc, "crop_km2"] / area) if pc in st.index else 0
        # December outlook: half persistence of current state, half analog-year December anomaly
        dec = t[(t.index.month == 12) & (t.index.year.isin(ANALOG_YEARS))]
        o_agri = 0.5 * cur["agri_index"] + 0.5 * dec["agri_index"].mean()
        o_water = 0.5 * cur["water_index"] + 0.5 * dec["water_index"].mean()
        a_now = alert(cur, crop_frac)
        a_dec = alert({"agri_index": o_agri, "water_index": o_water}, crop_frac)
        r = lambda v, n=2: None if pd.isna(v) else round(float(v), n)
        p.update({
            "pop_2020": r(st.loc[pc, "pop_2020"], 0) if pc in st.index else None,
            "crop_frac": r(crop_frac), "alert_now": a_now, "alert_dec": a_dec,
            "agri_now": r(cur["agri_index"]), "water_now": r(cur["water_index"]),
            "agri_dec": r(o_agri), "water_dec": r(o_water),
            **{f"{c}_now": r(cur[c]) for c in ["spi3", "spi6", "sm_z", "vhi", "water_z"]},
        })
        sub = t[t.index >= "2018-01-01"]
        series[pc] = {"dates": [x.strftime("%Y-%m") for x in sub.index],
                      **{c: [r(v) for v in sub[c]] for c in ["rain_mm", "spi3", "sm_z", "vhi", "water_z", "agri_index", "water_index"]}}
        summary.append({k: p.get(k) for k in ["pcode", "amphoe", "tambon", "alert_now", "alert_dec", "agri_now", "water_now",
                                              "agri_dec", "water_dec", "moo_total", "pop_2020"]})
    meta = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "mock": mock,
            "now_month": now.strftime("%Y-%m"),
            "latest_by_indicator": {c: latest_month[c].strftime("%Y-%m") for c in cols},
            "analog_years": ANALOG_YEARS,
            "alert_counts_now": pd.Series([s["alert_now"] for s in summary]).value_counts().to_dict(),
            "alert_counts_dec": pd.Series([s["alert_dec"] for s in summary]).value_counts().to_dict()}
    meta = json.loads(json.dumps(meta, default=int))
    json.dump(geo, open(os.path.join(out, "tambons.geojson"), "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(series, open(os.path.join(out, "timeseries.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(summary).sort_values(["alert_dec", "water_dec"], ascending=False).to_csv(
        os.path.join(out, "tambon_alerts.csv"), index=False, encoding="utf-8-sig")
    return meta


def validate(ts, geo):
    """How well do the indices separate DDPM-declared tambons in the 2019/20 event? (AUC)"""
    lab = {f["properties"]["pcode"]: f["properties"].get("moo_total", 0) > 0 for f in geo["features"]}
    res = {}
    for month in ["2019-06", "2019-12", "2020-03"]:
        x = ts[ts.date == pd.Timestamp(month + "-01")].set_index("pcode")
        for c in ["agri_index", "water_index"]:
            y = x.index.map(lab); s = x[c]
            pos, neg = s[y], s[~y]
            if len(pos) and len(neg):
                auc = (pos.values[:, None] > neg.values[None, :]).mean() + 0.5 * (pos.values[:, None] == neg.values[None, :]).mean()
                res[f"{month} {c}"] = round(float(auc), 2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw"); ap.add_argument("--boundaries", required=True)
    ap.add_argument("--out", default="data"); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    geo = json.load(open(a.boundaries))
    d = make_mock(geo) if a.mock else load_raw(a.raw)
    ts = compute(d, geo)
    print("validation AUC vs DDPM 2019/20 (0.5 = no skill):", validate(ts, geo))
    meta = write(ts, geo, d, a.out, a.mock)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
