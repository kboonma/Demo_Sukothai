"""
ESP-style probabilistic drought outlook (Ensemble Streamflow Prediction idea, Day 1985; used for drought by
e.g. the US NIDIS / NOAA and in many SPI outlook studies).

For each tambon, start from the latest observed state and append each historical year's next months ("traces"):
  rain        observed months + the trace year's rain for the target months -> SPI-1/3/6 with the same gamma fits
  soil / veg  damped persistence + trace departure: z_f = rho*z_now + (z_trace(target) - rho*z_trace(issue))
              rho = lag-h correlation of that tambon's anomaly for the issue month (AR(1)-style ESP)
              (vegetation before 2001 has no MODIS data -> mean departure of the other traces)
Each trace then runs through the same CDI cascade, drinking-water hazard and alert rules as the monitor.
P(level) = share of traces reaching the level. Two ensembles:
  all   every year 1981-2025
  nino  only years with El Niño in Sep-Nov (ONI >= +0.5): the conditional outlook used when El Niño is on.

Hindcast: issue every September 1981-2024, leave the issue year out of its own ensemble, score
Oct-Jan against what the monitor itself computed (Brier skill vs climatology, ROC AUC).
"""
import numpy as np, pandas as pd
from scipy import stats

# Sep-Nov ONI >= +0.5 (NOAA CPC ONI table). 2026 itself is El Niño (CPC advisory, Oct 2026).
NINO_YEARS = [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2023]
BASE = (1991, 2020)
H = 4   # months ahead
DAMP = 0.25  # rho_eff = rho ** DAMP ; 0 = no damping (pure anomaly trajectory), 1 = full AR damping


def gamma_params(rain, k):
    """Per calendar month: (q0, shape, scale) of k-month rain totals, baseline 1991-2020 (same as the monitor)."""
    acc = rain.rolling(k, min_periods=k).sum()
    par = {}
    for m in range(1, 13):
        sel = acc[acc.index.month == m].dropna()
        base = sel[(sel.index.year >= BASE[0]) & (sel.index.year <= BASE[1])]
        if len(base) < 15:
            continue
        q0 = (base == 0).mean(); nz = base[base > 0]
        a, _, sc = stats.gamma.fit(nz, floc=0)
        par[m] = (q0, a, sc)
    return par


class Tambon:
    """One tambon's monthly series as arrays (fast lookups) plus fitted SPI parameters."""
    def __init__(self, t):
        t = t.sort_index()
        self.t = t
        self.idx = {d: i for i, d in enumerate(t.index)}
        self.rain = t["rain_mm"].to_numpy(float)
        self.z = {c: t[c].to_numpy(float) for c in ("sm_z", "smdeep_z", "veg_z")}
        self.agri = t["agri_level"].to_numpy(int); self.haz = t["water_haz"].to_numpy(int)
        self.par = {k: gamma_params(t["rain_mm"], k) for k in (1, 3, 6)}
        self._rho = {}

    def rho(self, c, issue_month, h):
        """Lag-h correlation of anomaly c between the issue month and h months later, over all years (>= 0)."""
        key = (c, issue_month, h)
        if key not in self._rho:
            x = self.t[c]; a = x[x.index.month == issue_month]
            b = x.shift(-h)[x.index.month == issue_month]
            ok = a.notna() & b.notna()
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 8 else 0.0
            self._rho[key] = max(0.0, r if np.isfinite(r) else 0.0)
        return self._rho[key] ** DAMP

    def _spi(self, totals, month, k):
        if month not in self.par[k]:
            return np.full(len(totals), np.nan)
        q0, a, sc = self.par[k][month]
        p = q0 + (1 - q0) * stats.gamma.cdf(np.clip(totals, 0, None), a, scale=sc)
        return stats.norm.ppf(np.clip(p, 0.001, 0.999))


def ensemble(tb, issue, years, cdi_step, alert_fn, horizon=H, exclude=None):
    """All traces at once. -> dict h -> {"agri","haz","alert"} arrays over usable traces."""
    i0 = tb.idx.get(issue)
    empty = {h: {"agri": np.array([], int), "haz": np.array([], int), "alert": np.array([], int)} for h in range(1, horizon + 1)}
    if i0 is None or i0 < 6:
        return empty
    ys = [y for y in years if y != exclude]
    # trace index of the issue month in each trace year; keep traces whose target months exist
    ti = np.array([tb.idx.get(issue + pd.DateOffset(years=y - issue.year), -1) for y in ys])
    ok = (ti >= 0) & (ti + horizon < len(tb.rain))
    ti = ti[ok]
    if not len(ti):
        return empty
    n = len(ti)
    hist = np.tile(tb.rain[i0 - 5:i0 + 1], (n, 1))                    # last 6 observed months
    fut = np.stack([tb.rain[ti + h] for h in range(1, horizon + 1)], axis=1)
    seq = np.concatenate([hist, fut], axis=1)                          # n x (6 + horizon)
    good = ~np.isnan(fut).any(axis=1)
    z_now = {c: tb.z[c][i0] for c in tb.z}
    prev = np.full(n, tb.agri[i0])
    res = {}
    for h in range(1, horizon + 1):
        tgt = issue + pd.DateOffset(months=h); j = 5 + h
        spi = {k: tb._spi(seq[:, j - k + 1:j + 1].sum(axis=1), tgt.month, k) for k in (1, 3, 6)}
        z = {}
        for c in tb.z:
            # damped persistence + the trace year's own departure from that persistence (AR-conditioned ESP):
            #   z_f = rho*z_now + (z_trace(target) - rho*z_trace(issue))
            a, b = tb.z[c][ti + h], tb.z[c][ti]
            r = tb.rho(c, issue.month, h)
            if np.isnan(z_now[c]):
                z[c] = np.full(n, np.nan)
            else:
                eps = np.where(np.isnan(a) | np.isnan(b), np.nan, a - r * b)
                if np.isnan(eps).all():
                    eps = np.zeros(n)
                else:                                                   # MODIS < 2001: draw from the other traces
                    eps = np.where(np.isnan(eps), np.nanmean(eps), eps)
                z[c] = np.clip(r * z_now[c] + eps, -3, 3)
        v = lambda x: np.where(np.isnan(x), 99.0, x)
        s1, s3, s6, sm, sd, vg = v(spi[1]), v(spi[3]), v(spi[6]), v(z["sm_z"]), v(z["smdeep_z"]), v(z["veg_z"])
        agri = np.array([cdi_step(s1[i], s3[i], sm[i], vg[i], prev[i]) for i in range(n)])
        haz = np.minimum(2, (s6 <= -1).astype(int) + (sd <= -1).astype(int))
        al = np.array([alert_fn(int(agri[i]), int(haz[i]), tgt) for i in range(n)])
        res[h] = {"agri": agri[good], "haz": haz[good], "alert": al[good]}
        prev = agri
    return res


def probs(e):
    n = len(e["alert"])
    if not n:
        return None
    return {"n": n, "p_watch": round(float((e["alert"] >= 1).mean()), 2), "p_act": round(float((e["alert"] >= 2).mean()), 2),
            "p_water": round(float((e["alert"] == 3).mean()), 2), "p_cdi2": round(float((e["agri"] >= 2).mean()), 2),
            "p_haz2": round(float((e["haz"] >= 2).mean()), 2)}


# ---------- skill ----------
def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    return float(stats.mannwhitneyu(p[y], p[~y]).statistic / (y.sum() * (~y).sum()))


def brier_skill(p, y, p_ref):
    p, y, r = np.asarray(p, float), np.asarray(y, float), np.asarray(p_ref, float)
    bs, bref = np.mean((p - y) ** 2), np.mean((r - y) ** 2)
    return None if bref == 0 else float(1 - bs / bref)


def hindcast(tambons, cdi_step, alert_for, years, issue_month=9, horizon=H):
    """
    tambons: dict pc -> Tambon ; alert_for(pc) -> alert_fn(agri, haz, tgt) with historical rules.
    Returns skill table per lead (h) for events: action alert (>=2) and CDI Warning (>=2), ESP-all vs ESP-ENSO.
    """
    rows = []
    for pc, tb in tambons.items():
        afn = alert_for(pc)
        t = tb.t
        for y0 in years:
            issue = pd.Timestamp(year=y0, month=issue_month, day=1)
            if issue not in t.index:
                continue
            e_all = ensemble(tb, issue, years, cdi_step, afn, horizon, exclude=y0)
            same = [y for y in years if (y in NINO_YEARS) == (y0 in NINO_YEARS)]
            e_enso = ensemble(tb, issue, same, cdi_step, afn, horizon, exclude=y0)
            for h in range(1, horizon + 1):
                tgt = issue + pd.DateOffset(months=h)
                if tgt not in t.index or not len(e_all[h]["alert"]):
                    continue
                ob = t.loc[tgt]
                obs_alert = afn(int(ob["agri_level"]), int(ob["water_haz"]), tgt)
                rows.append({"pcode": pc, "year": y0, "h": h, "tgt_month": tgt.month,
                             "obs_act": obs_alert >= 2, "obs_cdi2": int(ob["agri_level"]) >= 2,
                             "p_act_all": (e_all[h]["alert"] >= 2).mean(), "p_cdi2_all": (e_all[h]["agri"] >= 2).mean(),
                             "p_act_enso": (e_enso[h]["alert"] >= 2).mean() if len(e_enso[h]["alert"]) else np.nan,
                             "p_cdi2_enso": (e_enso[h]["agri"] >= 2).mean() if len(e_enso[h]["agri"]) else np.nan,
                             "persist_act": float(afn(int(t.loc[issue, "agri_level"]), int(t.loc[issue, "water_haz"]), tgt) >= 2),
                             "persist_cdi2": float(int(t.loc[issue, "agri_level"]) >= 2)})
    df = pd.DataFrame(rows)
    # climatology reference: frequency of the event in that tambon and calendar month in the other years
    for ev in ("act", "cdi2"):
        g = df.groupby(["pcode", "h"])[f"obs_{ev}"]
        tot, cnt = g.transform("sum"), g.transform("count")
        df[f"clim_{ev}"] = (tot - df[f"obs_{ev}"]) / (cnt - 1)
    return df


def skill_table(df):
    out = {}
    for h, g in df.groupby("h"):
        r = {}
        for ev in ("act", "cdi2"):
            y = g[f"obs_{ev}"]
            gg = g.dropna(subset=[f"p_{ev}_enso"])
            r[ev] = {"base_rate": round(float(y.mean()), 3),
                     "auc_all": _r(auc(g[f"p_{ev}_all"], y)), "auc_enso": _r(auc(gg[f"p_{ev}_enso"], gg[f"obs_{ev}"])),
                     "bss_all": _r(brier_skill(g[f"p_{ev}_all"], y, g[f"clim_{ev}"])),
                     "bss_enso": _r(brier_skill(gg[f"p_{ev}_enso"], gg[f"obs_{ev}"], gg[f"clim_{ev}"])),
                     "bss_persist": _r(brier_skill(g[f"persist_{ev}"], y, g[f"clim_{ev}"]))}
        out[int(h)] = r
    return out


def _r(x, n=2):
    return None if x is None or pd.isna(x) else round(float(x), n)
