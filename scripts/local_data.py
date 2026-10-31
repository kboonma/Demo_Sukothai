"""
Local data the province holds (all optional; the pipeline runs without them and says what is missing).
Files live in raw/local/ ; templates in intake/local/ ; spec in intake/INTAKE.md.

  water_supply.csv     tap-water coverage per tambon (PWA / local water supply)       -> exposure + vulnerability
  rid_allocation.csv   RID dry-season allocation plan per tambon (napr = นาปรัง)       -> farmer-warning rule in irrigated tambons
  impact_reports.csv   water requests / water-truck trips / crop damage / declarations -> system KPIs (hit rate, lead time)
  outlook.json         ENSO, TMD seasonal rain outlook, crop calendar                 -> header outlook + timing of warnings

DDPM drought declarations 2562-2563 (data/source/ddpm_sukhothai_records.csv) are always used as impact events,
so the KPIs exist before any local report arrives.
"""
import csv, json, os, re
import numpy as np, pandas as pd

TH_MONTH = {"ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12}
PREFIX = re.compile(r"^\s*(ต\.|ตำบล|อ\.|อำเภอ)\s*")
ALLOC_OK = {"full": "full", "ปกติ": "full", "เต็ม": "full", "reduced": "reduced", "จำกัด": "reduced", "ลดลง": "reduced",
            "none": "none", "ไม่สนับสนุน": "none", "งด": "none", "ไม่มี": "none"}
NO_TAP_HIGH = 0.30   # tambon counted as more vulnerable when >= 30% of households have no piped water


def _clean(s):
    return PREFIX.sub("", str(s or "")).strip().replace(" ", "")


def _keys(geo):
    by_name = {(_clean(f["properties"]["amphoe"]), _clean(f["properties"]["tambon"])): f["properties"]["pcode"] for f in geo["features"]}
    by_t = {}
    for f in geo["features"]:
        by_t.setdefault(_clean(f["properties"]["tambon"]), []).append(f["properties"]["pcode"])
    valid = {f["properties"]["pcode"] for f in geo["features"]}
    return by_name, by_t, valid


def _pcode(row, geo_keys):
    by_name, by_t, valid = geo_keys
    pc = str(row.get("pcode") or row.get("tambon_code") or "").strip().upper()
    if pc:
        pc = pc if pc.startswith("TH") else "TH" + pc.zfill(6)
        return pc if pc in valid else None
    t, a = _clean(row.get("tambon")), _clean(row.get("amphoe"))
    return by_name.get((a, t)) or (by_t[t][0] if len(by_t.get(t, [])) == 1 else None)


def _read(path):
    for enc in ("utf-8-sig", "cp874"):
        try:
            with open(path, encoding=enc) as fh:
                return [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(fh)]
        except UnicodeDecodeError:
            continue
    return []


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _date(s):
    """2019-12-25, 2019-12, 25/12/2019, 25/12/2562, 4-มิ.ย.-62 -> Timestamp (first of month)."""
    s = str(s or "").strip()
    m = re.match(r"(\d{1,2})-(\S+?)-(\d{2,4})$", s)
    if m and m.group(2) in TH_MONTH:
        y = int(m.group(3)); y = y + 2500 if y < 100 else y
        return pd.Timestamp(year=y - 543, month=TH_MONTH[m.group(2)], day=1)
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        y = int(m.group(3)); y = y - 543 if y > 2400 else y
        return pd.Timestamp(year=y, month=int(m.group(2)), day=1)
    m = re.match(r"(\d{4})-(\d{1,2})", s)
    if m:
        y = int(m.group(1)); y = y - 543 if y > 2400 else y
        return pd.Timestamp(year=y, month=int(m.group(2)), day=1)
    return None


# ---------- 1. tap-water coverage ----------
def load_water_supply(path, geo):
    """pcode|tambon+amphoe, households, hh_no_tap (or tap_pct), [villages_no_tap], [source], [date]."""
    if not path or not os.path.exists(path):
        return {}, None
    keys, out, bad = _keys(geo), {}, 0
    for r in _read(path):
        pc = _pcode(r, keys)
        hh = _num(r.get("households"))
        nt = _num(r.get("hh_no_tap"))
        if nt is None and _num(r.get("tap_pct")) is not None and hh:
            nt = hh * (1 - _num(r.get("tap_pct")) / 100)
        if not pc or hh is None or nt is None:
            bad += 1; continue
        out[pc] = {"hh": int(hh), "hh_no_tap": int(round(nt)), "no_tap_share": round(nt / hh, 3) if hh else None,
                   "villages_no_tap": int(_num(r.get("villages_no_tap")) or 0) if r.get("villages_no_tap") else None}
    print(f"[water_supply] {len(out)} ตำบล" + (f", อ่านไม่ได้ {bad} แถว" if bad else ""))
    return out, os.path.basename(path)


def vuln_with_supply(v_level, supply):
    """IPCC risk = hazard x exposure x vulnerability: little piped water -> one vulnerability class up (max 2)."""
    if supply and supply.get("no_tap_share") is not None and supply["no_tap_share"] >= NO_TAP_HIGH:
        return min(2, v_level + 1)
    return v_level


# ---------- 2. RID dry-season allocation ----------
def load_allocation(path, geo):
    """pcode|tambon+amphoe, napr_status (full|reduced|none / ปกติ|จำกัด|งด), [plan_date], [project], [note]."""
    if not path or not os.path.exists(path):
        return {}, None
    keys, out, bad = _keys(geo), {}, 0
    for r in _read(path):
        pc = _pcode(r, keys)
        st = ALLOC_OK.get((r.get("napr_status") or "").lower())
        if not pc or not st:
            bad += 1; continue
        out[pc] = {"status": st, "plan_date": r.get("plan_date") or None, "project": r.get("project") or None}
    print(f"[rid_allocation] {len(out)} ตำบล" + (f", อ่านไม่ได้ {bad} แถว" if bad else ""))
    return out, os.path.basename(path)


# ---------- 3. impact events (DDPM declarations + local reports) ----------
def load_impacts(ddpm_records, reports, geo):
    """-> DataFrame pcode, month, type, n, source"""
    keys, rows = _keys(geo), []
    if ddpm_records and os.path.exists(ddpm_records):
        for r in _read(ddpm_records):
            pc = _pcode({"tambon": r.get("subdistrict"), "amphoe": r.get("district")}, keys)
            mo = _date(r.get("disaster declared"))
            if pc and mo is not None:
                rows.append({"pcode": pc, "month": mo, "type": "declared", "n": 1, "source": "ปภ. ประกาศเขตภัยแล้ง"})
    n_ddpm = len(rows)
    if reports and os.path.exists(reports):
        for r in _read(reports):
            pc, mo = _pcode(r, keys), _date(r.get("date"))
            if pc and mo is not None:
                rows.append({"pcode": pc, "month": mo, "type": r.get("type") or "report", "n": _num(r.get("value")) or 1,
                             "source": r.get("source") or "รายงาน อปท."})
    df = pd.DataFrame(rows, columns=["pcode", "month", "type", "n", "source"])
    if len(df):
        df = df.groupby(["pcode", "month", "type", "source"], as_index=False)["n"].sum()
    print(f"[impacts] เหตุการณ์ {df[['pcode', 'month']].drop_duplicates().shape[0] if len(df) else 0} ตำบล-เดือน "
          f"(ปภ. {n_ddpm} หมู่บ้าน, รายงานท้องถิ่น {len(rows) - n_ddpm} แถว)")
    return df


def system_kpis(hist, impacts, months_before=6):
    """
    hist: DataFrame pcode, date, alert (monthly alert level reconstructed with today's rules).
    Data for month m is ready around mid m+1, so what the system could show before an event in month E is months <= E-1.
      hit(L)  = an alert >= L in months E-3 .. E-1
      lead(L) = E - earliest month in E-6 .. E-1 with alert >= L   (for hits only)
    Precision(L): of tambons with alert >= L in the window the event record covers, share that had an event.
    Each tambon counts once: its first event.
    """
    if impacts is None or not len(impacts):
        return None
    ev_all = impacts[["pcode", "month"]].drop_duplicates()
    H = hist.set_index(["pcode", "date"])["alert"].sort_index()
    have = set(H.index.get_level_values(0))
    res = {"events": int(len(ev_all)), "tambons_with_events": int(ev_all.pcode.nunique()),
           "event_months": sorted({m.strftime("%Y-%m") for m in ev_all.month}), "by_level": {}}
    episodes = ev_all.sort_values("month").drop_duplicates("pcode", keep="first")
    episodes = episodes[episodes.pcode.isin(have)]
    for L in (1, 2):
        hits, leads = 0, []
        for _, e in episodes.iterrows():
            s = H.loc[e.pcode]
            near = s[(s.index >= e.month - pd.DateOffset(months=3)) & (s.index <= e.month - pd.DateOffset(months=1))]
            if (near >= L).any():
                hits += 1
                far = s[(s.index >= e.month - pd.DateOffset(months=months_before)) & (s.index <= e.month - pd.DateOffset(months=1))]
                first = far[far >= L].index.min()
                leads.append((e.month.year - first.year) * 12 + e.month.month - first.month)
        n = len(episodes)
        # precision over the months the event record covers
        lo = episodes.month.min() - pd.DateOffset(months=months_before); hi = episodes.month.max() - pd.DateOffset(months=1)
        hh = hist[(hist.date >= lo) & (hist.date <= hi)]
        alerted = set(hh[hh.alert >= L].pcode)
        tp = len(alerted & set(episodes.pcode))
        res["by_level"][str(L)] = {"hit_rate": round(hits / n, 2) if n else None, "hits": hits, "events": n,
                                   "lead_median_months": float(np.median(leads)) if leads else None,
                                   "precision": round(tp / len(alerted), 2) if alerted else None, "alerted_tambons": len(alerted)}
    return res


def load_outlook(path):
    """raw/local/outlook.json, see intake/local/outlook.json."""
    if not path or not os.path.exists(path):
        return {}
    o = json.load(open(path, encoding="utf-8"))
    print(f"[outlook] ENSO {o.get('enso', {}).get('nino34')} · TMD {o.get('tmd', {}).get('rain_category')}")
    return o
