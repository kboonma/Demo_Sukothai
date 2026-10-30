"""
Intake for drought indices computed outside this pipeline (e.g. the ARDA / สวก. project).

Drop files into  raw/external/<index_id>/ :
    data.csv    one row per tambon per month (or per dekad/week; averaged to months)
    meta.json   what the numbers mean (see INTAKE.md and the template in raw/external/_template/)

Check a delivery on its own:
    python scripts/external_intake.py --check raw/external/<index_id> --boundaries data/source/sukhothai_tambon_web.geojson

compute_indices.py picks up every folder in raw/external/ automatically (--external raw/external).
"""
import argparse, json, os, re, sys
import numpy as np, pandas as pd

ALLOWED_KIND = {"zscore", "raw", "class", "percent_of_normal"}
ALLOWED_USE = {"display", "agri", "water"}

PREFIX = re.compile(r"^\s*(ต\.|ตำบล|อ\.|อำเภอ|จ\.|จังหวัด|tambon|amphoe)\s*", re.I)


def _clean(s):
    return PREFIX.sub("", str(s)).strip().replace(" ", "")


def _empirical_z(s):
    from scipy import stats
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        sel = s[s.index.month == m].dropna()
        if len(sel) < 5:
            continue
        r = sel.rank(pct=True) * len(sel) / (len(sel) + 1)
        out.loc[sel.index] = stats.norm.ppf(r)
    return out


def read_meta(folder):
    p = os.path.join(folder, "meta.json")
    if not os.path.exists(p):
        raise ValueError(f"{folder}: ไม่มี meta.json")
    m = json.load(open(p, encoding="utf-8"))
    m.setdefault("id", os.path.basename(os.path.normpath(folder)))
    m["id"] = re.sub(r"[^a-z0-9_]", "_", m["id"].lower())
    m.setdefault("name", m["id"]); m.setdefault("units", ""); m.setdefault("use_in", "display")
    m.setdefault("value_column", "value"); m.setdefault("drier_is", "lower")
    errs = []
    if m.get("kind") not in ALLOWED_KIND: errs.append(f"kind ต้องเป็น {sorted(ALLOWED_KIND)}")
    if m["use_in"] not in ALLOWED_USE: errs.append(f"use_in ต้องเป็น {sorted(ALLOWED_USE)}")
    if m["drier_is"] not in ("lower", "higher"): errs.append("drier_is ต้องเป็น lower หรือ higher")
    if m.get("kind") == "class" and not m.get("class_to_z"): errs.append("kind=class ต้องมี class_to_z เช่น {\"0\":0,\"1\":-1,\"2\":-1.5,\"3\":-2}")
    if errs:
        raise ValueError(f"{folder}/meta.json: " + "; ".join(errs))
    return m


def read_data(folder, meta, geo):
    """Return (long DataFrame pcode,date,value), report dict."""
    path = next((os.path.join(folder, f) for f in ("data.csv", "data.xlsx") if os.path.exists(os.path.join(folder, f))), None)
    if path is None:
        raise ValueError(f"{folder}: ไม่มี data.csv หรือ data.xlsx")
    if path.endswith(".xlsx"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = None
        for enc in ("utf-8-sig", "cp874"):
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc); break
            except UnicodeDecodeError:
                continue
    df.columns = [c.strip().lower() for c in df.columns]
    vc = meta["value_column"].lower()
    rep = {"rows_in": len(df), "problems": []}
    if vc not in df:
        raise ValueError(f"{path}: ไม่พบคอลัมน์ค่า '{vc}' (คอลัมน์ที่มี: {list(df.columns)})")

    # --- tambon key: pcode, 6-digit code, or amphoe + tambon names ---
    feats = [f["properties"] for f in geo["features"]]
    valid = {p["pcode"] for p in feats}
    by_name = {(_clean(p["amphoe"]), _clean(p["tambon"])): p["pcode"] for p in feats}
    by_tambon = {}
    for p in feats:
        by_tambon.setdefault(_clean(p["tambon"]), []).append(p["pcode"])
    if "pcode" in df:
        pc = df["pcode"].str.strip().str.upper()
        pc = pc.where(pc.str.startswith("TH"), "TH" + pc.str.zfill(6))
    elif "tambon_code" in df:
        pc = "TH" + df["tambon_code"].str.strip().str.zfill(6)
    elif "tambon" in df:
        pcs = []
        for _, r in df.iterrows():
            t = _clean(r["tambon"]); a = _clean(r.get("amphoe", ""))
            pcs.append(by_name.get((a, t)) or (by_tambon[t][0] if len(by_tambon.get(t, [])) == 1 else None))
        pc = pd.Series(pcs, index=df.index)
    else:
        raise ValueError(f"{path}: ต้องมีคอลัมน์ pcode หรือ tambon_code หรือ tambon (+amphoe)")
    df["pcode"] = pc
    bad = df[~df["pcode"].isin(valid)]
    if len(bad):
        sample = bad.drop(columns=[vc]).head(8).to_dict("records")
        rep["problems"].append(f"จับคู่ตำบลไม่ได้ {len(bad)} แถว ตัวอย่าง: {sample}")
    df = df[df["pcode"].isin(valid)]

    # --- date: YYYY-MM, YYYY-MM-DD, or year+month columns; Buddhist years converted ---
    if "date" in df:
        d = df["date"].str.strip()
        yr = d.str.slice(0, 4).astype(int)
        d = np.where(yr > 2400, (yr - 543).astype(str) + d.str.slice(4), d)
        df["date"] = pd.to_datetime(pd.Series(d, index=df.index).str.slice(0, 7) + "-01", errors="coerce")
    elif {"year", "month"} <= set(df.columns):
        y = df["year"].astype(int); y = y.where(y < 2400, y - 543)
        df["date"] = pd.to_datetime(dict(year=y, month=df["month"].astype(int), day=1), errors="coerce")
    else:
        raise ValueError(f"{path}: ต้องมีคอลัมน์ date (YYYY-MM) หรือ year + month")
    nd = df["date"].isna().sum()
    if nd:
        rep["problems"].append(f"อ่านวันที่ไม่ได้ {nd} แถว")
    df["value"] = pd.to_numeric(df[vc].str.replace(",", ""), errors="coerce")
    nv = df["value"].isna().sum()
    if nv:
        rep["problems"].append(f"ค่าว่างหรือไม่ใช่ตัวเลข {nv} แถว")
    for k in ("missing_value", "nodata"):
        if k in meta:
            df.loc[df["value"] == float(meta[k]), "value"] = np.nan
    df = df.dropna(subset=["date", "value"])
    dup = df.duplicated(["pcode", "date"]).sum()
    if dup:
        rep["problems"].append(f"มีหลายค่าต่อตำบลต่อเดือน {dup} แถว (เฉลี่ยรวมให้)")
    df = df.groupby(["pcode", "date"], as_index=False)["value"].mean()
    rep.update({"rows_used": len(df), "tambons": df.pcode.nunique(), "of_tambons": len(valid),
                "first": df.date.min().strftime("%Y-%m") if len(df) else None,
                "last": df.date.max().strftime("%Y-%m") if len(df) else None,
                "months": df.date.nunique(), "min": round(float(df.value.min()), 3) if len(df) else None,
                "max": round(float(df.value.max()), 3) if len(df) else None})
    if rep["tambons"] < len(valid):
        rep["problems"].append(f"มีข้อมูล {rep['tambons']} จาก {len(valid)} ตำบล")
    return df, rep


def standardize(df, meta):
    """Long df -> wide (date x pcode) z where negative = drier, -1 = moderate drought."""
    wide = df.pivot(index="date", columns="pcode", values="value").sort_index()
    k = meta["kind"]
    if k == "class":
        cmap = {float(a): float(b) for a, b in meta["class_to_z"].items()}
        return wide.apply(lambda c: c.map(cmap))
    if k == "percent_of_normal":     # 100 = normal; map to z with empirical ranks per month
        k = "raw"
    if k == "raw":
        z = wide.apply(_empirical_z)
    else:
        z = wide.copy()
    if meta["drier_is"] == "higher":
        z = -z
    return z


def load_external(root, geo, verbose=True):
    """Every sub-folder of root (except those starting with _ ) is one index."""
    out = []
    if not root or not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if name.startswith(("_", ".")) or not os.path.isdir(folder):
            continue
        try:
            meta = read_meta(folder)
            df, rep = read_data(folder, meta, geo)
            z = standardize(df, meta)
            meta["report"] = rep
            out.append((meta, z))
            if verbose:
                print(f"[external] {meta['id']}: {rep['tambons']}/{rep['of_tambons']} ตำบล, {rep['first']}–{rep['last']}, use_in={meta['use_in']}")
                for pmsg in rep["problems"]:
                    print(f"   ! {pmsg}")
        except Exception as e:  # one bad delivery must not stop the monthly run
            print(f"[external] ข้ามโฟลเดอร์ {folder}: {e}", file=sys.stderr)
    return out


def check(folder, geo, ts_path=None):
    meta = read_meta(folder)
    df, rep = read_data(folder, meta, geo)
    z = standardize(df, meta)
    print(json.dumps({"meta": {k: v for k, v in meta.items() if k != "report"}, "report": rep}, ensure_ascii=False, indent=1))
    zz = z.stack()
    print(f"ค่าหลังแปลงเป็น z: เฉลี่ย {zz.mean():.2f}, ส่วนเบี่ยงเบน {zz.std():.2f}, สัดส่วน ≤ -1 = {(zz <= -1).mean():.0%}")
    # quick back-test: share of tambons <= -1 in each Nov-Mar dry season
    prov = (z <= -1).mean(axis=1)
    rows = []
    for y in range(z.index.min().year, z.index.max().year + 1):
        sel = prov[(prov.index >= f"{y}-11-01") & (prov.index <= f"{y+1}-03-01")]
        if len(sel):
            rows.append((f"{y+543-2500:02d}/{y+544-2500:02d}", round(float(sel.mean()), 2)))
    print("สัดส่วนตำบลที่ ≤ -1 รายฤดูแล้ง (พ.ย.–มี.ค.):", rows)
    print("ผ่าน" if not rep["problems"] else "ใช้ได้ แต่มีข้อควรแก้ข้างบน")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="folder raw/external/<index_id>")
    ap.add_argument("--boundaries", default="data/source/sukhothai_tambon_web.geojson")
    a = ap.parse_args()
    check(a.check, json.load(open(a.boundaries, encoding="utf-8")))
