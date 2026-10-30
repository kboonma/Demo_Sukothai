"""
Fetch current storage of medium reservoirs in Sukhothai from the RID reservoir database
(https://app.rid.go.th/reservoir/rsvmiddle, public, no login or key) -> raw/reservoirs.csv

    python3 scripts/fetch_rid_reservoirs.py                 # today
    python3 scripts/fetch_rid_reservoirs.py --date 2026-11-06
then  python3 scripts/compute_indices.py ... --reservoirs raw/reservoirs.csv   (or run reservoirs_overlay alone)

Uses only the Python standard library. Run it from the repository folder.
"""
import argparse, csv, datetime as dt, json, os, urllib.parse, urllib.request

API = "https://app.rid.go.th/reservoir/api/rsvmiddles"
PROVINCE = "สุโขทัย"


def fetch(date):
    body = urllib.parse.urlencode({"date": date, "region": "", "percent": "", "percent_from": "", "percent_to": "", "status": 1}).encode()
    req = urllib.request.Request(API, data=body, headers={"User-Agent": "sukhothai-drought-watch (research prototype)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return [x for reg in d.get("region", []) for x in reg.get("reservoir", []) if PROVINCE in (x.get("tprov") or "")]


def _inring(x, y, ring):
    c = False
    for i in range(len(ring)):
        x1, y1 = ring[i][:2]; x2, y2 = ring[i - 1][:2]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def locate(geo, lon, lat):
    for f in geo["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        if any(_inring(lon, lat, p[0]) and not any(_inring(lon, lat, h) for h in p[1:]) for p in polys):
            return f["properties"]
    return {}


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--boundaries", default="data/source/sukhothai_tambon_web.geojson")
    ap.add_argument("--out", default="raw/reservoirs.csv")
    a = ap.parse_args()
    rows = fetch(a.date)
    if not rows:
        raise SystemExit(f"ไม่พบข้อมูลอ่างใน{PROVINCE} วันที่ {a.date} (ลองวันก่อนหน้า --date)")
    geo = json.load(open(a.boundaries, encoding="utf-8"))
    out = []
    for r in rows:
        lat, lon = num(r["cresv_lat"]), num(r["cresv_lng"])
        p = locate(geo, lon, lat)
        out.append({"name": r["nresv"].replace("อ่างเก็บน้ำ ", "อ่างเก็บน้ำ"), "type": "ขนาดกลาง",
                    "tambon": p.get("tambon", ""), "amphoe": p.get("amphoe", ""), "lat": lat, "lon": lon,
                    "capacity_mcm": num(r["cap_resv"]), "storage_mcm": num(r["qdisc_curr"]), "date": r["date"],
                    "storage_prev_mcm": num(r["qdisc_prev"]), "usable_mcm": num(r["water_workable"]), "dead_mcm": num(r["low_qdisc"]),
                    "rid_code": r["cresv"], "pcode": p.get("pcode", ""), "source": "กรมชลประทาน app.rid.go.th/reservoir"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for x in out:
        pct = x["storage_mcm"] / x["capacity_mcm"] * 100 if x["capacity_mcm"] else float("nan")
        print(f"{x['name']:<28} อ.{x['amphoe']:<14} {pct:5.1f}%")
    print(f"-> {a.out} ({len(out)} แห่ง, ข้อมูล {a.date})")
