"""Static figures for the briefing deck and technical document (reads data/)."""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

D = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
fm.fontManager.addfont("/usr/share/fonts/opentype/tlwg/Loma.otf")
fm.fontManager.addfont("/usr/share/fonts/opentype/tlwg/Loma-Bold.otf")
plt.rcParams.update({"font.family": "Loma", "font.size": 11, "axes.spines.top": False, "axes.unicode_minus": False, "axes.spines.right": False,
                     "axes.edgecolor": "#5b6866", "axes.labelcolor": "#17201f", "xtick.color": "#5b6866", "ytick.color": "#5b6866"})
INK, MUTED, ACCENT = "#17201f", "#5b6866", "#0b6e82"

geo = json.load(open(f"{D}/tambons.geojson"))["features"]
ts = json.load(open(f"{D}/timeseries.json"))
P = pd.DataFrame([f["properties"] for f in geo])


def draw_map(values_or_colors, fname, title, legend=None, cmap=None, norm=None, cbar_label=None, labels=None):
    fig, ax = plt.subplots(figsize=(5.2, 6.6))
    patches, vals = [], []
    for f, v in zip(geo, values_or_colors):
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            patches.append(Polygon(poly[0], closed=True)); vals.append(v)
    if cmap is None:
        pc = PatchCollection(patches, facecolor=vals, edgecolor="white", linewidth=0.5)
    else:
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.5); pc.set_array(np.array(vals, float))
    ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect("equal"); ax.set_axis_off()
    # amphoe labels at mean centroid
    for am, grp in P.groupby("amphoe_en"):
        cs = []
        for f in geo:
            if f["properties"]["amphoe_en"] == am:
                g = f["geometry"]; ring = (g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0])
                cs.append(np.mean(ring, axis=0))
        c = np.mean(cs, axis=0)
        ax.text(c[0], c[1], grp["amphoe"].iloc[0], ha="center", va="center", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    if labels is not None:
        for f, lab in zip(geo, labels):
            if lab:
                g = f["geometry"]; ring = (g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0])
                c = np.mean(ring, axis=0); ax.text(c[0], c[1] - 0.025, lab, ha="center", fontsize=7, color=INK)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor="#999", label=l) for l, c in legend], loc="lower left",
                  frameon=False, fontsize=9, bbox_to_anchor=(-0.02, -0.02))
    if cmap is not None:
        cb = fig.colorbar(pc, ax=ax, shrink=0.45, pad=0.01); cb.set_label(cbar_label); cb.outline.set_visible(False)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", fontweight="bold")
    fig.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


ALERT_C = ["#b9dcb0", "#f0c93a", "#ea7d1f", "#c23a2c"]
ALERT_L = ["ปกติ", "เฝ้าระวัง", "เตือนเกษตร", "เตรียมแจกจ่ายน้ำ"]
CDI_C = ["#b9dcb0", "#f3c56b", "#e07a2f", "#a8321f"]
CDI_L = ["ปกติ", "Watch: ฝนขาด", "Warning: ดินแห้ง", "Alert: พืชเครียด"]
VUL_C = ["#e6ece9", "#c9a9d6", "#6b3f8a"]
VUL_L = ["ไม่เคย", "1 ปี", "2–3 ปี"]

cnt = P.alert_dec.value_counts()
draw_map([ALERT_C[a] for a in P.alert_dec], "map_alert_dec.png", "ระดับเตือนแนวโน้ม ธ.ค. 2569",
         legend=[(f"{ALERT_L[i]} ({cnt.get(i, 0)})", ALERT_C[i]) for i in range(4)])
cnt = P.agri_now.value_counts()
draw_map([CDI_C[a] for a in P.agri_now], "map_cdi_now.png", "แล้งเกษตร (CDI) ก.ย. 2569",
         legend=[(f"{CDI_L[i]} ({cnt.get(i, 0)})", CDI_C[i]) for i in range(4)],
         labels=None)
cnt = P.vuln.value_counts()
draw_map([VUL_C[a] for a in P.vuln], "map_vulnerability.png", "ปีที่มีหมู่บ้านในพื้นที่ภัยแล้ง 2560–2567 (ปภ.)",
         legend=[(f"{VUL_L[i]} ({cnt.get(i, 0)} ตำบล)", VUL_C[i]) for i in range(3)])
div = LinearSegmentedColormap.from_list("dw", ["#8c4a16", "#d9a35c", "#efe9dc", "#6fb3c9", "#1f5f8b"])
draw_map(P.spi3_now.values, "map_spi3_now.png", "ฝนสะสม 3 เดือน (SPI-3) ก.ย. 2569", cmap=div,
         norm=TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2), cbar_label="SPI-3 (ต่ำกว่า -1 = แล้ง)")
draw_map(P.sm_z_now.values, "map_sm_now.png", "ความชื้นดินผิดปกติ ก.ย. 2569", cmap=div,
         norm=TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2), cbar_label="z-score (ต่ำกว่า -1 = ดินแห้ง)")

# province-level series
rows = []
for pc, s in ts.items():
    df = pd.DataFrame({k: s[k] for k in ["dates", "spi3", "sm_z", "veg_z", "agri_level", "water_haz"]}); df["pc"] = pc; rows.append(df)
A = pd.concat(rows); A["date"] = pd.to_datetime(A.dates + "-01")
M = A.groupby("date").agg(spi3=("spi3", "mean"), sm=("sm_z", "mean"), veg=("veg_z", "mean"),
                          warn=("agri_level", lambda x: (x >= 2).mean()), haz=("water_haz", lambda x: (x >= 2).mean()))

# back-test bar
seasons, share = [], []
for y in range(2018, 2026):
    sel = M[(M.index >= f"{y}-11-01") & (M.index <= f"{y+1}-03-01")]
    seasons.append(f"{y+543-2500}/{y+544-2500}"); share.append(sel.warn.mean())
fig, ax = plt.subplots(figsize=(8, 3.6))
cols = [("#a8321f" if s == "62/63" else "#c9d3d1") for s in seasons]
b = ax.bar(seasons, share, color=cols, width=0.6)
for r, v in zip(b, share):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 1); ax.set_ylabel("สัดส่วนตำบล"); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("ฤดูแล้ง (พ.ย.–มี.ค.)")
ax.annotate("ปีที่ ปภ. ประกาศภัยแล้ง\n(ลงนาม 25 ธ.ค. 2562)", xy=(1.32, share[1] - 0.04), xytext=(2.3, 0.72), fontsize=9.5, color="#a8321f",
            arrowprops=dict(arrowstyle="-", color="#a8321f", lw=0.8))
ax.annotate("El Niño 2566/67\nไม่มีการประกาศภัย", xy=(5, 0.06), xytext=(4.6, 0.35), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_title("ทดสอบย้อนหลัง: ตำบลที่ CDI ระดับ Warning ขึ้นไป", loc="left", fontsize=12.5, fontweight="bold", color=INK)
fig.savefig(f"{OUT}/backtest.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 2019 vs 2026 trajectory (May -> Apr)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
mon = ["พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย."]
for ax, (col, lab) in zip(axs, [("spi3", "ฝนสะสม 3 เดือน (SPI-3)"), ("sm", "ความชื้นดิน (z)")]):
    for y, c, lw, name in [(2019, "#a8321f", 2.2, "2562/63 (ปีแล้ง)"), (2023, "#c9a14a", 1.5, "2566/67 (El Niño)"), (2026, ACCENT, 2.6, "2569/70 (ปีนี้)")]:
        idx = pd.date_range(f"{y}-05-01", periods=12, freq="MS")
        v = M[col].reindex(idx).values
        ax.plot(range(12), v, color=c, lw=lw, label=name, marker="o" if y == 2026 else None, ms=4)
    ax.axhline(-1, color=MUTED, lw=0.8, ls="--"); ax.axhline(0, color="#d5dcdc", lw=0.8)
    ax.text(11.2, -1.05, "เกณฑ์ -1", fontsize=8.5, color=MUTED, va="top", ha="right")
    ax.set_xticks(range(12), mon, fontsize=9); ax.set_title(lab, loc="left", fontsize=11.5, color=INK)
    ax.set_ylim(-2.6, 2.2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("ค่าเฉลี่ยทั้งจังหวัด: ปีนี้เทียบปีแล้ง", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/trajectory.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# counts used in text
summary = {"alert_dec": P.alert_dec.value_counts().to_dict(), "alert_now": P.alert_now.value_counts().to_dict(),
           "cdi_now": P.agri_now.value_counts().to_dict(), "vuln": P.vuln.value_counts().to_dict(),
           "cdi_warning_tambons": P[P.agri_now >= 2][["amphoe", "tambon"]].values.tolist(),
           "backtest": dict(zip(seasons, [round(x, 3) for x in share])),
           "prov_2026": M.loc["2026-05":, ["spi3", "sm", "veg"]].round(2).reset_index().astype(str).values.tolist(),
           "prov_2019": M.loc["2019-05":"2020-04", ["spi3", "sm", "veg"]].round(2).reset_index().astype(str).values.tolist(),
           "pop_watch_dec": float(P[P.alert_dec >= 1].pop_2020.sum()), "pop_total": float(P.pop_2020.sum())}
json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1, default=int)
print(json.dumps(summary, ensure_ascii=False, default=int)[:2500])
