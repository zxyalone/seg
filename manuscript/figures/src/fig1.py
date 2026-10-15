"""Fig. 1 of PhASER-3D: (a) overall architecture, (b) PFCR module, (c) ranking regions.

usage: python fig1.py <thumbs.npz> <out_prefix> [en|zh]
All coordinates are in millimetres; the figure is 190 mm wide (Elsevier double column).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import re

TH = np.load(sys.argv[1])
OUT = sys.argv[2]
LANG = sys.argv[3] if len(sys.argv) > 3 else "en"
ZH = {
    "PhASER-3D: VoComni-pretrained nnU-Net with PFCR on every skip":
        "PhASER-3D：在 VoComni 预训练 nnU-Net 的每个跳连上加入 PFCR",
    "CT patch": "CT 图像块",
    "320×4³ (bottleneck)": "320×4³（瓶颈层）",
    "Tumor-prior head": "肿瘤先验头",
    "3 conv + IN, softmax": "3 层卷积 + IN，softmax",
    "resample": "重采样",
    r"Deep supervision  $\mathcal{L}_{\rm seg}$ (Dice + CE)": "深监督损失（Dice + CE）",
    "Segmentation": "分割结果",
    "3×128³ (bg / liver / tumor)": "3×128³（背景 / 肝脏 / 肿瘤）",
    "VoComni-pretrained encoder": "VoComni 预训练编码器",
    "Decoder (unchanged)": "解码器（结构不变）",
    "PFCR (identity at init.)": "PFCR（初始化为恒等映射）",
    "Down-sampling (stride 2)": "下采样（步长 2）",
    "Up-sampling": "上采样",
    "Tumor prior $P$": "肿瘤先验 $P$",
    "Auxiliary loss": "辅助损失",
    r"PFCR module at skip level $l$ (five modules, no shared parameters)":
        r"第 $l$ 层跳连上的 PFCR 模块（共 5 个，参数互不共享）",
    "Phase evidence (3D monogenic PC, fp32)": "相位证据（三维单演相位一致性，fp32）",
    "Frequency content (single-level 3D Haar)": "频率内容（单层三维 Haar 小波）",
    "Conv 1³": "1³ 卷积",
    "init: mean": "初值：均值",
    "Learnable log-Gabor $G_s(\\rho)$": "可学习 log-Gabor $G_s(\\rho)$",
    r"× $\{1,\ iu_z/\rho,\ iu_y/\rho,\ iu_x/\rho\}$ (Riesz)": r"× $\{1,\ iu_j/\rho\}_{j=z,y,x}$（Riesz）",
    r"$\tau$: median of $A_1$ (stop-grad)": r"$\tau$：$A_1$ 的中位数（停止梯度）",
    "contrast-invariant;": "对比度不变；",
    "0 inside noise circle": "噪声圆内取 0",
    "$P_l$: tumor prior": "$P_l$：肿瘤先验",
    "from prior head": "来自先验头",
    "Router": "路由器",
    "Conv 3³ (2→8) → ReLU": "3³ 卷积（2→8）→ ReLU",
    "→ Conv 1³ (8→3, zero-init)": "→ 1³ 卷积（8→3，零初始化）",
    "→ softmax over bands": "→ 沿频带做 softmax",
    r"$R_l$ (Σ = 1 per voxel) ↓2": r"$R_l$（逐体素和为 1）↓2",
    r"region means → $\mathcal{L}_{\rm budget}$": r"区域均值 → $\mathcal{L}_{\rm budget}$",
    "Haar\nDWT": "Haar\n分解",
    "low": "低频",
    "mid (1 high-pass axis)": "中频（1 个高通轴）",
    "high (≥ 2 high-pass axes)": "高频（≥ 2 个高通轴）",
    r"$X_l^{(b)}$: 8 sub-bands, $C_l$×$\frac{D}{2}$×$\frac{H}{2}$×$\frac{W}{2}$":
        r"$X_l^{(b)}$：8 个子带，$C_l$×$\frac{D}{2}$×$\frac{H}{2}$×$\frac{W}{2}$",
    "broadcast over channels": "沿通道广播",
    "Haar IDWT": "Haar 逆变换",
    "identity": "初始时",
    "at init.": "为恒等",
    "identity path (residual)": "恒等路径（残差连接）",
    "to decoder": "送入解码器",
    r"Regions of $\mathcal{L}_{\rm rank}$": r"$\mathcal{L}_{\rm rank}$ 的监督区域",
    "$B$: boundary shell (+)": "$B$：边界壳层（正）",
    "$I$: eroded interior (−)": "$I$：腐蚀内部（负）",
    "$O$: peritumoral ring (−)": "$O$：瘤周环带（负）",
    r"$\Delta=0.2$, per level": r"$\Delta=0.2$，逐层计算",
}

W, H = 190.0, 176.0
plt.rcParams.update({
    "font.family": ["Arial", "Heiti TC"] if LANG == "zh" else "Arial", "font.size": 6.5, "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.4,
})
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ---------------- palette ----------------
C = dict(
    enc="#5B8DBE", dec="#A9C8E6", pfcr="#F2A154", prior="#5FB3A9",
    low="#7FBF7B", mid="#F2CC5B", high="#E06A5F",
    down="#6A4C9C", up="#C8463D", line="#333333", loss="#8A8A8A",
    phase="#EDE7F6", freq="#FFF4E5", panel="#F7F9FC",
)
LW = 0.6


def shade(c, f):
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))


def text(x, y, s, size=6.5, ha="center", va="center", **kw):
    size = max(size, 5.0)
    if LANG == "zh":
        s = ZH.get(s, s)
        if "$" in s and re.search(r"[\u3000-\u9fff\uff00-\uffef]", s):
            # mathtext cannot fall back to a CJK font: lay out text and math segments side by side
            props = dict(fontsize=size, color=kw.get("color", "black"), weight=kw.get("weight", "normal"),
                         style=kw.get("style", "normal"))
            segs = [g for g in re.split(r"(\$[^$]*\$)", s) if g]
            pack = HPacker(children=[TextArea(g, textprops=props) for g in segs], sep=0, pad=0,
                           align="baseline")
            ab = AnnotationBbox(pack, (x, y), frameon=False, pad=0, zorder=kw.get("zorder", 6),
                                box_alignment=({"left": 0, "center": 0.5, "right": 1}[ha],
                                               {"bottom": 0, "center": 0.5, "top": 1}[va]))
            ax.add_artist(ab)
            return
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def box(x, y, w, h, fc, ec="#333333", lw=LW, r=0.8, ls="-", z=3, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha))


def cuboid(x, y, w, h, d, c, label=None, z=3, tc="white"):
    dx, dy = 0.8 * d, 0.6 * d
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(c, 1.35), ec="#333333", lw=0.35, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(c, 0.75), ec="#333333", lw=0.35, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="#333333", lw=0.35, zorder=z))
    if label:
        text(x + w / 2, y + h / 2, label, 6.5, color=tc, weight="bold", zorder=z + 1)


def arrow(pts, color=C["line"], lw=LW, ls="-", head=True, z=4, ms=4.5):
    pts = [tuple(p) for p in pts]
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="butt")
    a, b = pts[-2], pts[-1]
    if head:
        ax.annotate("", xy=b, xytext=a, zorder=z,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=color,
                                    lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=ms))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z)


def op(x, y, sym, r=1.5, fc="white", z=5):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="#333333", lw=LW, zorder=z))
    s = r * 0.62
    if sym == "+":
        ax.plot([x - s, x + s], [y, y], color="#333333", lw=LW, zorder=z + 1)
        ax.plot([x, x], [y - s, y + s], color="#333333", lw=LW, zorder=z + 1)
    elif sym == "x":
        k = s * 0.72
        ax.plot([x - k, x + k], [y - k, y + k], color="#333333", lw=LW, zorder=z + 1)
        ax.plot([x - k, x + k], [y + k, y - k], color="#333333", lw=LW, zorder=z + 1)
    else:
        text(x, y, sym, 6, zorder=z + 1)


def thumb(img, x, y, w, h, cmap="gray", vmin=None, vmax=None, stack=0, z=4, ec="#333333", lw=0.4):
    for k in range(stack, 0, -1):
        ax.add_patch(Rectangle((x + 0.9 * k, y + 0.9 * k), w, h, fc="#DDDDDD", ec="#555555",
                               lw=0.3, zorder=z - 0.5))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + w, y, y + h),
              zorder=z, interpolation="bilinear", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.1))


def panel_label(x, y, s):
    text(x, y, s, 8, ha="left", va="top", weight="bold")


def win(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-9), 0, 1)


ct = TH["ct"]; gt = TH["gt"]
ct_n = win(ct, 0.5, 99.5)

# ======================================================================
# (a) overall architecture
# ======================================================================
box(1, 100, 188, 75, C["panel"], ec="#B8C4D6", lw=0.5, r=1.5, z=0)
panel_label(2.5, 174, "(a)")
text(30, 172.7, "PhASER-3D: VoComni-pretrained nnU-Net with PFCR on every skip", 7, ha="left",
     va="top", weight="bold")

LV = [160.0, 148.5, 137.0, 125.5, 114.0]       # level centres (l = 1..5)
YB = 105.0                                     # bottleneck centre
EW = [13.0, 11.5, 10.0, 8.5, 7.0]              # front widths
BH, DEP = 6.0, 2.6
XE, XD, XP0, XP1 = 33.0, 113.0, 75.0, 93.0     # encoder/decoder centres, PFCR box span
SH = ["32×128³", "64×64³", "128×32³", "256×16³", "320×8³"]

# input
thumb(ct_n, 3, 151, 15, 15, stack=2)
text(10.5, 148.3, "CT patch", 6.5)
text(10.5, 145.6, "1×128³", 6)
arrow([(19.8, 160.0), (XE - EW[0] / 2, 160.0)])

# encoder & decoder
for l in range(5):
    y, w = LV[l], EW[l]
    cuboid(XE - w / 2, y - BH / 2, w, BH, DEP, C["enc"], f"E{l + 1}")
    cuboid(XD - w / 2, y - BH / 2, w, BH, DEP, C["dec"], f"D{l + 1}", tc="#1E3A5F")
    if l < 4:
        arrow([(XE, y - BH / 2), (XE, LV[l + 1] + BH / 2 + 0.6 * DEP)], C["down"], 0.9)
        arrow([(XD, LV[l + 1] + BH / 2 + 0.6 * DEP), (XD, y - BH / 2)], C["up"], 0.9)
cuboid(73.0 - 3.5, YB - BH / 2, 7.0, BH, DEP, C["enc"], "E6")
text(73.0 + 1, YB - 5.2, "320×4³ (bottleneck)", 5.5, color="#444444")
arrow([(XE, LV[4] - BH / 2), (XE, YB), (73.0 - 3.5, YB)], C["down"], 0.9)
arrow([(73.0 + 3.5 + 0.8 * DEP, YB), (XD, YB), (XD, LV[4] - BH / 2)], C["up"], 0.9)

# skips through PFCR
for l in range(5):
    y = LV[l]
    xs = XE + EW[l] / 2 + 0.8 * DEP
    arrow([(xs, y), (XP0, y)])
    box(XP0, y - 3.2, XP1 - XP0, 6.4, C["pfcr"], r=0.9)
    text((XP0 + XP1) / 2, y + 0.1, f"PFCR$_{{{l + 1}}}$", 6.5, weight="bold")
    arrow([(XP1, y), (XD - EW[l] / 2, y)])
    text(xs + 1.2, y - 1.6, SH[l], 5.5, ha="left", color="#444444")
    text(XP1 + 1.2, y - 1.6, SH[l], 5.5, ha="left", color="#444444")

# tumor-prior head and P bus
px0, px1, py0, py1 = 47.5, 68.6, LV[1] + 3.2, LV[0] - 3.2
box(px0, py0, px1 - px0, py1 - py0, C["prior"], r=0.8)
text((px0 + px1) / 2, (py0 + py1) / 2 + 0.9, "Tumor-prior head", 6, weight="bold", color="white")
text((px0 + px1) / 2, (py0 + py1) / 2 - 1.3, "3 conv + IN, softmax", 5, color="white")
for xt, yt, yb in ((52.0, LV[0], py1), (56.0, LV[1], py0)):
    ax.add_patch(Circle((xt, yt), 0.5, fc="#333333", lw=0, zorder=6))
    arrow([(xt, yt), (xt, yb)], "#333333", 0.5)
XB = 71.0
ax.plot([px1, XB], [(py0 + py1) / 2] * 2, color=C["prior"], lw=0.9, ls=(0, (2.2, 1.2)), zorder=4)
ax.plot([XB, XB], [LV[0] - 1.8, LV[4] - 1.8], color=C["prior"], lw=0.9, ls=(0, (2.2, 1.2)), zorder=4)
for y in LV:
    arrow([(XB, y - 1.8), (XP0, y - 1.8)], C["prior"], 0.8)
text(XB - 0.6, LV[2] + 7.6, "$P$", 7, color=C["prior"], ha="right", weight="bold")
text(XB - 0.6, LV[2] + 4.6, "resample", 5, color=C["prior"], ha="right")

# auxiliary losses on PFCR
ax.plot([84.0, 84.0], [LV[4] - 3.2, 108.6], color=C["loss"], lw=0.6, ls=(0, (1, 1)), zorder=3)
text(85.0, 108.6, r"$\mathcal{L}_{\rm rank}$, $\mathcal{L}_{\rm budget}$", 5.5, ha="left",
     color="#555555")
text(63.5, py0 - 1.5, r"$\mathcal{L}_{\rm prior}$", 5.5, color="#555555")

# deep supervision
XS = 128.5
bar_y0, bar_y1 = LV[4] - 3.5, LV[1] + 3.5
grad = np.linspace(0, 1, 50)[:, None]
ax.imshow(grad, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("ds", ["#2F7D6E", "#BFE3C0"]),
          extent=(XS, XS + 3.2, bar_y0, bar_y1), aspect="auto", zorder=3)
ax.add_patch(Rectangle((XS, bar_y0), 3.2, bar_y1 - bar_y0, fc="none", ec="#333333", lw=0.35, zorder=4))
text(XS + 1.6, (bar_y0 + bar_y1) / 2, r"Deep supervision  $\mathcal{L}_{\rm seg}$ (Dice + CE)", 5.5,
     rotation=90, color="#1F3F38", zorder=5)
for l, wgt in zip(range(1, 5), ["0.267", "0.133", "0.067", "0"]):
    y = LV[l]
    xs = XD + EW[l] / 2 + 0.8 * DEP
    arrow([(xs, y), (XS, y)], C["line"] if wgt != "0" else C["loss"], 0.5,
          ls="-" if wgt != "0" else (0, (1.5, 1)))
    text((xs + XS) / 2 + 0.6, y + 1.2, f"ω={wgt}", 4.8, color="#444444")
# output
xo = XD + EW[0] / 2 + 0.8 * DEP
arrow([(xo, LV[0]), (137.5, LV[0])])
text((xo + 137.5) / 2 + 0.3, LV[0] + 1.2, "ω=0.533", 4.8, color="#444444")
seg = np.dstack([ct_n] * 3)
for lab, col in ((1, "#4C9EE8"), (2, "#E24B4A")):
    m = gt == lab
    seg[m] = 0.55 * seg[m] + 0.45 * np.array(matplotlib.colors.to_rgb(col))
thumb(seg, 138, 152.5, 15, 15)
text(145.5, 150.3, "Segmentation", 6)
text(145.5, 147.8, "3×128³ (bg / liver / tumor)", 5)

# legend
lx, ly = 150.0, 141.5
items = [("cub", C["enc"], "VoComni-pretrained encoder"), ("cub", C["dec"], "Decoder (unchanged)"),
         ("box", C["pfcr"], "PFCR (identity at init.)"), ("box", C["prior"], "Tumor-prior head"),
         ("arr", C["down"], "Down-sampling (stride 2)"), ("arr", C["up"], "Up-sampling"),
         ("dash", C["prior"], "Tumor prior $P$"), ("dot", C["loss"], "Auxiliary loss")]
for i, (kind, col, lab) in enumerate(items):
    y = ly - i * 3.6
    if kind == "cub":
        cuboid(lx, y - 1.1, 4.0, 2.2, 1.2, col)
    elif kind == "box":
        box(lx, y - 1.2, 5.0, 2.4, col, r=0.4)
    elif kind == "arr":
        arrow([(lx, y), (lx + 5.2, y)], col, 0.9)
    elif kind == "dash":
        ax.plot([lx, lx + 5.2], [y, y], color=col, lw=0.9, ls=(0, (2.2, 1.2)))
    else:
        ax.plot([lx, lx + 5.2], [y, y], color=col, lw=0.7, ls=(0, (1, 1)))
    text(lx + 7.0, y, lab, 5.6, ha="left")
text(lx - 1, ly - 8 * 3.6 - 0.6,
     r"$\mathcal{L}=\mathcal{L}_{\rm seg}+\psi(e)\,(0.2\,\mathcal{L}_{\rm rank}+0.05\,\mathcal{L}_{\rm budget}"
     r"+0.1\,\mathcal{L}_{\rm prior})$", 5.4, ha="left")

# ======================================================================
# (b) PFCR module
# ======================================================================
box(1, 1, 188, 97, "white", ec="#B8C4D6", lw=0.5, r=1.5, z=0)
panel_label(2.5, 97, "(b)")
text(9, 95.7, r"PFCR module at skip level $l$ (five modules, no shared parameters)", 7, ha="left",
     va="top", weight="bold")
# lane backgrounds
box(20, 62, 106, 25, C["phase"], ec="none", r=1.2, z=0.5)
text(21.2, 85.6, "Phase evidence (3D monogenic PC, fp32)", 5.8, ha="left", va="top",
     color="#4A3A75", style="italic")
box(20, 1.8, 167, 37.2, C["freq"], ec="none", r=1.2, z=0.5)
text(21.2, 37.6, "Frequency content (single-level 3D Haar)", 5.8, ha="left", va="top",
     color="#8A5A12", style="italic")

YT = 73.5        # top-lane centre
# input feature
feat = win(TH["small"])
thumb(feat, 4.5, 44, 10, 10, stack=2)
text(13.0, 41.6, "$X_l$", 7, ha="left")
text(11.0, 38.9, "$C_l$×$D$×$H$×$W$", 5.0, ha="left")
arrow([(9.5, 57.0), (9.5, YT), (22.5, YT)])
arrow([(9.5, 43.5), (9.5, 21.0), (23.5, 21.0)])

# proxy
box(22.5, YT - 4.5, 12, 9, "white")
text(28.5, YT + 1.6, "Conv 1³", 5.8, weight="bold")
text(28.5, YT - 1.0, "$C_l$→1", 5.5)
text(28.5, YT - 3.1, "init: mean", 4.8, color="#555555")
arrow([(34.5, YT), (37.0, YT)])
thumb(feat, 37, YT - 4.5, 9, 9)
text(41.5, YT - 6.6, "$p_l$", 6.5)
arrow([(46.0, YT), (48.5, YT)])
box(48.5, YT - 4.5, 7.5, 9, "white")
text(52.25, YT + 0.8, "rFFT", 5.8, weight="bold")
text(52.25, YT - 1.8, r"$\hat p$", 6)
arrow([(56.0, YT), (58.5, YT)])

# log-Gabor bank (inset plot)
box(58.5, YT - 9.5, 26, 19, "white")
text(71.5, YT + 7.9, "Learnable log-Gabor $G_s(\\rho)$", 5.5, weight="bold")
pa = fig.add_axes([61.0 / W, (YT - 3.7) / H, 21.5 / W, 10.0 / H])
rho = np.linspace(1e-4, 0.5, 400)
for lam, col in zip((3.0, 6.3, 13.2), ("#3B2A6B", "#7E66B3", "#BBA9DD")):
    G = np.exp(-np.log(rho * lam) ** 2 / (2 * np.log(0.55) ** 2))
    pa.plot(rho, G, color=col, lw=0.8)
pa.set_xlim(0, 0.5); pa.set_ylim(0, 1.08)
pa.set_xticks([0, 0.25, 0.5]); pa.set_xticklabels(["0", "", "0.5"], fontsize=5)
pa.set_yticks([]); pa.tick_params(length=1.2, width=0.3, pad=0.6)
for s in ("top", "right"):
    pa.spines[s].set_visible(False)
pa.set_facecolor("none")
text(83.3, YT - 3.0, "ρ", 5.0, ha="left")
text(71.5, YT - 8.1, r"× $\{1,\ iu_z/\rho,\ iu_y/\rho,\ iu_x/\rho\}$ (Riesz)", 4.9)
arrow([(84.5, YT), (87.0, YT)])
box(87.0, YT - 4.5, 8.5, 9, "white")
text(91.25, YT + 1.4, "irFFT", 5.8, weight="bold")
text(91.25, YT - 1.6, r"$\mathbf{v}_s$, 3×4", 5.2)
arrow([(95.5, YT), (98.0, YT)])

# energy, threshold, PC
box(98.0, YT - 7.5, 25.5, 15, "white")
text(110.75, YT + 5.0, r"$E=\|\sum_s \mathbf{v}_s\|$,  $\Sigma_A=\sum_s\|\mathbf{v}_s\|$", 5.3)
text(110.75, YT + 1.4, r"$T=\tau g\,(\sqrt{\pi/2}+k\sqrt{(4-\pi)/2})$", 5.3)
text(110.75, YT - 1.5, r"$\tau$: median of $A_1$ (stop-grad)", 4.6, color="#555555")
text(110.75, YT - 4.8, r"PC $=[E-T]_+\,/\,(\Sigma_A+\epsilon\bar\Sigma_A)$", 5.5, weight="bold")
arrow([(123.5, YT), (127.0, YT)])
pc = TH["pc"]
thumb(pc, 127.0, YT - 5, 10, 10, cmap="magma", vmin=0, vmax=0.8)
text(132.0, YT + 6.6, r"PC$_l\in[0,1]$", 5.8)
text(132.0, YT - 7.0, "contrast-invariant;", 4.6, color="#555555")
text(132.0, YT - 9.0, "0 inside noise circle", 4.6, color="#555555")

# ----- router (middle band) -----
YM = 49.0
op(132.0, YM, "C", r=1.6)
arrow([(132.0, YT - 10.2), (132.0, YM + 1.6)])
prior_img = TH["prior"]
thumb(prior_img, 112.5, YM - 4.5, 9, 9, cmap="viridis", vmin=0, vmax=1)
text(117.0, YM + 6.4, "$P_l$: tumor prior", 5.2, color="#2E6F67")
ax.plot([104.0, 112.5], [YM, YM], color=C["prior"], lw=0.9, ls=(0, (2.2, 1.2)), zorder=4)
arrow([(121.5, YM), (130.4, YM)], C["prior"], 0.8)
text(103.5, YM, "from prior head", 4.8, ha="right", color="#2E6F67")
arrow([(133.6, YM), (137.0, YM)])
box(137.0, YM - 6.5, 27.0, 13, "#FFE7CF", ec=C["pfcr"], lw=0.8)
text(150.5, YM + 4.1, "Router", 6, weight="bold")
text(150.5, YM + 0.9, "Conv 3³ (2→8) → ReLU", 5.0)
text(150.5, YM - 1.6, "→ Conv 1³ (8→3, zero-init)", 5.0)
text(150.5, YM - 4.2, "→ softmax over bands", 5.0)
arrow([(164.0, YM), (167.0, YM)])
# route maps
for i, (key, col, lab) in enumerate((("low", C["low"], "low"), ("mid", C["mid"], "mid"),
                                     ("high", C["high"], "high"))):
    x = 167.5 + i * 6.8
    ax.add_patch(Rectangle((x, YM - 2.6), 5.2, 5.2, fc=col, ec="#333333", lw=0.4, zorder=4))
    text(x + 2.6, YM + 4.4, rf"$r^{{\rm {lab}}}$", 5.6)
text(177.8, YM - 4.7, r"$R_l$ (Σ = 1 per voxel) ↓2", 5.0)
text(177.8, 56.8, r"region means → $\mathcal{L}_{\rm budget}$", 5.0, color="#555555")

# ----- Haar decomposition (bottom lane) -----
box(23.5, 7.5, 7.5, 27, "white")
text(27.25, 21.0, "Haar\nDWT", 5.8, weight="bold", linespacing=1.1)
sub = {"LL": win(TH["LL"]), "LH": win(np.abs(TH["LH"]), 2, 98), "HL": win(np.abs(TH["HL"]), 2, 98),
       "HH": win(np.abs(TH["HH"]), 2, 98)}
rows = [("low", 29.0, ["LLL"], ["LL"]),
        ("mid", 20.0, ["LLH", "LHL", "HLL"], ["LH", "HL", "LH"]),
        ("high", 11.0, ["LHH", "HLH", "HHL", "HHH"], ["HH", "HH", "HL", "HH"])]
XQ = 157.0
for name, y, labs, imgs in rows:
    col = C[name]
    n = len(labs)
    gx0 = 35.0
    box(gx0 - 1.2, y - 3.9, n * 10.5 + 1.0, 7.8, shade(col, 1.55), ec=col, lw=0.6, r=0.8, z=1)
    arrow([(31.0, 21.0 + (y - 20.0) * 0.55), (gx0 - 1.2, y)], "#555555", 0.45)
    for j, (lb, im) in enumerate(zip(labs, imgs)):
        x = gx0 + j * 10.5
        img = sub[im] if j < 2 or name != "mid" else np.maximum(sub["LH"], sub["HL"])
        thumb(img, x, y - 2.6, 5.2, 5.2)
        text(x + 5.5, y + 2.0, lb, 4.2, ha="left", va="top", color="#333333")
    lab_txt = {"low": "low", "mid": "mid (1 high-pass axis)", "high": "high (≥ 2 high-pass axes)"}[name]
    lx0 = gx0 + n * 10.5 + 1.2
    text(lx0, y, lab_txt, 5.2, ha="left", color=shade(col, 0.6), weight="bold")
    off = {"low": 6, "mid": 21.5, "high": 23.5} if LANG == "en" else {"low": 5.5, "mid": 18.5, "high": 21.0}
    arrow([(lx0 + off[name], y), (XQ - 1.4, y)], "#555555", 0.5)
    op(XQ, y, "x", r=1.4)
    text(XQ - 3.2, y + 2.3, rf"$\bar r^{{\rm {name}}}$", 5.4, color=shade(col, 0.6))
    arrow([(XQ + 1.4, y), (164.5, y)], "#555555", 0.5)
text(100, 33.0, r"$X_l^{(b)}$: 8 sub-bands, $C_l$×$\frac{D}{2}$×$\frac{H}{2}$×$\frac{W}{2}$", 5.4, ha="left")
# R_l broadcast to the multipliers
box(152.0, 7.0, 9.0, 27.0, "none", ec=C["pfcr"], lw=0.7, ls=(0, (2, 1)), r=0.8, z=2)
arrow([(177.8, YM - 6.3), (177.8, 40.5), (156.5, 40.5), (156.5, 34.0)], C["pfcr"], 0.8)
text(167.0, 38.9, "broadcast over channels", 4.6, color="#8A5A12")

# IDWT, gate, residual
box(164.5, 7.0, 7.0, 27.0, "white")
text(168.0, 20.5, "Haar IDWT", 5.6, weight="bold", rotation=90)
arrow([(171.5, 20.0), (174.6, 20.0)])
op(176.2, 20.0, "x", r=1.6)
text(176.2, 24.6, r"tanh($\gamma_l$)", 4.9)
text(176.2, 22.6, r"$\gamma_l^{(0)}=0$", 4.4, color="#555555")
text(173.0, 18.2, "$Y_l$", 5.4)
op(184.0, 20.0, "+", r=1.6)
arrow([(177.8, 20.0), (182.4, 20.0)])
ax.plot([9.5, 9.5], [21.0, 3.2], color=C["line"], lw=LW, zorder=4)
arrow([(9.5, 3.2), (184.0, 3.2), (184.0, 18.4)], C["line"], LW)
text(80.0, 4.3, "identity path (residual)", 4.8, color="#555555", style="italic")
arrow([(184.0, 21.6), (184.0, 27.0)])
text(184.0, 28.6, r"$\tilde X_l$", 6.2, weight="bold")
text(184.0, 31.2, "to decoder", 4.6, color="#555555")
text(180.0, 13.0, "identity", 4.6, color="#555555")
text(180.0, 11.0, "at init.", 4.6, color="#555555")

# ======================================================================
# (c) ranking regions (inset, top right of panel b)
# ======================================================================
box(141.0, 59.5, 47.0, 29.0, "white", ec="#B8C4D6", lw=0.5, r=1.2, z=1)
panel_label(142.2, 87.6, "(c)")
text(148.5, 86.6, r"Regions of $\mathcal{L}_{\rm rank}$", 5.8, ha="left", va="top", weight="bold")
les = gt == 2
yy, xx = np.where(les)
cy, cx = int(yy.mean()), int(xx.mean()); hw = 80
crop = (slice(max(cy - hw, 0), cy + hw), slice(max(cx - hw, 0), cx + hw))
rg = np.dstack([win(ct, 0.5, 99.5)[crop]] * 3) * 0.85
for key, colr in (("O", "#4C9EE8"), ("I", "#F2CC5B"), ("B", "#E24B4A")):
    m = TH[key][crop]
    rg[m] = 0.55 * rg[m] + 0.45 * np.array(matplotlib.colors.to_rgb(colr))
thumb(rg, 143.0, 61.3, 21.0, 21.0)
for i, (colr, lab) in enumerate((("#E24B4A", "$B$: boundary shell (+)"), ("#F2CC5B", "$I$: eroded interior (−)"),
                                 ("#4C9EE8", "$O$: peritumoral ring (−)"))):
    y = 79.0 - i * 4.2
    ax.add_patch(Rectangle((166.0, y - 1.1), 2.2, 2.2, fc=colr, ec="#333333", lw=0.3, zorder=5))
    text(169.0, y, lab, 5.0, ha="left")
text(176.3, 64.8, r"$[\Delta-(\mu_B-\mu_{I\cup O})]_+$", 5.4)
text(176.3, 62.0, r"$\Delta=0.2$, per level", 4.8, color="#555555")
ax.plot([137.0, 141.0], [YT, YT], color=C["loss"], lw=0.6, ls=(0, (1, 1)), zorder=3)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=600 if ext == "png" else None)
print("saved", OUT)
