#!/usr/bin/env python3
"""
SALT architecture figure (docs/SALT模型设计.md), hand-built SVG in the style of fig1_architecture.

    python manuscript/figures/src/fig_salt.py          # writes fig_salt_architecture_{en,zh}.svg

Thumbnails are cropped from fig1_architecture.png (one CT slice and its liver / tumour overlay). The tumour
probability map is illustrative (the overlay's tumour mask, smoothed). The anchors and the scale responses
in panel (b) are computed on that slice with the 2D analogue of the SSA operator (sigma = r / sqrt(2));
the eigenvalue ratios in panel (e) are the values measured in experiments/test_salt.py.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

FIG = Path(__file__).resolve().parents[1]
SRC_PNG = FIG / "fig1_architecture.png"

W, H = 1600, 1612
FONT = {"en": "Liberation Sans, Helvetica, Arial, sans-serif",
        "zh": "WenQuanYi Zen Hei, Noto Sans CJK SC, Microsoft YaHei, PingFang SC, sans-serif"}

C = dict(
    panel="#f7f9fc", panel_stroke="#b8c4d6", ink="#222222", grey="#666666",
    enc=("#4f81bd", "#6f9bd1", "#3a6496"), dec=("#a9c8ec", "#c9def6", "#86a8d2"),
    faf="#f4a259", faf_dark="#b8641c", faf_bg="#fff4e5",
    ssa="#9d8ac7", ssa_dark="#5e4b8b", ssa_bg="#ede7f6",
    lqd="#4caf9a", lqd_dark="#2e7d6c", lqd_bg="#e3f4f0",
    smd="#e57f94", smd_dark="#a23b55", smd_bg="#fdebef",
    down="#6a4c9c", up="#c0504d", pt="#2e9c8a", anchor="#5e4b8b", fuse="#f6d55c",
)


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s: str, size: float) -> str:
    """Mini markup: _{...} subscript, ^{...} superscript, as tspans with absolute dy shifts (px)."""
    out, shift, i = [], 0.0, 0
    buf = ""

    def flush(txt, target, small):
        nonlocal shift
        if not txt:
            return
        dy = target - shift
        attrs = f' dy="{dy:.2f}"' if abs(dy) > 1e-6 else ""
        if small:
            attrs += f' font-size="{0.72 * size:.1f}"'
        out.append(f"<tspan{attrs}>{esc(txt)}</tspan>")
        shift = target

    while i < len(s):
        if s[i] in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            flush(buf, 0.0, False)
            buf = ""
            j = s.index("}", i)
            flush(s[i + 2:j], 0.26 * size if s[i] == "_" else -0.40 * size, True)
            i = j + 1
            continue
        buf += s[i]
        i += 1
    flush(buf, 0.0, False)
    return "".join(out)


class Svg:
    def __init__(self, lang):
        self.lang = lang
        self.items = []

    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=None, op=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{op}"' if op is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{o} '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=14, anchor="start", weight="normal", fill=None, italic=False, rotate=None):
        st = ' font-style="italic"' if italic else ""
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
                 f'fill="{fill or C["ink"]}"{st}{tr}>{rich(s, size)}</text>')

    def lines(self, x, y, rows, size=14, gap=None, **kw):
        gap = gap or size * 1.32
        for k, r in enumerate(rows):
            self.text(x, y + k * gap, r, size=size, **kw)

    def line(self, pts, color=C["ink"], sw=1.6, dash=None, arrow=True, op=None):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        me = f' marker-end="url(#ah{color[1:]})"' if arrow else ""
        o = f' stroke-opacity="{op}"' if op is not None else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{me}{o} '
                 f'stroke-linejoin="round"/>')

    def circle(self, cx, cy, r, fill="none", stroke=C["ink"], sw=1.2, dash=None, op=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{op}"' if op is not None else ""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{o} stroke="{stroke}" '
                 f'stroke-width="{sw}"{da}/>')

    def poly(self, pts, fill, stroke="none", sw=1.0, op=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o = f' fill-opacity="{op}"' if op is not None else ""
        self.add(f'<polygon points="{p}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"/>')

    def image(self, x, y, w, h, png_bytes, stroke="#333333"):
        b64 = base64.b64encode(png_bytes).decode()
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>')
        if stroke:
            self.rect(x, y, w, h, stroke=stroke, sw=1.2)

    def cuboid(self, x, y, w, h, cols, label, d=16, size=17, fill_txt="white"):
        front, top, side = cols
        dy = d * 0.75
        self.poly([(x, y), (x + d, y - dy), (x + w + d, y - dy), (x + w, y)], top, stroke="#2b2b2b", sw=0.6)
        self.poly([(x + w, y), (x + w + d, y - dy), (x + w + d, y + h - dy), (x + w, y + h)], side,
                  stroke="#2b2b2b", sw=0.6)
        self.rect(x, y, w, h, fill=front, stroke="#2b2b2b", sw=0.8)
        self.text(x + w / 2, y + h / 2 + size * 0.35, label, size=size, anchor="middle", weight="bold", fill=fill_txt)

    def box(self, x, y, w, h, fill, stroke, title=None, rows=(), size=14, tsize=15, rx=8, sw=1.4,
            title_fill=None, rows_fill=None, center=True, top_pad=None):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)
        n = (1 if title else 0) + len(rows)
        gap = size * 1.3
        total = (tsize * 1.3 if title else 0) + gap * len(rows)
        yy = y + (top_pad if top_pad is not None else (h - total) / 2) + (tsize if title else size) * 0.95
        xx = x + w / 2 if center else x + 10
        anc = "middle" if center else "start"
        if title:
            self.text(xx, yy, title, size=tsize, anchor=anc, weight="bold", fill=title_fill)
            yy += tsize * 1.3
        for r in rows:
            self.text(xx, yy, r, size=size, anchor=anc, fill=rows_fill)
            yy += gap
        return n

    def svg(self):
        heads = "".join(
            f'<marker id="ah{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/>'
            f'</marker>' for c in {C["ink"], C["down"], C["up"], C["pt"], C["anchor"], C["lqd_dark"], C["grey"],
                                   C["faf_dark"], C["smd_dark"]})
        return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                f'font-family="{FONT[self.lang]}">\n<defs>{heads}</defs>\n'
                f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>\n' + "\n".join(self.items) + "\n</svg>\n")


def png(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG", optimize=True)
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Image material and the SSA illustration
# ---------------------------------------------------------------------------

def material():
    src = Image.open(SRC_PNG).convert("RGB")
    ct = src.crop((73, 238, 424, 589)).resize((352, 352), Image.LANCZOS)
    ov = src.crop((3260, 200, 3613, 553)).resize((352, 352), Image.LANCZOS)
    a = np.asarray(ov).astype(int)
    tum = (a[..., 0] > a[..., 1] + 40) & (a[..., 0] > a[..., 2] + 40)
    liv = (a[..., 2] > a[..., 0] + 25) & (a[..., 2] > 90)
    return ct, ov, tum, liv | tum


def log2d(v, sp_mm, radii):
    """2D analogue of spectral_anchors' bank: (2 pi s |f|)^2 exp(-2 pi^2 s^2 |f|^2), s = r / sqrt(2)."""
    pad = 48
    vp = np.pad(v, pad)
    fy = np.fft.fftfreq(vp.shape[0], d=sp_mm)[:, None]
    fx = np.fft.rfftfreq(vp.shape[1], d=sp_mm)[None, :]
    f2 = fy ** 2 + fx ** 2
    V = np.fft.rfft2(vp)
    out = []
    for r in radii:
        s = r / math.sqrt(2)
        h = (2 * math.pi * s) ** 2 * f2 * np.exp(-2 * math.pi ** 2 * s ** 2 * f2)
        out.append(np.fft.irfft2(V * h, s=vp.shape)[pad:-pad, pad:-pad])
    return np.stack(out)


def peaks2d(R, radii, sp_mm, k, valid=None, min_resp=0.0):
    from scipy import ndimage as ndi
    mx = ndi.maximum_filter(R, size=(1, 3, 3))
    is_max = (R >= mx) & (R > min_resp)
    up = np.concatenate([R[1:], np.full_like(R[:1], -np.inf)])
    dn = np.concatenate([np.full_like(R[:1], -np.inf), R[:-1]])
    is_max &= (R >= up) & (R >= dn)
    if valid is not None:
        is_max &= valid[None]
    idx = np.argwhere(is_max)
    vals = R[is_max]
    order = np.argsort(-vals)
    keep = []
    for i in order:
        c = idx[i, 1:] * sp_mm
        r = radii[idx[i, 0]]
        if all(np.hypot(*(c - idx[j, 1:] * sp_mm)) >= max(r, radii[idx[j, 0]]) for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return [(int(idx[i, 1]), int(idx[i, 2]), float(radii[idx[i, 0]])) for i in keep]


def ssa_illustration(ct, tum, fg):
    """On a 128 x 128 grid (0.779 mm, the plan spacing): illustrative p_t, standardised foreground CT,
    |R| at three radii, semantic and image anchors."""
    from scipy import ndimage as ndi
    import matplotlib.cm as cm
    sp = 0.7791720032691956
    g = np.asarray(ct.convert("L").resize((128, 128), Image.LANCZOS)).astype(float) / 255.0
    t = np.asarray(Image.fromarray(tum.astype(np.uint8) * 255).resize((128, 128), Image.BILINEAR)) > 127
    f = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((128, 128), Image.BILINEAR)) > 127
    pt = 0.9 * ndi.gaussian_filter(t.astype(float), 1.2)
    v = np.where(f, (g - g[f].mean()) / (g[f].std() + 1e-6), 0.0)
    radii = np.array([2.5, 4.0, 6.0, 9.0, 13.5, 20.0])
    r_sem = log2d(pt, sp, radii)
    r_img = np.abs(log2d(v, sp, radii))
    sem = peaks2d(r_sem, radii, sp, 2, min_resp=0.02)
    img = peaks2d(r_img, radii, sp, 6, valid=ndi.binary_erosion(f, iterations=2))

    def cmap(a, name):
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        return (getattr(cm, name)(a)[..., :3] * 255).astype(np.uint8)

    up = lambda arr: np.asarray(Image.fromarray(arr).resize((256, 256), Image.NEAREST))
    thumbs = dict(pt=up(cmap(pt, "magma")), v=up(cmap(np.clip(v, -3, 3), "gray")),
                  r4=up(cmap(r_img[1], "viridis")), r9=up(cmap(r_img[3], "viridis")),
                  r20=up(cmap(r_img[5], "viridis")))
    return thumbs, sem, img, sp


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def L(lang, en, zh):
    return en if lang == "en" else zh


def panel(s, x, y, w, h, tag, title, fill=C["panel"]):
    s.rect(x, y, w, h, fill=fill, stroke=C["panel_stroke"], sw=1.0, rx=10)
    s.text(x + 14, y + 30, tag, size=22, weight="bold")
    s.text(x + 56, y + 31, title, size=19, weight="bold")


def build(lang: str) -> str:
    s = Svg(lang)
    T = lambda en, zh: L(lang, en, zh)
    ct, ov, tum, fg = material()
    ct_png, ov_png = png(np.asarray(ct)), png(np.asarray(ov))
    thumbs, sem_a, img_a, sp = ssa_illustration(ct, tum, fg)

    # ======================= (a) overview =======================
    panel(s, 8, 8, 1584, 818, "(a)",
          T("SALT: spectral-anchored lesion transformer on a VoComni-pretrained nnU-Net (no DCA / CFC)",
            "SALT：VoComni 预训练 nnU-Net 上的谱尺度空间锚定病灶 Transformer（不含 DCA / CFC）"))
    Y = [175, 280, 385, 490, 595]
    xe = [250 + 8 * i for i in range(5)]
    xd, bw, bh = 760, 110, 60
    sizes = ["32×128³", "64×64³", "128×32³", "256×16³", "320×8³"]

    # CT input (stacked sheets) and the CT line to SSA
    for k, dx in enumerate((16, 8)):
        s.rect(28 + dx, 112 - dx, 150, 150, fill="#e3e3e3", stroke="#555", sw=1)
    s.image(28, 112, 150, 150, ct_png)
    s.text(103, 290, T("CT patch", "CT 图块"), size=17, anchor="middle")
    s.text(103, 311, "1×128³", size=15, anchor="middle")
    s.line([(178, Y[0]), (xe[0] - 2, Y[0])])
    s.line([(103, 104), (103, 66), (1187, 66), (1187, 88)], color=C["grey"], sw=1.4)
    s.text(640, 60, T("CT intensities x (input of the image anchors)", "CT 强度 x（图像锚点的输入）"),
           size=13, anchor="middle", fill=C["grey"])

    # encoder, skips, FAF, decoder
    for i in range(5):
        s.cuboid(xe[i], Y[i] - bh / 2, bw, bh, C["enc"], f"E{i + 1}")
        if i < 4:
            s.line([(xe[i] + 55, Y[i] + bh / 2), (xe[i] + 55, Y[i + 1] - bh / 2 - 12)], color=C["down"], sw=2.2)
        s.text(xe[i] + bw + 30, Y[i] - 9, sizes[i], size=14, fill="#333")
        s.cuboid(xd, Y[i] - bh / 2, bw, bh, C["dec"], f"D{i + 1}", fill_txt="#1d3b66")
        if i in (1, 2, 3):
            fx0 = 568
            s.line([(xe[i] + bw + 18, Y[i]), (fx0 - 2, Y[i])])
            s.box(fx0, Y[i] - 26, 116, 52, C["faf"], C["faf_dark"], title="FAF-3D",
                  rows=[T("α, β = 0 at init.", "α、β 初值为 0")], size=11.5, tsize=15)
            s.line([(fx0 + 116, Y[i]), (xd - 2, Y[i])])
        else:
            s.line([(xe[i] + bw + 18, Y[i]), (xd - 2, Y[i])])
    # bottleneck
    s.cuboid(470, 660, bw, bh, C["enc"], "E6")
    s.text(525, 744, T("320×4³ (bottleneck)", "320×4³（瓶颈层）"), size=14, anchor="middle", fill="#333")
    s.line([(xe[4] + 55, Y[4] + bh / 2), (xe[4] + 55, 690), (468, 690)], color=C["down"], sw=2.2)
    s.line([(580 + 16, 690), (815, 690), (815, Y[4] + bh / 2 + 2)], color=C["up"], sw=2.2)
    # up-sampling: into FAF for D2-D4, directly into D1
    for i in (1, 2, 3):
        s.line([(778, Y[i + 1] - bh / 2), (778, Y[i] + 52), (626, Y[i] + 52), (626, Y[i] + 28)],
               color=C["up"], sw=2.2)
    s.line([(815, Y[1] - bh / 2 - 12), (815, Y[0] + bh / 2 + 2)], color=C["up"], sw=2.2)

    # semantic head and p
    s.line([(xd + bw + 18, Y[0] - 14), (903, Y[0] - 14)])
    s.box(905, Y[0] - 40, 128, 50, "white", "#333", title=T("Semantic head", "语义头"),
          rows=[T("1×1³ conv, softmax", "1×1³ 卷积 + softmax")], size=12, tsize=14, rx=6)
    s.text(969, Y[0] + 30, T("deep supervision on D1–D4", "D1–D4 深监督"), size=12, anchor="middle", fill=C["grey"])
    s.text(969, Y[0] + 46, "(Dice + CE)", size=12, anchor="middle", fill=C["grey"])
    # p_t into SSA and (along the top) into the fusion
    s.line([(1033, Y[0] - 15), (1073, Y[0] - 15)], color=C["pt"], sw=2, dash="7 4")
    s.text(1053, Y[0] - 22, "p_{t}", size=14, anchor="middle", fill=C["pt"], italic=True)
    s.line([(969, Y[0] - 40), (969, 80), (1565, 80), (1565, 418)], color=C["pt"], sw=1.8, dash="7 4")
    s.text(1430, 74, T("p = (p_{b}, p_{l}, p_{t}) to the fusion", "p = (p_{b}, p_{l}, p_{t}) 送入融合"),
           size=13, anchor="middle", fill=C["pt"])

    # SSA box (top right of the decoder column)
    s.box(1075, 90, 225, 122, C["ssa_bg"], C["ssa_dark"], sw=1.6)
    s.text(1187, 113, T("SSA: spectral anchors", "SSA：谱尺度空间锚点"), size=15, anchor="middle", weight="bold",
           fill=C["ssa_dark"])
    s.lines(1085, 135, [T("Fourier LoG bank, r = 2.5–20 mm", "傅里叶域 LoG 组，r = 2.5–20 mm"),
                        T("space–scale maxima → (c, r)", "空间–尺度极值 → (c, r)"),
                        T("16 from p_{t} + 16 from |R(x)|", "p_{t} 取 16 个 + |R(x)| 取 16 个")],
            size=12.5, gap=19)
    # anchors into LQD
    s.line([(1187, 212), (1187, 236)], color=C["anchor"], sw=2, dash="6 3")
    s.text(1195, 228, T("anchors", "锚点"), size=12, fill=C["anchor"])

    # pixel embedding and memory into LQD
    s.line([(xd + bw + 18, Y[0] + 16), (1046, Y[0] + 16), (1046, 262), (1073, 262)], color=C["lqd_dark"], sw=2)
    s.text(1040, 251, "φ: 32×128³", size=12, anchor="end", fill=C["lqd_dark"])
    for i, lv in ((1, "64³"), (2, "32³"), (3, "16³")):
        s.line([(xd + bw + 18, Y[i]), (1073, Y[i])], color=C["lqd_dark"], sw=2)
        s.text(960, Y[i] - 7, T(f"memory {lv}", f"记忆 {lv}"), size=12.5, anchor="middle", fill=C["lqd_dark"])

    # LQD block
    s.rect(1075, 240, 225, 330, fill=C["lqd_bg"], stroke=C["lqd_dark"], sw=1.6, rx=8)
    s.text(1187, 262, T("Lesion query decoder", "病灶查询解码器"), size=15, anchor="middle", weight="bold",
           fill=C["lqd_dark"])
    # query tokens
    for k in range(40):
        col = C["ssa_dark"] if k < 16 else (C["ssa"] if k < 32 else C["lqd"])
        s.rect(1088 + (k % 20) * 10, 274 + (k // 20) * 11, 8, 8, fill=col, rx=1.5)
    s.text(1187, 310, T("16 + 16 anchored + 8 free queries", "16 + 16 个锚定查询 + 8 个自由查询"), size=11.5,
           anchor="middle", fill="#333")
    for k, (lab, yy) in enumerate(((T("masked cross-attn.", "掩膜交叉注意力"), 322), (T("self-attention", "查询自注意力"), 356),
                                   ("FFN", 390))):
        s.rect(1098, yy, 150, 26, fill="white", stroke=C["lqd_dark"], sw=1.1, rx=5)
        s.text(1173, yy + 18, lab, size=12.5, anchor="middle")
        if k < 2:
            s.line([(1173, yy + 26), (1173, yy + 33)], color=C["lqd_dark"], sw=1.2)
    s.rect(1092, 318, 162, 102, stroke=C["lqd_dark"], sw=1.1, rx=6, dash="4 3")
    s.text(1262, 375, "×6", size=16, weight="bold", fill=C["lqd_dark"])
    s.text(1187, 438, T("levels 16³ → 32³ → 64³, round robin", "16³ → 32³ → 64³ 轮流"), size=11.5, anchor="middle",
           fill="#333")
    s.box(1088, 450, 96, 54, C["smd"], C["smd_dark"], title=T("objectness", "病灶置信度"),
          rows=["s_{q} ← SMD"], size=12, tsize=12.5, rx=6)
    s.box(1192, 450, 96, 54, "white", C["lqd_dark"], title=T("mask", "掩膜"), rows=["m_{q} = e_{q}·φ"],
          size=12, tsize=12.5, rx=6)
    s.text(1187, 524, T("training: Hungarian set loss", "训练：匈牙利集合损失"), size=11.5, anchor="middle", fill="#333")
    s.text(1187, 541, T("on lesion instances", "（病灶实例）"), size=11.5, anchor="middle", fill="#333")
    s.text(1187, 560, "L = L_{DiceCE} + 0.2 L_{set}", size=11.5, anchor="middle", fill="#333")

    # per-query outputs (schematic shapes)
    s.line([(1300, 400), (1323, 400)], color=C["lqd_dark"], sw=2)
    s.rect(1325, 250, 255, 150, fill="white", stroke=C["lqd_dark"], sw=1.2, rx=8)
    s.text(1452, 272, T("one mask + one score per query", "每个查询：一个掩膜 + 一个置信度"), size=12.5, anchor="middle",
           weight="bold", fill=C["lqd_dark"])
    shapes = [("blob", T("lesion", "病灶"), "s_{q} ↑"), ("tube", T("vessel", "血管"), "s_{q} ↓"),
              ("sheet", T("capsule", "包膜"), "s_{q} ↓")]
    for k, (kind, lab, sc) in enumerate(shapes):
        cx, cy = 1370 + k * 82, 322
        s.rect(cx - 32, cy - 32, 64, 64, fill="#111", rx=3)
        if kind == "blob":
            s.circle(cx, cy, 15, fill="#e8e8e8", stroke="none")
        elif kind == "tube":
            s.add(f'<path d="M {cx - 26} {cy + 20} Q {cx} {cy - 8} {cx + 26} {cy - 22}" stroke="#e8e8e8" '
                  f'stroke-width="7" fill="none" stroke-linecap="round"/>')
        else:
            s.add(f'<path d="M {cx - 28} {cy - 26} Q {cx - 10} {cy + 4} {cx - 20} {cy + 30}" stroke="#e8e8e8" '
                  f'stroke-width="3" fill="none"/>')
        s.text(cx, cy + 50, lab, size=12, anchor="middle")
        s.text(cx, cy + 66, sc, size=12.5, anchor="middle", weight="bold",
               fill=C["lqd_dark"] if "↑" in sc else C["up"])

    # fusion and output
    s.line([(1452, 400), (1452, 416)], color=C["lqd_dark"], sw=2)
    s.box(1325, 418, 255, 92, C["fuse"], "#b39012", title=T("Fusion (w = 0.5)", "融合（w = 0.5）"),
          rows=["P(v) = max_{q} s_{q} σ(m_{q}(v))", "p′_{t} = w p_{t} + (1−w) P (p_{l} + p_{t})"], size=12.5,
          tsize=14, rx=8)
    s.line([(1452, 510), (1452, 534)])
    s.image(1392, 536, 120, 120, ov_png)
    s.text(1452, 676, T("Segmentation", "分割结果"), size=15, anchor="middle")
    s.text(1452, 694, T("bg / liver / tumour", "背景 / 肝 / 肿瘤"), size=12.5, anchor="middle", fill="#333")
    s.text(1452, 711, T("(w = 1: semantic head only)", "（w = 1 即仅语义头）"), size=11.5, anchor="middle",
           fill=C["grey"])

    # legend
    ly = 790
    items = [("cub", C["enc"], T("VoComni encoder", "VoComni 编码器")),
             ("cub", C["dec"], T("nnU-Net decoder", "nnU-Net 解码器")),
             ("sq", (C["faf"], C["faf_dark"]), T("FAF-3D (new)", "FAF-3D（新）")),
             ("sq", (C["ssa_bg"], C["ssa_dark"]), T("SSA (new)", "SSA（新）")),
             ("sq", (C["lqd_bg"], C["lqd_dark"]), T("LQD (new)", "LQD（新）")),
             ("sq", (C["smd"], C["smd_dark"]), T("SMD (new)", "SMD（新）")),
             ("ln", C["down"], T("down-sampling", "下采样")), ("ln", C["up"], T("up-sampling", "上采样")),
             ("dash", C["pt"], T("tumour prob. p_{t}", "肿瘤概率 p_{t}")),
             ("dash", C["anchor"], T("anchors", "锚点"))]
    x = 30
    for kind, col, lab in items:
        if kind == "cub":
            s.rect(x, ly - 12, 24, 15, fill=col[0], stroke="#2b2b2b", sw=0.7)
        elif kind == "sq":
            s.rect(x, ly - 12, 24, 15, fill=col[0], stroke=col[1], sw=1.2, rx=3)
        else:
            s.line([(x, ly - 5), (x + 26, ly - 5)], color=col, sw=2.2, dash="6 3" if kind == "dash" else None)
        s.text(x + 32, ly, lab, size=13)
        x += 32 + 7.0 * len(lab.replace("_{", "").replace("}", "")) + (24 if lang == "en" else 70)

    # ======================= (b) SSA =======================
    panel(s, 8, 838, 792, 392, "(b)", T("SSA: spectral scale-space anchors", "SSA：谱尺度空间锚点"), fill=C["ssa_bg"])
    s.image(26, 880, 112, 112, png(thumbs["pt"]))
    s.text(82, 1008, T("tumour prob. p_{t}", "肿瘤概率 p_{t}"), size=12.5, anchor="middle")
    s.image(26, 1030, 112, 112, png(thumbs["v"]))
    s.text(82, 1158, T("CT, standardised", "CT（前景内标准化）"), size=12.5, anchor="middle")
    s.text(82, 1174, T("inside foreground", ""), size=12.5, anchor="middle")
    s.line([(140, 936), (166, 936), (166, 1000), (178, 1000)], color=C["ssa_dark"], sw=1.6)
    s.line([(140, 1086), (166, 1086), (166, 1020), (178, 1020)], color=C["ssa_dark"], sw=1.6)
    s.box(180, 985, 46, 50, "white", C["ssa_dark"], title="ℱ", rows=[], tsize=22, rx=6)
    s.line([(226, 1010), (246, 1010)], color=C["ssa_dark"], sw=1.6)
    # band-pass bank plot
    px, py, pw, ph = 250, 898, 230, 186
    s.rect(px, py, pw, ph, fill="white", stroke=C["ssa_dark"], sw=1.2, rx=6)
    s.text(px + pw / 2, py + 18, T("band-pass bank H_{σ}(|f|)", "带通滤波组 H_{σ}(|f|)"), size=13, anchor="middle",
           weight="bold")
    ax0, ay0, aw, ah = px + 26, py + ph - 40, pw - 44, ph - 74
    s.line([(ax0, ay0), (ax0 + aw + 6, ay0)], color="#444", sw=1)
    s.line([(ax0, ay0), (ax0, ay0 - ah - 6)], color="#444", sw=1)
    lo_f, hi_f = math.log10(0.004), math.log10(0.6)
    xf = lambda f_: ax0 + aw * (math.log10(f_) - lo_f) / (hi_f - lo_f)
    cols = ["#4a3384", "#6c50a8", "#8c70c4", "#a991d6", "#c2b2e6", "#d9cff0"]     # 20 mm (dark) ... 2.5 mm
    for r, col in zip((20, 13.5, 9, 6, 4, 2.5), cols):
        sig = r / math.sqrt(3)
        pts = []
        for j in range(161):
            f_ = 10 ** (lo_f + (hi_f - lo_f) * j / 160)
            hval = (2 * math.pi * sig * f_) ** 2 * math.exp(-2 * math.pi ** 2 * sig ** 2 * f_ ** 2)
            pts.append((xf(f_), ay0 - ah * hval / 0.8))
        s.line(pts, color=col, sw=2, arrow=False)
    for f_, lab in ((0.01, "0.01"), (0.1, "0.1"), (0.5, "0.5")):
        s.line([(xf(f_), ay0), (xf(f_), ay0 + 4)], color="#444", sw=1, arrow=False)
        s.text(xf(f_), ay0 + 16, lab, size=11, anchor="middle")
    s.text(px + pw / 2, ay0 + 31, T("|f| (cycles/mm, log scale)", "|f|（cycles/mm，对数坐标）"), size=11, anchor="middle",
           italic=True)
    s.text(xf(0.0196) - 4, ay0 - ah * 0.92 / 0.8 * 0.8 - 4, "20", size=11, anchor="middle", fill=cols[0])
    s.text(xf(0.156) + 4, ay0 - ah * 0.92 / 0.8 * 0.8 - 4, "2.5 mm", size=11, anchor="middle", fill="#8c70c4")
    s.lines(px + 6, py + ph + 24, ["H_{σ} = (2πσ|f|)^{2} e^{−2π²σ²|f|²}",
                                   T("σ = r/√3, r ∈ {2.5, 4, 6, 9, 13.5, 20} mm", "σ = r/√3，r ∈ {2.5, 4, 6, 9, 13.5, 20} mm")],
            size=12.5, gap=18)
    s.line([(px + pw, 1010), (px + pw + 22, 1010)], color=C["ssa_dark"], sw=1.6)
    s.box(504, 985, 50, 50, "white", C["ssa_dark"], title="ℱ⁻¹", rows=[], tsize=19, rx=6)
    # response stack
    s.line([(554, 1010), (572, 1010)], color=C["ssa_dark"], sw=1.6)
    for k, (key, lab) in enumerate((("r4", "r = 4 mm"), ("r9", "9 mm"), ("r20", "20 mm"))):
        yy = 876 + k * 100
        s.image(579, yy, 78, 78, png(thumbs[key]))
        s.text(618, yy + 92, f"|R|, {lab}" if k == 0 else lab, size=11.5, anchor="middle")
    s.text(618, 1188, "R_{σ} = −σ²∇²(G_{σ} ∗ v)", size=12, anchor="middle")
    s.line([(662, 1010), (680, 1010)], color=C["ssa_dark"], sw=1.6)
    # anchors on the CT
    ix, iy, isz = 684, 880, 104
    s.image(ix, iy, isz, isz, ct_png)
    k_ = isz / 128.0
    for (yy, xx, r) in img_a:
        s.circle(ix + (xx + 0.5) * k_, iy + (yy + 0.5) * k_, r / sp * k_, stroke="#ffe066", sw=1.7, dash="3 2")
    for (yy, xx, r) in sem_a:
        s.circle(ix + (xx + 0.5) * k_, iy + (yy + 0.5) * k_, r / sp * k_, stroke="#c58bff", sw=2.4)
    s.lines(ix + isz / 2, iy + isz + 18, [T("anchors (c, r)", "锚点 (c, r)"), T("── from p_{t}", "── 来自 p_{t}"),
                                          T("- - from |R(x)|", "- - 来自 |R(x)|")], size=11.5, gap=16, anchor="middle")
    s.box(682, 1052, 108, 62, "white", C["ssa_dark"], title=T("maxima", "极值"),
          rows=[T("space × scale", "空间 × 尺度"), "+ NMS"], size=11.5, tsize=12.5, rx=6)
    s.lines(682, 1136, [T("|f| in cycles/mm:", "|f| 以 cycles/mm 计："), T("anisotropic spacing", "各向异性间距"),
                        T("handled exactly", "被精确处理"), T("no gradient", "不回传梯度")],
            size=11.5, gap=16, fill="#333")
    s.text(26, 1218, T("Thumbnails: one CT slice; p_{t} illustrative; anchors and |R| computed on it with the 2D analogue (σ = r/√2).",
                       "缩略图：同一张 CT 切片；p_{t} 为示意；锚点与 |R| 用二维形式（σ = r/√2）在该切片上实算。"),
           size=11, fill=C["grey"])

    # ======================= (c) LQD =======================
    panel(s, 808, 838, 784, 392, "(c)", T("Lesion query decoder (LQD) and set loss", "病灶查询解码器（LQD）与集合损失"),
          fill=C["lqd_bg"])
    # queries
    s.rect(826, 880, 210, 128, fill="white", stroke=C["lqd_dark"], sw=1.2, rx=6)
    s.text(931, 900, T("queries q (40 × 128)", "查询 q（40 × 128）"), size=13, anchor="middle", weight="bold")
    for k in range(40):
        col = C["ssa_dark"] if k < 16 else (C["ssa"] if k < 32 else C["lqd"])
        s.rect(838 + (k % 20) * 9.5, 910 + (k // 20) * 10, 7.5, 7.5, fill=col, rx=1.2)
    s.lines(834, 948, [T("anchored: φ_{mem}(c) + e_{src} + MLP(r, R)", "锚定：φ_{mem}(c) + e_{src} + MLP(r, R)"),
                       T("   position: MLP(PE(c))", "   位置：MLP(PE(c))"),
                       T("free: learned content, position", "自由：可学习的内容与位置")], size=11.5, gap=17)
    # memory pyramid
    s.rect(826, 1020, 210, 116, fill="white", stroke=C["lqd_dark"], sw=1.2, rx=6)
    s.text(931, 1040, T("memory (D4, D3, D2)", "记忆（D4、D3、D2）"), size=13, anchor="middle", weight="bold")
    for k, (lv, sz) in enumerate((("16³", 18), ("32³", 26), ("64³", 34))):
        cx = 862 + k * 62
        s.cuboid(cx - sz / 2, 1092 - sz / 2, sz, sz, C["dec"], "", d=8)
        s.text(cx + 4, 1128, lv, size=12, anchor="middle")
    s.text(931, 1150, T("1×1³ → 128, + level emb. + 3D PE", "1×1³ → 128，加层嵌入与 3D 位置编码"), size=11.5,
           anchor="middle", fill="#333")
    # layers
    s.rect(1062, 880, 236, 240, fill="white", stroke=C["lqd_dark"], sw=1.4, rx=8, dash="5 3")
    s.text(1288, 904, "×6", size=18, anchor="end", weight="bold", fill=C["lqd_dark"])
    blocks = [(T("masked cross-attention", "掩膜交叉注意力"),
               [T("allowed: σ(m_{q}) ≥ 0.5 ∪ ball(c_{q}, 2r_{q})", "可见：σ(m_{q}) ≥ 0.5 ∪ 球(c_{q}, 2r_{q})"),
                T("levels 16³ → 32³ → 64³ (round robin)", "16³ → 32³ → 64³ 轮流")]),
              (T("query self-attention", "查询自注意力"), [T("4 heads, d = 128, float32", "4 头，d = 128，float32")]),
              ("FFN", ["128 → 512 → 128, post-LN"])]
    yy = 914
    for k, (tl, rows) in enumerate(blocks):
        h_ = 26 + 16 * len(rows)
        s.box(1074, yy, 212, h_, C["lqd_bg"], C["lqd_dark"], title=tl, rows=rows, size=11, tsize=12.5, rx=6)
        if k < 2:
            s.line([(1180, yy + h_), (1180, yy + h_ + 10)], color=C["lqd_dark"], sw=1.3)
        yy += h_ + 12
    s.line([(1036, 944), (1060, 944)], color=C["lqd_dark"], sw=1.6)
    s.line([(1036, 1078), (1050, 1078), (1050, 960), (1072, 960)], color=C["lqd_dark"], sw=1.6)
    # heads
    s.line([(1298, 1000), (1320, 1000)], color=C["lqd_dark"], sw=1.6)
    s.box(1322, 880, 252, 74, C["smd"], C["smd_dark"], title=T("objectness head", "置信度头"),
          rows=["s_{q} = σ(Lin(LN(q) + MLP(d_{q})))", T("d_{q}: SMD of the mask (panel e)", "d_{q}：掩膜的 SMD（图 e）")],
          size=11.5, tsize=12.5, rx=6)
    s.box(1322, 964, 252, 74, "white", C["lqd_dark"], title=T("mask head", "掩膜头"),
          rows=["m_{q}(v) = e_{q}^{T} φ(v)", T("e_{q} = MLP(LN(q)) ∈ R^{32}; φ: 32×128³", "e_{q} = MLP(LN(q)) ∈ R^{32}；φ：32×128³")],
          size=11.5, tsize=12.5, rx=6)
    # training / inference
    s.box(826, 1162, 480, 60, "white", "#555",
          rows=[T("Train: Hungarian matching per layer to 26-conn. tumour components (≥ 20 vox);",
                  "训练：逐层把查询匈牙利匹配到肿瘤 26 连通域（≥ 20 体素）；"),
                T("cost 2·cls + 5·BCE + 5·Dice on 4096 uniform + 512 points / lesion",
                  "代价 2·cls + 5·BCE + 5·Dice，4096 个均匀点 + 每病灶 512 点"),
                T("L = L_{DiceCE} + 0.2 ψ(e) L_{set}  (7 predictions averaged)", "L = L_{DiceCE} + 0.2 ψ(e) L_{set}（7 个预测取平均）")],
          size=11, center=False, rx=6)
    s.box(1322, 1050, 252, 172, C["fuse"], "#b39012", title=T("Inference", "推理"),
          rows=["P(v) = max_{q} s_{q} σ(m_{q}(v))", "p′_{t} = w p_{t} + (1 − w) P (p_{l} + p_{t})",
                T("w = 0.5; liver = rest of foreground", "w = 0.5；肝 = 其余前景"),
                T("w = 1 ⇒ semantic head only", "w = 1 ⇒ 仅语义头（内置消融）"),
                T("one score per lesion ⇒ FROC", "每个病灶一个置信度 ⇒ FROC")],
          size=12, tsize=13.5, rx=8)

    # ======================= (d) FAF-3D =======================
    panel(s, 8, 1242, 792, 362, "(d)", T("FAF-3D: frequency-aware decoder fusion (at D2, D3, D4)",
                                         "FAF-3D：频率感知解码融合（D2、D3、D4）"), fill=C["faf_bg"])
    s.cuboid(28, 1300, 70, 46, C["dec"], "up", d=10, size=14, fill_txt="#1d3b66")
    s.text(118, 1306, T("TransConv(D_{l+1})", "转置卷积(D_{l+1})"), size=11.5)
    s.cuboid(28, 1500, 70, 46, C["enc"], "skip", d=10, size=14)
    s.text(118, 1510, T("encoder E_{l}", "编码器 E_{l}"), size=11.5)
    # kernel generator (from both inputs)
    s.box(150, 1392, 196, 76, "white", C["faf_dark"], title=T("kernel generator", "核生成器"),
          rows=[T("1×1³ compress (32 + 32)", "1×1³ 压缩（32 + 32）"), T("3×3³ conv → softmax over k³", "3×3³ 卷积 → k³ 上 softmax")],
          size=11.5, tsize=12.5, rx=6)
    s.line([(66, 1352), (66, 1376), (180, 1376), (180, 1390)], color=C["faf_dark"], sw=1.3)
    s.line([(66, 1494), (66, 1482), (180, 1482), (180, 1470)], color=C["faf_dark"], sw=1.3)
    # per-voxel kernels
    rng = np.random.default_rng(3)
    for k, (kk, y0, lab) in enumerate(((5, 1372, T("W_{low}: k = 5 (1/8, 1/4), 3 (1/2)", "W_{low}：k = 5（1/8、1/4），3（1/2）")),
                                       (3, 1436, T("W_{high}: k = 3", "W_{high}：k = 3")))):
        cell = 8
        g_ = np.exp(-0.5 * ((np.arange(kk) - kk // 2)[:, None] ** 2 + (np.arange(kk) - kk // 2)[None] ** 2) / 1.5)
        g_ = g_ * (1 + 0.25 * rng.standard_normal(g_.shape))
        g_ = (g_ - g_.min()) / (g_.max() - g_.min())
        x0 = 372
        for a in range(kk):
            for b in range(kk):
                v_ = g_[a, b]
                colr = f"rgb({int(255 - 70 * v_)},{int(240 - 120 * v_)},{int(220 - 180 * v_)})"
                s.rect(x0 + b * cell, y0 + a * cell, cell, cell, fill=colr, stroke="#b8641c", sw=0.4)
        s.text(x0 + kk * cell + 8, y0 + kk * cell / 2 + 4, lab, size=11, fill="#333")
    s.line([(346, 1410), (360, 1410), (360, 1392), (370, 1392)], color=C["faf_dark"], sw=1.3)
    s.line([(346, 1450), (370, 1450)], color=C["faf_dark"], sw=1.3)
    s.line([(392, 1372), (392, 1356), (500, 1356), (500, 1350)], color=C["faf_dark"], sw=1.3)
    s.line([(384, 1460), (384, 1478), (500, 1478), (500, 1492)], color=C["faf_dark"], sw=1.3)
    # up branch: adaptive low-pass, gated residual
    s.line([(108, 1322), (466, 1322)], color=C["ink"], sw=1.5)
    s.box(468, 1282, 210, 68, C["faf"], C["faf_dark"], title=T("adaptive low-pass (ALPF)", "自适应低通（ALPF）"),
          rows=["Σ_{j} W_{low,j} · up(v + o_{j})", "up′ = up + α (ALPF(up) − up)"], size=11.5, tsize=12.5, rx=6)
    # skip branch: adaptive high-pass, gated residual
    s.line([(108, 1522), (466, 1522)], color=C["ink"], sw=1.5)
    s.box(468, 1494, 210, 68, C["faf"], C["faf_dark"], title=T("adaptive high-pass", "自适应高通"),
          rows=["HP = skip − Σ_{j} W_{high,j} skip(v + o_{j})", "skip′ = skip + β · HP(skip)"], size=11.5, tsize=12.5,
          rx=6)
    # frequency-response sketch
    fx0, fy0 = 552, 1460
    s.line([(fx0, fy0), (fx0 + 118, fy0)], color="#777", sw=0.9, arrow=False)
    s.line([(fx0, fy0), (fx0, fy0 - 62)], color="#777", sw=0.9, arrow=False)
    for kind in ("low", "high"):
        pts = []
        for j in range(61):
            f_ = j / 60
            hv = math.exp(-(f_ / 0.35) ** 2) if kind == "low" else 1 + 0.6 * (1 - math.exp(-(f_ / 0.35) ** 2))
            pts.append((fx0 + 112 * f_, fy0 - 34 * hv))
        s.line(pts, color=C["faf_dark"], sw=1.8, arrow=False, dash=None if kind == "low" else "5 3")
    s.text(fx0 + 60, fy0 + 15, T("|f| (schematic)", "|f|（示意）"), size=10.5, anchor="middle", fill=C["grey"])
    s.text(fx0 + 116, fy0 - 64, T("high-boost", "高频增强"), size=10.5, anchor="end", fill=C["faf_dark"])
    s.text(fx0 + 116, fy0 - 6, T("low-pass", "低通"), size=10.5, anchor="end", fill=C["faf_dark"])
    # concat and decoder stage
    s.line([(678, 1316), (712, 1316), (712, 1408)], color=C["ink"], sw=1.5)
    s.line([(678, 1528), (712, 1528), (712, 1442)], color=C["ink"], sw=1.5)
    s.circle(712, 1425, 14, fill="white", stroke="#333", sw=1.4)
    s.text(712, 1430, "C", size=13, anchor="middle", weight="bold")
    s.line([(726, 1425), (736, 1425)], color=C["ink"], sw=1.5)
    s.cuboid(738, 1403, 44, 44, C["dec"], "D_{l}", d=8, size=13, fill_txt="#1d3b66")
    s.lines(760, 1470, [T("2 × conv–", "2 × 卷积–"), "IN–LReLU"], size=10.5, gap=14, fill="#333", anchor="middle")
    s.text(26, 1592, T("α, β = 0 at initialisation ⇒ identical to the pretrained decoder; k³ shifted views summed, memory O(|x|); "
                       "3D, gated form of FreqFusion (TPAMI 2024).",
                       "α、β 初值为 0 ⇒ 初始化时与预训练解码器完全相同；k³ 个平移视图加权求和，显存 O(|x|)；FreqFusion（TPAMI 2024）的三维门控形式。"),
           size=11, fill=C["grey"])

    # ======================= (e) SMD =======================
    panel(s, 808, 1242, 784, 362, "(e)", T("SMD: spectral moment descriptor of each query mask",
                                           "SMD：查询掩膜的谱矩形状描述子"), fill=C["smd_bg"])
    s.text(1200, 1300, "T = Σ_{f} f f^{T} |M̂(f)|² / Σ_{f} |M̂(f)|²  =  (1/4π²) Σ_{v} ∇M ∇M^{T} / Σ_{v} M²",
           size=14, anchor="middle")
    s.text(1200, 1320, T("(Parseval; eigenvalues λ_{1} ≥ λ_{2} ≥ λ_{3})", "（Parseval 定理；特征值 λ_{1} ≥ λ_{2} ≥ λ_{3}）"),
           size=12, anchor="middle", fill="#333")
    cols3 = [("ball", T("blob: lesion", "团块：病灶"), (1.00, 0.93)),
             ("tube", T("tube: vessel", "管：血管"), (1.00, 0.00)),
             ("sheet", T("sheet: capsule, fissure", "片：包膜、裂隙"), (0.00, 0.00))]
    for k, (kind, lab, (r2, r3)) in enumerate(cols3):
        cx = 900 + k * 220
        cy = 1385
        # spatial shape
        if kind == "ball":
            s.add(f'<defs><radialGradient id="gb{lang}" cx="35%" cy="35%" r="70%"><stop offset="0" stop-color="#fbe3e8"/>'
                  f'<stop offset="1" stop-color="{C["smd"]}"/></radialGradient></defs>')
            s.circle(cx - 40, cy, 24, fill=f"url(#gb{lang})", stroke=C["smd_dark"], sw=1.2)
        elif kind == "tube":
            s.rect(cx - 74, cy - 11, 68, 22, fill=C["smd"], stroke=C["smd_dark"], sw=1.2, rx=11)
        else:
            s.poly([(cx - 78, cy + 16), (cx - 52, cy - 22), (cx - 2, cy - 22), (cx - 28, cy + 16)], C["smd"],
                   stroke=C["smd_dark"], sw=1.2, op=0.85)
        s.text(cx - 40, cy + 46, T("space", "空间"), size=11, anchor="middle", fill=C["grey"])
        # spectral energy
        s.line([(cx + 8, cy), (cx + 82, cy)], color="#999", sw=0.8, arrow=False)
        s.line([(cx + 45, cy - 34), (cx + 45, cy + 34)], color="#999", sw=0.8, arrow=False)
        if kind == "ball":
            s.circle(cx + 45, cy, 22, fill=C["smd"], stroke=C["smd_dark"], sw=1, op=0.45)
        elif kind == "tube":
            s.add(f'<ellipse cx="{cx + 45}" cy="{cy}" rx="10" ry="30" fill="{C["smd"]}" fill-opacity="0.45" '
                  f'stroke="{C["smd_dark"]}" stroke-width="1"/>')
        else:
            s.add(f'<ellipse cx="{cx + 45}" cy="{cy}" rx="34" ry="5" fill="{C["smd"]}" fill-opacity="0.55" '
                  f'stroke="{C["smd_dark"]}" stroke-width="1" transform="rotate(-55 {cx + 45} {cy})"/>')
        s.text(cx + 45, cy + 46, "|M̂(f)|²", size=11, anchor="middle", fill=C["grey"])
        s.text(cx + 2, cy + 72, lab, size=13, anchor="middle", weight="bold")
        # eigenvalue ratio bars
        for j, (name, val) in enumerate((("λ_{2}/λ_{1}", r2), ("λ_{3}/λ_{1}", r3))):
            by = cy + 88 + j * 24
            s.text(cx - 44, by + 12, name, size=12, anchor="end")
            s.rect(cx - 38, by, 110, 15, fill="white", stroke="#999", sw=0.8)
            s.rect(cx - 38, by, 110 * val, 15, fill=C["smd_dark"])
            s.text(cx + 78, by + 12, f"{val:.2f}", size=12)
    s.text(1200, 1556, T("d_{q} = [log tr T, λ_{2}/λ_{1}, λ_{3}/λ_{1}, log(1 + V)] → MLP (zero-init.) → objectness head (detached)",
                         "d_{q} = [log tr T, λ_{2}/λ_{1}, λ_{3}/λ_{1}, log(1 + V)] → MLP（零初始化）→ 置信度头（不回传梯度）"),
           size=12.5, anchor="middle")
    s.text(1200, 1580, T("Bars: values measured on soft ball / tube / sheet masks (experiments/test_salt.py).",
                         "条形：在软球 / 管 / 片掩膜上的实测值（experiments/test_salt.py）。"), size=11, anchor="middle",
           fill=C["grey"])
    return s.svg()


def main():
    for lang in ("en", "zh"):
        out = FIG / f"fig_salt_architecture_{lang}.svg"
        out.write_text(build(lang), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
