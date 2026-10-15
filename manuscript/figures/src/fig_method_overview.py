#!/usr/bin/env python3
"""Fig. 2, method overview, in a journal schematic style (zh and en).

    python manuscript/figures/src/fig_method_overview.py      # from the repo root

Data: method_case.npz (cut by extract_method_case.py from validation case
LiTS_006 on the server: 1-mm isotropic CT cube, labels, DCA applied in 3D with
Eq. (4) at target CNR 0.6, A13b seed-0 validation prediction) and the
low-CNR target quantiles used in training (phaser/lowcnr.py). Nothing here is
from the test set.
Writes manuscript/figures/fig2_method_{zh,en}.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle
from matplotlib.transforms import Affine2D
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_detect import INK, INK2, MM, OUT, mtext  # noqa: E402  (also sets rcParams)

HERE = Path(__file__).resolve().parent
LOW_CNR_QUANTILES = (0.0223, 0.1985, 0.3378, 0.4206, 0.4786, 0.5271, 0.5882, 0.6487, 0.7108,
                     0.7638, 0.8089, 0.872, 0.9263, 0.971, 1.0284, 1.086, 1.1446, 1.1772,
                     1.2278, 1.2545, 1.2777)   # phaser/lowcnr.py

# palette: muted journal tones
ORANGE, ORANGE_BG = '#d9662a', '#fdf2ea'
BLUE, BLUE_L, BLUE_M = '#2f5f98', '#dbe6f3', '#a9c1df'
GREEN, GREEN_BG = '#2a8a5c', '#e9f5ee'
GRAY_L, GRAY_M = '#eeeeec', '#c9c8c3'
LESION, RING = '#e0452b', '#27a36a'
WIN = (-60, 220)          # HU display window

TXT = {
    'zh': dict(
        x='训练图块 $x$', y='标签 $y$', xt='衰减图块 $\\tilde{x}$',
        dca='(a) 可检出性校准的对比度衰减（DCA）',
        s1='① 病灶 $M_k$ 与肝实质环带 $R_k$', s2='② 测量对比噪声比',
        s3='③ 从困难病灶分布抽取目标', s4='④ 病灶强度整体平移',
        cdf_x='CNR', cdf_y='$u$', prof='强度', inside='病灶',
        before='原始', after='衰减后', nets='(b) 共享权重的学生 / 教师网络',
        student='学生 $f_\\theta$：输入 $\\tilde{x}$', teacher='教师 $f_\\theta$：输入 $x$，不回传梯度',
        shared='共享权重 $\\theta$', sg='sg', enc='编码器', dec='解码器', taps='最细两层\n解码器特征',
        cfc='(c) 对比度不变特征一致性（CFC）', omega='区域 $\\Omega$：被衰减病灶膨胀 6 体素',
        pred='预测 $\\hat{y}$', total='总损失',
        infer='推理：只运行 $f_\\theta$，与基线完全相同 —— DCA 与 CFC 均不参与，无新增参数与计算',
        lg_fwd='前向', lg_sg='不回传梯度（sg）', lg_share='共享权重', lg_train='仅训练时使用',
        lg_feat='特征对齐',
    ),
    'en': dict(
        x='Training patch $x$', y='Label $y$', xt='Attenuated patch $\\tilde{x}$',
        dca='(a) Detectability-calibrated contrast attenuation (DCA)',
        s1='① Lesion $M_k$ and liver ring $R_k$', s2='② Measure lesion CNR',
        s3='③ Sample target CNR', s4='④ Shift lesion mean',
        cdf_x='CNR', cdf_y='$u$', prof='Intensity', inside='lesion',
        before='before', after='after', nets='(b) Shared-weight student and teacher',
        student='Student $f_\\theta$ on $\\tilde{x}$', teacher='Teacher $f_\\theta$ on $x$ (stop-gradient)',
        shared='shared $\\theta$', sg='sg', enc='encoder', dec='decoder', taps='two finest\ndecoder stages',
        cfc='(c) Contrast-invariant feature consistency (CFC)', omega='$\\Omega$: attenuated lesions dilated by 6 voxels',
        pred='Prediction $\\hat{y}$', total='Total loss',
        infer='Inference: only $f_\\theta$ runs, identical to the baseline (no extra parameters or computation)',
        lg_fwd='forward', lg_sg='stop-gradient (sg)', lg_share='shared weights', lg_train='training only',
        lg_feat='feature alignment',
    ),
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def arrow(ax, p, q, color=INK2, lw=0.8, ls='-', style='-|>', ms=7, conn='arc3', z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, connectionstyle=conn, zorder=z))


def panel(ax, x, y, w, h, title, ec, fc, size=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2', fc=fc, ec=ec, lw=0.8,
                                ls=(0, (4, 2)), zorder=0))
    mtext(ax, x + 2, y + h - 2.6, title, size=size, ha='left', color=ec, z=6)


def face(ax, img, M, z=3, cmap='gray', vmin=WIN[0], vmax=WIN[1], alpha=1.0):
    """Draw img (row 0 at the bottom) through the affine map M (3x3, image
    column/row -> data)."""
    h, w = img.shape
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=(0, w, 0, h), interpolation='bilinear',
              transform=Affine2D(np.asarray(M, float)) + ax.transData, zorder=z, alpha=alpha)


def cube(ax, vol, x0, y0, s, d, z0, depth, contour=None):
    """CT cube: front face = axial slice z0 (s x s), top and right faces show
    the next `depth` slices, oblique offset d = (dx, dy)."""
    dx, dy = d
    front = vol[z0][::-1]                       # anatomical up = up
    ny, nx = front.shape
    top = vol[z0:z0 + depth, 0, :]              # (depth, x): top row of the axial plane
    right = vol[z0:z0 + depth, ::-1, -1].T      # (y from bottom, depth)
    face(ax, front, [[s / nx, 0, x0], [0, s / ny, y0], [0, 0, 1]], z=3)
    face(ax, top, [[s / nx, dx / depth, x0], [0, dy / depth, y0 + s], [0, 0, 1]], z=2)
    face(ax, right, [[dx / depth, 0, x0 + s], [dy / depth, s / ny, y0], [0, 0, 1]], z=2)
    for poly in ([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)],
                 [(x0, y0 + s), (x0 + s, y0 + s), (x0 + s + dx, y0 + s + dy), (x0 + dx, y0 + s + dy)],
                 [(x0 + s, y0), (x0 + s + dx, y0 + dy), (x0 + s + dx, y0 + s + dy), (x0 + s, y0 + s)]):
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec=INK, lw=0.6, zorder=5))
    if contour is not None:
        m = contour[z0][::-1].astype(float)
        yy, xx = np.mgrid[0:ny, 0:nx]
        ax.contour(x0 + (xx + 0.5) * s / nx, y0 + (yy + 0.5) * s / ny, m, levels=[0.5], colors=LESION,
                   linewidths=0.6, zorder=4)


def cuboid(ax, x, y, w, h, d, fc, ec, z=3, lw=0.5):
    dx, dy = d
    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], fc=shade(fc, 0.82),
                   ec=ec, lw=lw, zorder=z)
    top = Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], fc=shade(fc, 0.92),
                  ec=ec, lw=lw, zorder=z)
    front = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z + 0.1)
    for p in (side, top, front):
        ax.add_patch(p)


def shade(c, f):
    c = np.array(plt.matplotlib.colors.to_rgb(c))
    return tuple(np.clip(c * f, 0, 1))


def unet(ax, x0, ymid, fc_enc, fc_dec, ec, tap_color, step=5.0):
    """3D-U-Net glyph: 4 encoder levels, bottleneck, 4 decoder levels; the two
    finest decoder stages are framed (CFC taps). Returns anchor points."""
    hs = [15, 11, 7.5, 5]                  # block heights per level
    ws = [1.5, 1.9, 2.3, 2.7]              # block thickness (channels)
    ydrop = [0, 4.0, 7.2, 9.6]             # vertical drop per level
    d = (1.3, 1.0)
    pos = []                               # (x, yc, h, w, kind, level)
    x = x0
    for l in range(4):
        pos.append((x, ymid - ydrop[l], hs[l], ws[l], 'enc', l)); x += step
    pos.append((x, ymid - 11.4, 3.6, 3.2, 'bot', 4)); x += step + 0.6
    for l in (3, 2, 1, 0):
        pos.append((x, ymid - ydrop[l], hs[l], ws[l], 'dec', l)); x += step
    for (xa, ya, ha, wa, *_), (xb, yb, hb, wb, *_) in zip(pos[:-1], pos[1:]):
        arrow(ax, (xa + wa + d[0] + 0.25, ya), (xb - 0.25, yb), color=ec, lw=0.45, ms=3.6, z=6)
    enc = {p[5]: p for p in pos if p[4] == 'enc'}; dec = {p[5]: p for p in pos if p[4] == 'dec'}
    for l in range(4):
        xe, ye, he, we, *_ = enc[l]; xd, yd, hd, wd, *_ = dec[l]
        yy = ye + he / 2 - 0.9
        ax.plot([xe + we + d[0], xd], [yy, yy], color='#a7a6a1', lw=0.45, ls=(0, (2, 1.3)), zorder=1)
    for xb, yb, hb, wb, kind, l in pos:
        fc = fc_enc if kind == 'enc' else shade(fc_enc, 0.88) if kind == 'bot' else fc_dec
        cuboid(ax, xb, yb - hb / 2, wb, hb, d, fc, ec)
    x1, y1 = dec[1][0] - 0.8, dec[1][1] - dec[1][2] / 2 - 0.8
    x2 = dec[0][0] + dec[0][3] + d[0] + 0.8
    y2 = dec[0][1] + dec[0][2] / 2 + d[1] + 0.8
    ax.add_patch(FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle='round,pad=0,rounding_size=0.8', fc='none',
                                ec=tap_color, lw=0.8, zorder=7))
    return dict(inp=(x0 - 0.3, ymid), out=(x2, dec[0][1]), tap=(x2, (y1 + y2) / 2), tap_box=(x1, y1, x2, y2),
                enc_mid=(enc[0][0] + enc[3][0]) / 2 + 1, dec_mid=(dec[3][0] + dec[0][0]) / 2 + 1, bottom=ymid - 13.4,
                right=x2)


def thumb(ax, img, x, y, w, h, z=3, cmap='gray', vmin=WIN[0], vmax=WIN[1], ec=INK):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + w, y, y + h), origin='upper',
              interpolation='bilinear', zorder=z)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.5, zorder=z + 0.2))


def overlay_contour(ax, mask, x, y, w, h, color, lw=0.6, ls='solid', z=4):
    ny, nx = mask.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ax.contour(x + (xx + 0.5) * w / nx, y + h - (yy + 0.5) * h / ny, mask.astype(float), levels=[0.5], colors=color,
               linewidths=lw, linestyles=ls, zorder=z)


def mini_axes(ax, x, y, w, h):
    ax.plot([x, x, x + w], [y + h, y, y], color=INK2, lw=0.5, zorder=4)


def circled(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 1.5, fc=color, ec='none', zorder=7))
    ax.text(x, y - 0.05, str(n), fontsize=5.2, color='white', ha='center', va='center', zorder=8, fontweight='bold')


# ---------------------------------------------------------------------------
def draw(lang):
    T = TXT[lang]
    zh = lang == 'zh'
    d = np.load(HERE / 'method_case.npz')
    ct, att, gt, les, ring, omega = d['ct'], d['att'], d['gt'], d['lesion'], d['ring'], d['omega']
    pred = d['pred_a13b']
    c = d['center']; cnr0, cnr1, alpha = float(d['cnr0']), float(d['cnr1']), float(d['alpha'])
    W, H = 200, 118
    fig = plt.figure(figsize=(190 * MM, 190 * MM * H / W))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')
    labcmap = plt.matplotlib.colors.ListedColormap(['#1d1d1b', '#9cbbe0', LESION])

    def labimg(a):
        return np.where(a == 2, 2, np.where(a == 1, 1, 0))

    zc, yc, xc = (int(v) for v in c)
    half = 22
    cut = (slice(None), slice(max(0, yc - half), yc + half), slice(max(0, xc - half), xc + half))
    cvol, avol, lvol = ct[cut], att[cut], les[cut]
    z0 = zc - 1
    S, DEP = 20, (6, 4.6)
    r0 = 14
    sl2 = (slice(yc - r0, yc + r0), slice(xc - r0, xc + r0))
    r1 = 20
    sl3 = (slice(yc - r1, yc + r1), slice(xc - r1, xc + r1))

    # ===== stage frames and tabs
    ax.add_patch(FancyBboxPatch((1, 16), W - 2, H - 19, boxstyle='round,pad=0,rounding_size=2.5', fc='none',
                                ec=GRAY_M, lw=0.6, zorder=0))
    ax.add_patch(FancyBboxPatch((1, 1.5), W - 2, 12, boxstyle='round,pad=0,rounding_size=2', fc='#fafaf8',
                                ec=GRAY_M, lw=0.6, zorder=0))
    for (tx_, ty_, lab_) in ((4, H - 3, '训练阶段' if zh else 'Training'), (4, 13.5, '推理阶段' if zh else 'Inference')):
        ax.text(tx_ + 0.6, ty_, lab_, fontsize=6.2, ha='left', va='center', color=INK2, zorder=9,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRAY_M, lw=0.5))

    # ===== input patch x
    xin, yin = 3, 76
    cube(ax, cvol, xin, yin, S, DEP, z0, 18, contour=lvol)
    mtext(ax, xin + S / 2 + 2, yin - 3.4, T['x'], size=6.4)

    # ===== DCA panel
    px, py, pw, ph = 33, 47, 64, 66
    panel(ax, px, py, pw, ph, T['dca'], ORANGE, ORANGE_BG)
    # row 1: step 1 thumbnail, step 2 formula
    ty1 = py + 35
    tx, ts = px + 4, 20
    thumb(ax, ct[zc][sl2], tx, ty1, ts, ts)
    overlay_contour(ax, les[zc][sl2], tx, ty1, ts, ts, LESION, lw=0.75)
    rg = ring[zc][sl2]
    ax.imshow(np.ma.masked_where(~rg, np.ones_like(rg, float)), cmap=plt.matplotlib.colors.ListedColormap([RING]),
              alpha=0.45, extent=(tx, tx + ts, ty1, ty1 + ts), origin='upper', zorder=3.5)
    circled(ax, px + 4.2, py + ph - 9.2, 1, ORANGE)
    mtext(ax, px + 6.6, py + ph - 9.2, T['s1'][2:], size=5.6, ha='left')
    fx = px + 36
    circled(ax, fx + 0.2, py + ph - 9.2, 2, ORANGE)
    mtext(ax, fx + 2.6, py + ph - 9.2, T['s2'][2:], size=5.6, ha='left')
    mtext(ax, fx + 12, ty1 + 12.4, '$\\mathrm{CNR}_k=\\dfrac{|\\mu_k-\\nu_k|}{s_k}$', size=7.2)
    mtext(ax, fx + 12, ty1 + 6.6, f'$= {cnr0:.2f}$', size=6.6, color=INK2)
    arrow(ax, (tx + ts + 1.4, ty1 + ts / 2), (fx - 0.2, ty1 + ts / 2), color=ORANGE, lw=0.7, ms=5)
    # row 2: step 3 inverse CDF, step 4 profile
    q = np.array(LOW_CNR_QUANTILES); u = np.linspace(0, 1, len(q))
    cx, cy_, cw, ch = px + 7, py + 11, 18, 13
    yt2 = cy_ + ch + 5.4
    circled(ax, px + 4.2, yt2, 3, ORANGE)
    mtext(ax, px + 6.6, yt2, T['s3'][2:], size=5.6, ha='left')
    mini_axes(ax, cx, cy_, cw, ch)
    xmax = 1.35
    ax.plot(cx + q / xmax * cw, cy_ + u * ch, color=ORANGE, lw=1.0, zorder=5)
    uu = 0.45; tt = float(np.interp(uu * (len(q) - 1), np.arange(len(q)), q))
    ax.plot([cx, cx + tt / xmax * cw, cx + tt / xmax * cw], [cy_ + uu * ch, cy_ + uu * ch, cy_], color=INK2, lw=0.5,
            ls=(0, (2, 1.2)), zorder=5)
    ax.text(cx + tt / xmax * cw, cy_ - 0.8, '$t_k$', fontsize=5.8, ha='center', va='top', color=INK)
    ax.text(cx - 0.8, cy_ + uu * ch, '$u$', fontsize=5.8, ha='right', va='center', color=INK)
    ax.text(cx + cw + 0.6, cy_, 'CNR', fontsize=5.0, ha='left', va='center', color=INK2)
    ax.text(cx + 1.0, cy_ + ch, '$F_{\\mathrm{low}}^{-1}$' if False else '$F_{\\mathrm{low}}$', fontsize=6, ha='left',
            va='top', color=ORANGE)
    gx, gy, gw, gh = px + 35, cy_, 18, ch
    circled(ax, gx - 2.2, yt2, 4, ORANGE)
    mtext(ax, gx + 0.2, yt2, T['s4'][2:], size=5.6, ha='left')
    mini_axes(ax, gx, gy, gw, gh)
    row = (zc, yc)
    xs = np.arange(xc - r0, xc + r0)
    p0 = ndi.gaussian_filter1d(ct[row][xs], 1.0); p1 = ndi.gaussian_filter1d(att[row][xs], 1.0)
    inside = les[row][xs]
    lo, hi = min(p0.min(), p1.min()) - 8, max(p0.max(), p1.max()) + 8
    X = gx + np.linspace(0, gw, len(xs)); Y0 = gy + (p0 - lo) / (hi - lo) * gh; Y1 = gy + (p1 - lo) / (hi - lo) * gh
    xi = np.flatnonzero(inside)
    ax.add_patch(Rectangle((X[xi[0]], gy), X[xi[-1]] - X[xi[0]], gh, fc='#fbd9c9', ec='none', zorder=3))
    ax.plot(X, Y0, color='#9a9893', lw=0.7, zorder=5)
    ax.plot(X, Y1, color=ORANGE, lw=0.9, zorder=5)
    ax.text((X[xi[0]] + X[xi[-1]]) / 2, gy - 0.8, T['inside'], fontsize=5.0, ha='center', va='top', color=LESION)
    lgx = gx + gw + 0.8
    ax.plot([lgx, lgx + 2.2], [gy + gh - 1.0] * 2, color=ORANGE, lw=0.9)
    ax.text(lgx + 2.8, gy + gh - 1.0, T['after'], fontsize=4.8, va='center', ha='left', color=INK2)
    ax.plot([lgx, lgx + 2.2], [gy + gh - 4.0] * 2, color='#9a9893', lw=0.7)
    ax.text(lgx + 2.8, gy + gh - 4.0, T['before'], fontsize=4.8, va='center', ha='left', color=INK2)
    arrow(ax, (cx + cw + 4.6, cy_ + ch / 2), (gx - 1.4, gy + ch / 2), color=ORANGE, lw=0.7, ms=5)
    ax.text((cx + cw + 4.6 + gx - 1.4) / 2, cy_ + ch / 2 + 1.0, '$\\alpha_k$', fontsize=5.8, ha='center', va='bottom',
            color=INK)
    mtext(ax, px + pw / 2, py + 3.4,
          '$\\alpha_k=\\min(1,\\,t_k/\\mathrm{CNR}_k),\\quad \\tilde{x}=x-(1-\\alpha_k)(\\mu_k-\\nu_k)\\,w_k$', size=5.8)
    ycube = yin + S / 2
    arrow(ax, (xin + S + DEP[0] + 0.8, ycube), (px - 0.3, ycube), lw=0.9)

    # ===== attenuated cube
    xa, ya = 101, yin
    cube(ax, avol, xa, ya, S, DEP, z0, 18, contour=lvol)
    mtext(ax, xa + S / 2 + 2, ya - 3.4, T['xt'], size=6.4)
    mtext(ax, xa + S / 2 + 2, ya - 7.8, f'CNR {cnr0:.2f} → {cnr1:.2f}', size=5.8, color=ORANGE)
    arrow(ax, (px + pw + 0.3, ycube), (xa - 0.6, ycube), lw=0.9)

    # ===== networks
    nx0, ys, yt = 131.5, ycube, 41
    ax.add_patch(FancyBboxPatch((127.5, 24), 47, 89, boxstyle='round,pad=0,rounding_size=2', fc='#f6f9fd', ec=BLUE,
                                lw=0.8, zorder=0))
    mtext(ax, 129.5, 24 + 89 - 2.6, T['nets'], size=6.6, ha='left', color=BLUE, z=6)
    st = unet(ax, nx0, ys, BLUE_L, BLUE_M, BLUE, GREEN, step=4.6)
    te = unet(ax, nx0, yt, GRAY_L, GRAY_M, '#8c8a85', GREEN, step=4.6)
    mtext(ax, 151, ys + 11.4, T['student'], size=5.6, color=BLUE)
    mtext(ax, 151, yt + 11.4, T['teacher'], size=5.6, color='#6f6d68')
    for net in (st, te):
        mtext(ax, net['enc_mid'], net['bottom'] - 1.4, T['enc'], size=5.0, color=INK2)
        mtext(ax, net['dec_mid'], net['bottom'] - 1.4, T['dec'], size=5.0, color=INK2)
    arrow(ax, (xa + S + DEP[0] + 0.8, ycube), st['inp'], lw=0.9)
    # x -> teacher: around the bottom
    yr = 20.5
    ax.plot([xin + S / 2, xin + S / 2, 122.5, 122.5], [yin - 7.2, yr, yr, yt], color=INK2, lw=0.9, zorder=2)
    arrow(ax, (122.5, yt), te['inp'], lw=0.9)
    # shared weights
    arrow(ax, (138, st['bottom'] - 3.6), (138, yt + 14.2), color=BLUE, lw=0.8, ls=(0, (3, 1.6)), style='<|-|>', ms=5)
    mtext(ax, 139.6, (st['bottom'] - 3.6 + yt + 14.2) / 2, T['shared'], size=5.6, ha='left', color=BLUE)

    # ===== output, label, Dice-CE
    ox, oy, osz = 180, 95, 8.8
    thumb(ax, labimg(pred[zc][sl2]), ox, oy, osz, osz, cmap=labcmap, vmin=0, vmax=2)
    thumb(ax, labimg(gt[zc][sl2]), ox + osz + 1.4, oy, osz, osz, cmap=labcmap, vmin=0, vmax=2)
    mtext(ax, ox + osz / 2, oy + osz + 2.2, '$\\hat{y}$', size=6.4)
    mtext(ax, ox + osz * 1.5 + 1.4, oy + osz + 2.2, '$y$', size=6.4)
    mtext(ax, ox + osz + 0.7, oy - 2.8, '$\\mathcal{L}_{\\mathrm{DiceCE}}(\\hat{y},\\,y)$', size=6.0)
    arrow(ax, (st['right'], st['out'][1] + 3), (ox - 0.6, oy + osz / 2), lw=0.9, conn='arc3,rad=-0.2')

    # ===== CFC panel
    cxp, cyp, cwp, chp = 180, 30, 18.5, 56
    panel(ax, cxp, cyp, cwp, chp, '(c) CFC', GREEN, GREEN_BG)
    tsz = 14.5
    tx3, ty3 = cxp + 2, cyp + 33
    thumb(ax, att[zc][sl3], tx3, ty3, tsz, tsz)
    om = omega[zc][sl3]
    ax.imshow(np.ma.masked_where(~om, np.ones_like(om, float)), cmap=plt.matplotlib.colors.ListedColormap([GREEN]),
              alpha=0.28, extent=(tx3, tx3 + tsz, ty3, ty3 + tsz), origin='upper', zorder=3.5)
    overlay_contour(ax, om, tx3, ty3, tsz, tsz, GREEN, lw=0.7)
    overlay_contour(ax, les[zc][sl3], tx3, ty3, tsz, tsz, LESION, lw=0.5)
    mtext(ax, cxp + cwp / 2, ty3 - 2.6, '$\\Omega$', size=6.4, color=GREEN)
    mtext(ax, cxp + cwp / 2, cyp + 23.0, '$\\mathcal{L}_{\\mathrm{CFC}}=1-$', size=5.9)
    mtext(ax, cxp + cwp / 2, cyp + 18.8, '$\\cos(F(\\tilde{x}),\\,$', size=5.9)
    mtext(ax, cxp + cwp / 2, cyp + 14.6, '$\\mathrm{sg}[F(x)])$', size=5.9)
    mtext(ax, cxp + cwp / 2, cyp + 5.0, '最细两层\n解码器特征' if zh else 'two finest\ndecoder stages',
          size=4.9, color=INK2)
    # taps: one arrow from each framed decoder pair into the CFC panel
    arrow(ax, st['tap'], (cxp - 0.3, cyp + 44), color=GREEN, lw=0.8, ms=5, conn='arc3,rad=-0.25', z=8)
    arrow(ax, te['tap'], (cxp - 0.3, cyp + 14), color=GREEN, lw=0.8, ms=5, conn='arc3,rad=0.25', z=8)
    # stop-gradient marker on the teacher tap
    sgx, sgy = 178.0, te['tap'][1] - 0.6
    sgx, sgy = (te['tap'][0] + cxp) / 2, (te['tap'][1] + cyp + 14) / 2 - 1.0
    ax.add_patch(Circle((sgx, sgy), 1.35, fc='white', ec=INK, lw=0.6, zorder=10))
    ax.plot([sgx - 0.95, sgx + 0.95], [sgy - 0.95, sgy + 0.95], color=INK, lw=0.6, zorder=11)

    # ===== total loss
    mtext(ax, 163, 19.6, (T['total'] + '：' if zh else T['total'] + ':  ') +
          '$\\mathcal{L}=\\mathcal{L}_{\\mathrm{DiceCE}}+\\lambda\\,\\psi(e)\\,\\mathcal{L}_{\\mathrm{CFC}}$', size=6.4)

    # ===== legend (two rows below the DCA panel)
    lx, ly = 35, 40
    def leg_arrow(x, y, ls, col, lab):
        arrow(ax, (x, y), (x + 6, y), color=col, lw=0.8, ls=ls, ms=4.6)
        mtext(ax, x + 7.6, y, lab, size=5.4, ha='left')
    leg_arrow(lx, ly, '-', INK2, T['lg_fwd'])
    leg_arrow(lx + 27, ly, (0, (3, 1.6)), BLUE, T['lg_share'])
    leg_arrow(lx, ly - 5.4, '-', GREEN, T['lg_feat'])
    sx_, sy_ = lx + 30, ly - 5.4
    ax.add_patch(Circle((sx_, sy_), 1.3, fc='white', ec=INK, lw=0.6, zorder=8))
    ax.plot([sx_ - 0.9, sx_ + 0.9], [sy_ - 0.9, sy_ + 0.9], color=INK, lw=0.6, zorder=9)
    mtext(ax, lx + 34.6, sy_, T['lg_sg'], size=5.4, ha='left')
    ax.add_patch(FancyBboxPatch((lx, ly - 12.4), 6, 3.0, boxstyle='round,pad=0,rounding_size=0.8', fc='none',
                                ec=INK2, lw=0.6, ls=(0, (3, 1.6))))
    mtext(ax, lx + 7.6, ly - 10.9, T['lg_train'], size=5.4, ha='left')
    ax.add_patch(Rectangle((lx + 27, ly - 12.4), 6, 3.0, fc='none', ec=GREEN, lw=0.8))
    mtext(ax, lx + 34.6, ly - 10.9, 'CFC 取特征的解码层' if zh else 'decoder stages used by CFC', size=5.4, ha='left')

    # ===== inference strip
    ix, iy = 22, 3.4
    thumb(ax, ct[zc][sl2], ix, iy, 8, 8)
    mtext(ax, ix + 4, iy - 0.0, '', size=1)
    arrow(ax, (ix + 9, iy + 4), (ix + 13.2, iy + 4), lw=0.8, ms=5)
    for k_, (dx_, h_, fc_) in enumerate(((14, 6.0, BLUE_L), (17.2, 4.0, BLUE_L), (20.4, 6.0, BLUE_M))):
        cuboid(ax, ix + dx_, iy + 4 - h_ / 2, 1.5, h_, (0.9, 0.7), fc_, BLUE)
    arrow(ax, (ix + 23.8, iy + 4), (ix + 28, iy + 4), lw=0.8, ms=5)
    thumb(ax, labimg(pred[zc][sl2]), ix + 29, iy, 8, 8, cmap=labcmap, vmin=0, vmax=2)
    mtext(ax, ix + 42, iy + 4, T['infer'], size=6.0, ha='left')
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig2_method_{lang}.{ext}', dpi=600, pad_inches=0.0)
    plt.close(fig)
    print('wrote', f'fig2_method_{lang}', f'(CNR {cnr0:.2f} -> {cnr1:.2f}, alpha {alpha:.2f})')


if __name__ == '__main__':
    for lang in sys.argv[1:] or ('zh', 'en'):
        draw(lang)
