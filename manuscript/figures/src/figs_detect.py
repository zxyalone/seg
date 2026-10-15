#!/usr/bin/env python3
"""Figures for the detectability-guided training manuscript (Chinese labels).

    python manuscript/figures/src/figs_detect.py            # from the repo root

Writes manuscript/figures/fig{1..4}_*_zh.{pdf,png}. Data come only from
archived result files (validation diagnostics and the one-time test run).
Colors: reference data-viz palette slots 1-3 (blue, orange, aqua; validated
all-pairs) for the augmentation groups, neutral grays plus dashed lines and
distinct markers for the two baselines.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'manuscript/figures'
MM = 1 / 25.4

plt.rcParams.update({
    'font.family': ['Arial', 'STHeiti'], 'font.size': 7, 'mathtext.fontset': 'stixsans',
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.labelcolor': '#0b0b0b', 'legend.frameon': False,
})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STYLE = {  # group: (label, color, linestyle, marker, linewidth)
    'A0': ('nnU-Net', '#9a9893', (0, (3, 1.5)), 's', 1.0),
    'A1': ('nnU-Net + Gate', '#52514e', (0, (3, 1.5)), 'D', 1.0),
    'A11': ('nnU-Net + UCA', '#1baf7a', '-', '^', 1.0),
    'A13a': ('nnU-Net + DCA', '#eb6834', '-', 'v', 1.0),
    'A13b': ('nnU-Net + DCA + CFC（本文）', '#2a78d6', '-', 'o', 1.6),
}
RULES = ['argmax', '0.5', '0.4', '0.3', '0.2', '0.1', '0.05']


CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def mtext(ax, x, y, s, size=6.5, ha='center', va='center', color=INK, transform=None, z=6):
    """Text that may mix CJK and mathtext: mathtext cannot fall back to a CJK
    font, so each line is laid out as side-by-side text and math segments."""
    if '$' not in s or not CJK.search(s):
        ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, zorder=z,
                transform=transform or ax.transData, linespacing=1.3)
        return
    props = dict(fontsize=size, color=color)
    lines = []
    for line in s.split('\n'):
        segs = [g for g in re.split(r"(\$[^$]*\$)", line) if g]
        lines.append(HPacker(children=[TextArea(g, textprops=props) for g in segs], sep=0, pad=0, align='baseline'))
    pack = VPacker(children=lines, sep=2, pad=0, align={'left': 'left', 'center': 'center', 'right': 'right'}[ha])
    ab = AnnotationBbox(pack, (x, y), xycoords=transform or 'data', frameon=False, pad=0, zorder=z,
                        box_alignment=({'left': 0, 'center': 0.5, 'right': 1}[ha], {'bottom': 0, 'center': 0.5, 'top': 1}[va]))
    ax.add_artist(ab)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', name)


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=INK)


def sweep_means(csv, groups):
    S = pd.read_csv(csv)
    S = S[S.rule.str.endswith('_pp') & S.group.isin(groups)].copy()
    S['rule'] = S.rule.str[:-3]
    S['recall_low'] *= 100
    return S.groupby(['group', 'rule'])[['fp_per_scan', 'recall_low']].mean()


# ---------------------------------------------------------------------------
# Figure 1: diagnosis (validation, A0)
# ---------------------------------------------------------------------------
def fig_diagnosis():
    L = pd.read_csv(ROOT / 'results/missed_probability/lesions.csv')
    L = L[(L.group == 'A0') & (L.voxels >= 50)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.38))

    rng = np.random.default_rng(0)
    strata = [('low', '低 CNR'), ('mid', '中 CNR'), ('high', '高 CNR')]
    for i, (s, name) in enumerate(strata):
        for j, (det, col, lab) in enumerate(((True, '#9a9893', '检出'), (False, '#eb6834', '漏检'))):
            v = np.clip(L[(L.cnr_stratum == s) & (L['det@argmax'] == det)].p_q90.values, 1e-4, 1)
            x = i + (j - 0.5) * 0.36 + rng.uniform(-0.11, 0.11, len(v))
            a.scatter(x, v, s=7, color=col, edgecolor='white', linewidth=0.3, zorder=3, label=lab if i == 0 else None)
            if len(v):
                m = np.median(v)
                a.plot([i + (j - 0.5) * 0.36 - 0.15, i + (j - 0.5) * 0.36 + 0.15], [m, m], color=INK, lw=1.0, zorder=4)
    a.set_yscale('log'); a.set_ylim(7e-5, 1.6)
    a.set_xticks(range(3)); a.set_xticklabels([n for _, n in strata])
    a.set_ylabel('病灶内肿瘤概率 90% 分位数')
    a.axhline(0.5, color=GRID, lw=0.6, zorder=1)
    a.text(2.62, 0.5, '0.5', color=INK2, fontsize=6, va='center')
    low_missed = L[(L.cnr_stratum == 'low') & ~L['det@argmax']].p_q90
    a.annotate(f'漏检低 CNR 病灶\n中位数 {low_missed.median():.3f}', xy=(0.18, max(low_missed.median(), 1e-4)),
               xytext=(0.55, 6e-3), fontsize=6, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    a.legend(loc='upper left', bbox_to_anchor=(0.0, 1.13), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
    panel_label(a, 'a')

    M = sweep_means(ROOT / 'results/missed_probability/threshold_sweep_per_run.csv', ['A0'])
    pts = M.loc['A0'].loc[RULES]
    b.plot(pts.fp_per_scan, pts.recall_low, color='#9a9893', lw=1.0, marker='s', ms=3.5, mec='white', mew=0.4, zorder=3)
    for r, row in pts.iterrows():
        off = {'argmax': (-4, 6), '0.1': (0, -9), '0.05': (-6, 6)}
        if r in off:
            b.annotate('argmax' if r == 'argmax' else f'p > {r}', (row.fp_per_scan, row.recall_low), xytext=off[r],
                       textcoords='offset points', fontsize=5.8, color=INK2, ha='center')
    b.set_xlabel('每例假阳性数（4 mm 后处理）'); b.set_ylabel('低 CNR 病灶召回（%）')
    b.set_ylim(55, 90); b.grid(axis='y', color=GRID, lw=0.5); b.set_axisbelow(True)
    b.axhline(85, color='#1baf7a', lw=0.8, ls=(0, (2, 1.5)))
    b.text(b.get_xlim()[0] + 0.02, 85.6, '中、高 CNR 层召回约 83–87%', fontsize=6, color='#127a55')
    panel_label(b, 'b')
    save(fig, 'fig1_diagnosis_zh')


# ---------------------------------------------------------------------------
# Figure 2: method
# ---------------------------------------------------------------------------
def dca_2d(ct, gt, target):
    m = gt == 2
    lab, n = ndi.label(m)
    k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    m = lab == k
    d6 = ndi.maximum_filter(m, size=13); d1 = ndi.maximum_filter(m, size=3)
    ring = d6 & ~d1 & (gt == 1)
    mu, nu, s = ct[m].mean(), ct[ring].mean(), ct[ring].std()
    cnr = abs(mu - nu) / s
    alpha = min(1.0, target / cnr)
    w = np.maximum(ndi.gaussian_filter(m.astype(float), 1.0), m) * (gt > 0)
    out = ct - (1 - alpha) * (mu - nu) * w
    cnr_after = abs(out[m].mean() - out[ring].mean()) / out[ring].std()
    return out, m, ring, cnr, cnr_after, alpha


def fig_method():
    d = np.load(ROOT / 'manuscript/figures/src/th0.npz')
    ct, gt = d['ct'], d['gt']
    sl = (slice(200, 385), slice(80, 275))
    out, m, ring, cnr0, cnr1, alpha = dca_2d(ct, gt, target=0.6)
    fig = plt.figure(figsize=(190 * MM, 128 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], height_ratios=[1, 1.15], wspace=0.34, hspace=0.32)
    kw = dict(cmap='gray', vmin=0, vmax=255, interpolation='lanczos')
    cy = int(np.round(ndi.center_of_mass(m)[0]))

    a0 = fig.add_subplot(gs[0, 0]); a0.imshow(ct[sl], **kw)
    a0.contour(m[sl], levels=[0.5], colors='#eb6834', linewidths=0.7)
    a0.contour(ring[sl], levels=[0.5], colors='#1baf7a', linewidths=0.5, linestyles='dashed')
    a0.set_title(f'原始病灶   CNR = {cnr0:.2f}', fontsize=7, color=INK, pad=3)
    a1 = fig.add_subplot(gs[0, 1]); a1.imshow(out[sl], **kw)
    a1.contour(m[sl], levels=[0.5], colors='#eb6834', linewidths=0.7)
    a1.set_title(f'DCA 后   CNR = {cnr1:.2f}（α = {alpha:.2f}）', fontsize=7, color=INK, pad=3)
    for ax in (a0, a1):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.axhline(cy - sl[0].start, color='#2a78d6', lw=0.6, ls=(0, (2, 1.5)))
    panel_label(a0, 'a', x=-0.06, y=1.06)
    mtext(a0, 0.0, -0.06, '橙线：病灶 $M_k$　绿虚线：肝实质环带 $R_k$（病灶外 2–6 体素）　蓝虚线：b 的剖面', size=6,
          ha='left', va='top', color=INK2, transform=a0.transAxes)

    b = fig.add_subplot(gs[0, 2])
    xs = np.arange(sl[1].start, sl[1].stop)
    prof0 = ndi.gaussian_filter1d(ct[cy, xs].astype(float), 1.5)
    prof1 = ndi.gaussian_filter1d(out[cy, xs].astype(float), 1.5)
    inside = m[cy, xs]
    b.fill_between(np.arange(len(xs)), 0, 255, where=inside, color='#fbe1d6', lw=0, step='mid')
    b.plot(prof0, color='#9a9893', lw=0.9, label='原始')
    b.plot(prof1, color='#2a78d6', lw=1.1, label='DCA 后')
    b.set_ylim(0, 255); b.set_xlim(0, len(xs) - 1)
    b.set_xlabel('沿剖面的位置（体素）'); b.set_ylabel('强度（显示窗）')
    xm = np.flatnonzero(inside).mean()
    b.text(xm, 238, '病灶内强度整体平移，起伏（纹理）不变', ha='center', fontsize=6.2, color='#b6481b')
    b.legend(loc='lower right', fontsize=6.2, handlelength=1.6)
    panel_label(b, 'b', x=-0.14, y=1.06)

    c = fig.add_subplot(gs[1, :]); c.axis('off'); c.set_xlim(0, 100); c.set_ylim(0, 50)
    panel_label(c, 'c', x=0.0, y=1.0)

    def box(x, y, w, h, s, fc='#ffffff', ec=INK2, fs=6.5):
        c.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=0.6))
        mtext(c, x + w / 2, y + h / 2, s, size=fs)

    def arrow(p, q, color=INK2):
        c.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.7, color=color, shrinkA=0, shrinkB=0))

    mtext(c, 50, 47, '训练：$\\mathcal{L}=\\mathcal{L}_{\\mathrm{DiceCE}}(f_\\theta(\\tilde{x}),\\,y)+\\lambda\\,\\psi(e)\\,\\mathcal{L}_{\\mathrm{CFC}}$', size=7)
    y0, h = 31, 9
    box(1, y0, 13, h, '训练图块 $x$')
    box(19, y0, 22, h, 'DCA\n按目标 CNR 衰减病灶', fc='#e8f1fb', ec='#2a78d6')
    box(46, y0, 13, h, '衰减图块 $\\tilde{x}$')
    box(64, y0, 15, h, '学生 $f_\\theta$', fc='#f4f3ef')
    box(84, y0, 15, h, '$\\mathcal{L}_{\\mathrm{DiceCE}}$')
    for x1, x2 in ((14, 19), (41, 46), (59, 64), (79, 84)):
        arrow((x1, y0 + h / 2), (x2, y0 + h / 2))
    mtext(c, 30, y0 - 3.2, '$t \\sim F_{\\mathrm{low}}$（训练集低 CNR 层分布），$\\alpha=\\min(1,\\,t/\\mathrm{CNR})$', size=6, color=INK2)

    box(62, 8.5, 21, 9, '教师 $f_\\theta$\n（同一网络，不回传梯度）', fc='#f4f3ef', fs=6)
    c.plot([7.5, 7.5], [y0, 11], color=INK2, lw=0.7)
    arrow((7.5, 11), (62, 11))
    box(31, 14, 27, 9.5, '$\\mathcal{L}_{\\mathrm{CFC}}$：被衰减病灶邻域 $\\Omega$ 内\n$1-\\cos(F(\\tilde{x}),\\,\\mathrm{sg}[F(x)])$', fc='#fdf0ea', ec='#eb6834', fs=6.2)
    arrow((68, y0), (58.5, 21.5), color='#eb6834')
    arrow((62, 15.5), (58.5, 15.5), color='#eb6834')
    mtext(c, 66, 25.5, '最细两层解码器特征', size=5.8, color=INK2, ha='left')
    box(1, 0.5, 98, 6, '推理：只运行 $f_\\theta$，与基线完全相同（无新增参数，无额外计算）', ec=GRID, fs=6.5)
    save(fig, 'fig2_method_zh')
    print(f'  DCA example: CNR {cnr0:.2f} -> {cnr1:.2f} (alpha {alpha:.2f})')


# ---------------------------------------------------------------------------
# Figure 3: test FP/scan vs low-CNR recall
# ---------------------------------------------------------------------------
def fig_curves():
    M = sweep_means(ROOT / 'results/test_protocol/curves/threshold_sweep_per_run.csv', list(STYLE))
    fig, ax = plt.subplots(figsize=(110 * MM, 78 * MM))
    budget = M.loc[('A13b', 'argmax'), 'fp_per_scan']
    ax.axvline(budget, color=GRID, lw=0.8, zorder=1)
    ax.text(budget + 0.008, 56.3, f'DCA + CFC 默认输出\n每例假阳性 {budget:.2f}', fontsize=5.8, color=INK2, va='bottom')
    for g, (lab, col, ls, mk, lw) in STYLE.items():
        pts = M.loc[g].loc[RULES]
        ax.plot(pts.fp_per_scan, pts.recall_low, color=col, ls=ls, lw=lw, zorder=3)
        ax.scatter(pts.fp_per_scan, pts.recall_low, s=10, marker=mk, color=col, edgecolor='white', linewidth=0.4, zorder=4)
        am = M.loc[(g, 'argmax')]
        ax.scatter([am.fp_per_scan], [am.recall_low], s=26, marker=mk, color=col, edgecolor=INK, linewidth=0.5, zorder=5)
        lx, ly, ha = {'A0': (2.47, 64.4, 'left'), 'A1': (2.45, 66.4, 'left'), 'A11': (2.73, 72.0, 'left'),
                      'A13a': (2.37, 74.6, 'left'), 'A13b': (1.84, 74.5, 'left')}[g]
        ax.text(lx, ly, lab, fontsize=6.2, color=col if g != 'A0' else '#6e6d68', va='center', ha=ha,
                fontweight='bold' if g == 'A13b' else 'normal')
    ax.set_xlim(1.82, 2.95); ax.set_ylim(56, 77)
    ax.set_xlabel('每例假阳性数（4 mm 后处理，3 个种子均值）')
    ax.set_ylabel('低 CNR 病灶召回（%）')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(0.01, 0.99, '大标记：默认 argmax 输出；沿线依次为阈值 0.5 → 0.05', transform=ax.transAxes,
            fontsize=5.8, color=INK2, va='top')
    save(fig, 'fig3_test_fp_recall_zh')


# ---------------------------------------------------------------------------
# Figure 4: stratified change on the test set
# ---------------------------------------------------------------------------
def fig_strata():
    T = pd.read_csv(ROOT / 'results/test_protocol/stats/strata_recall.csv')
    T = T[T.cnr_stratum.isin(['low', 'mid', 'high'])]
    rows = [(c, s) for c in ('low', 'mid', 'high') for s in ('<10', '10-20', '>20')]
    cname = {'low': '低', 'mid': '中', 'high': '高'}
    sname = {'<10': '< 10 mm', '10-20': '10–20 mm', '>20': '> 20 mm'}
    fig, ax = plt.subplots(figsize=(110 * MM, 78 * MM))
    for i, (c, s) in enumerate(rows):
        y = len(rows) - 1 - i + (0 if c == 'low' else -0.4 if c == 'mid' else -0.8)
        r1 = T[(T.group == 'A1') & (T.cnr_stratum == c) & (T.size_stratum == s)].iloc[0]
        rb = T[(T.group == 'A13b') & (T.cnr_stratum == c) & (T.size_stratum == s)].iloc[0]
        delta = rb.recall - r1.recall
        ax.plot([r1.recall, rb.recall], [y, y], color='#c3c2b7', lw=1.4, zorder=2, solid_capstyle='round')
        ax.scatter([r1.recall], [y], s=22, marker='D', color='#52514e', edgecolor='white', lw=0.4, zorder=3)
        ax.scatter([rb.recall], [y], s=26, marker='o', color='#2a78d6', edgecolor='white', lw=0.4, zorder=4)
        ax.text(-0.02, y, f'{cname[c]} CNR · {sname[s]}（{int(r1.n)}）', ha='right', va='center', fontsize=6.2, color=INK,
                transform=ax.get_yaxis_transform())
        if abs(delta) >= 0.05:
            ax.text(103, y, f'{delta:+.1f}', ha='left', va='center', fontsize=6.2,
                    color='#1f5fae' if delta > 0 else '#b6481b', fontweight='bold' if c == 'low' else 'normal')
        else:
            ax.text(103, y, '0.0', ha='left', va='center', fontsize=6.2, color=INK2)
    ax.set_xlim(30, 101); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('测试集病灶召回（%，3 个种子均值）')
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(103, len(rows) - 0.25, '差值', fontsize=6.2, color=INK2, ha='left')
    ax.scatter([], [], s=22, marker='D', color='#52514e', label='nnU-Net + Gate')
    ax.scatter([], [], s=26, marker='o', color='#2a78d6', label='nnU-Net + DCA + CFC（本文）')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.3, handletextpad=0.2)
    save(fig, 'fig4_test_strata_zh')


if __name__ == '__main__':
    fig_diagnosis()
    # Fig. 2 (method overview) is drawn by fig_method_overview.py
    fig_curves()
    fig_strata()
