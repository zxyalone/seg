#!/usr/bin/env python3
"""Additional figures (selection rules registered in plan §11, 2026-10-14).

    python manuscript/figures/src/figs_extra.py [zh] [en]

fig3_samples      training samples: DCA at three target CNRs (Fig. 2 lesion)
                  and one Pixel2Cancer variant
fig6_cnr_curve    lesion detection rate against CNR (LiTS test, MSD Task08)
fig7_msd08        MSD Task08 examples (three gains, one false positive)
fig8_forest       differences in low-CNR recall and FP/scan with 95% CI
fig10_multi       six random low-CNR test lesions, five methods

Crops come from figs_extra.npz (extract_figs_extra.py) and method_case.npz;
numbers from the archived result tables only.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figs_detect import INK, INK2, GRID, MM, OUT, ROOT, STYLE  # noqa: E402  (also sets rcParams)
from fig_qualitative import GT_C, PRED_C, WL, WW, up  # noqa: E402


def contour(ax, m, color, lw, f=4, ls='-'):
    """Smooth outline of a binary mask (any aspect ratio)."""
    if m.any():
        h, w = m.shape
        ax.contour(np.linspace(-.5, w - .5, w * f), np.linspace(-.5, h - .5, h * f), up(m, f),
                   levels=[.5], colors=color, linewidths=lw, linestyles=ls)

R = ROOT / 'results'
SEED_BOOT, N_BOOT = 20261012, 10_000
A14 = ('Pixel2Cancer', '#4a3aa7', '-', 'P', 1.0)          # categorical slot 7; validated with slots 1-3
TXT = {
    'zh': dict(base='nnU-Net', gated='nnU-Net + Gate', ours='nnU-Net + DCA + CFC', p2c='nnU-Net + Pixel2Cancer', cc='nnU-Net + CC-DiceCE', ct='CT', ref='金标准',
               det='检出', mis='漏检', fp='假阳性', lits='LiTS 测试集', msd='MSD Task08', orig='原始', syn='合成',
               label='合成肿瘤标签', cnr='CNR', diam='直径', gain='增益', fpk='DCA + CFC 独有的假阳性', contrast='对比度',
               xcnr='病灶 CNR', ydet='病灶检出率（%）', lowtier='低 CNR 层', n='病灶数',
               d_recall='DCA + CFC 与对照的低 CNR 召回差值（个百分点）', d_fp='DCA + CFC 与对照的每例假阳性差值', conf='确认性检验', desc='描述性',
               vs='vs ', task08ref='Task08 肿瘤', dca='DCA：同一真实病灶衰减到不同目标 CNR',
               p2ct='Pixel2Cancer：在肝实质中合成新肿瘤'),
    'en': dict(base='nnU-Net', gated='nnU-Net + Gate', ours='nnU-Net + DCA + CFC', p2c='nnU-Net + Pixel2Cancer', cc='nnU-Net + CC-DiceCE', ct='CT', ref='Reference',
               det='detected', mis='missed', fp='false positive', lits='LiTS test', msd='MSD Task08', orig='original', syn='synthetic',
               label='synthetic label', cnr='CNR', diam='diameter', gain='Gain', fpk='False positive of DCA + CFC only', contrast='contrast',
               xcnr='Lesion CNR', ydet='Lesion detection rate (%)', lowtier='low-CNR tier', n='lesions',
               d_recall='DCA + CFC minus comparator: low-CNR recall (pp)', d_fp='DCA + CFC minus comparator: FP per scan', conf='confirmatory', desc='descriptive',
               vs='vs ', task08ref='Task08 tumour', dca='DCA: one real lesion attenuated to target CNRs',
               p2ct='Pixel2Cancer: new tumours synthesised in the liver'),
}
BINS = [0, 0.5, 1.0, 1.28, 1.92, 3.0, np.inf]


def window(hu):
    return np.clip((hu - (WL - WW / 2)) / WW, 0, 1)


def tag(ax, text, ok):
    ax.text(.96, .04, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.6,
            color='white' if ok else '#ffd2c2', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', fc=(0, 0, 0, .55), ec='none'))


def arrow(ax, tgt):
    if not tgt.any() or tgt.mean() > 0.2:      # no arrow when the lesion fills the crop
        return
    ty, tx = np.argwhere(tgt).mean(0); r = np.sqrt(tgt.sum() / np.pi) + 3
    ax.annotate('', xy=(tx - r * .62, ty + r * .62), xytext=(tx - r * .62 - 12, ty + r * .62 + 12),
                arrowprops=dict(arrowstyle='-|>', color='white', lw=.9, mutation_scale=6, shrinkA=0, shrinkB=0))


def grid_axes(fig, n_rows, n_cols, pw, gap, left, top, H, W, i, j):
    y0 = 1 - (top + i * (pw + gap) + pw) / H
    ax = fig.add_axes([(left + j * (pw + gap)) / W, y0, pw / W, pw / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax, y0


def save_img(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, pad_inches=0.0)
    plt.close(fig); print('wrote', name)


def save_chart(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig); print('wrote', name)


# ---------------------------------------------------------------- Fig. 3
def fig_samples(lang):
    T = TXT[lang]
    m = np.load(HERE / 'method_case.npz'); D = np.load(HERE / 'figs_extra.npz')
    ct, att, les = m['ct'], m['att'], m['lesion']
    cnr0, a0 = float(m['cnr0']), float(m['alpha'])
    z = int(np.argmax(les.sum((1, 2))))
    delta = (ct - att) / (1 - a0)                                  # (mu - nu) * w, Eq. (4)
    panels = [(ct[z], f"{T['orig']}  {T['cnr']} {cnr0:.2f}")]
    for t in (1.0, 0.6, 0.3):
        al = min(1.0, t / cnr0)
        panels.append(((ct - (1 - al) * delta)[z], f"{T['cnr']} {t:.1f}  (α = {al:.2f})"))
    pw, gap, left, top = 36 * MM, 1.5 * MM, 2 * MM, 11 * MM
    W = left + 4 * pw + 3 * gap; H = top + 2 * pw + 13 * MM
    fig = plt.figure(figsize=(W, H))
    for j, (im, title) in enumerate(panels):
        ax, y0 = grid_axes(fig, 2, 4, pw, gap, left, top, H, W, 0, j)
        ax.imshow(window(im), cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
        contour(ax, les[z], GT_C, .7, ls='--')
        ax.set_title(title, fontsize=6.5, color=INK, pad=2)
    fig.text(left / W, 1 - 1.0 * MM / H, '(a) ' + T['dca'], fontsize=7.5, fontweight='bold', va='top', color=INK)
    o, s, n = D['p2c_orig'], D['p2c_syn'], D['p2c_new']
    top2 = top + pw + 13 * MM
    for j, (im, title, lab) in enumerate([(o, T['orig'], None), (s, T['syn'], None), (s, T['label'], n)]):
        y0 = 1 - (top2 + pw) / H
        ax = fig.add_axes([(left + j * (pw + gap)) / W, y0, pw / W, pw / H])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.imshow(window(im), cmap='gray', vmin=0, vmax=1, interpolation='lanczos', aspect='auto')
        if lab is not None:
            contour(ax, lab, A14[1], .8)
        ax.set_title(title, fontsize=6.5, color=INK, pad=2)
    fig.text(left / W, 1 - (top + pw + 3.0 * MM) / H, '(b) ' + T['p2ct'], fontsize=7.5, fontweight='bold', va='top', color=INK)
    save_img(fig, f'fig3_samples_{lang}')


# ---------------------------------------------------------------- Fig. 6
def detection_by_bin(L, groups):
    L = L[np.isfinite(L.cnr)].copy()
    L['bin'] = pd.cut(L.cnr, BINS, right=False, labels=False)
    L['group'] = L.run.str.rsplit('_s', n=1).str[0]
    rows = []
    for g in groups:
        per_seed = L[L.group == g].groupby(['run', 'bin']).detected.mean().mul(100).unstack('run')
        rows.append(pd.DataFrame(dict(group=g, bin=per_seed.index, mean=per_seed.mean(1), lo=per_seed.min(1), hi=per_seed.max(1))))
    n = L[L.run == L.run.iloc[0]].groupby('bin').size()
    return pd.concat(rows), n


def fig_cnr_curve(lang):
    T = TXT[lang]
    test = pd.concat([pd.read_csv(R / 'test_protocol/tables/lesions_pp4mm.csv'),
                      pd.read_csv(R / 'a14_pixel2cancer/test_tables/lesions_pp4mm.csv')])
    msd = pd.read_csv(R / 'external_msd08/tables/lesions_pp4mm.csv')
    names = {'A0': T['base'], 'A1': T['gated'], 'A13b': T['ours'], 'A14': T['p2c']}
    style = {g: STYLE[g] for g in ('A0', 'A1', 'A13b')}; style['A14'] = A14
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), sharey=True)
    labels = ['<0.5', '0.5–1.0', '1.0–1.28', '1.28–1.92', '1.92–3.0', '≥3.0']
    for ax, (L, groups, title) in zip(axes, ((test, ('A0', 'A1', 'A14', 'A13b'), T['lits']),
                                            (msd, ('A0', 'A1', 'A13b'), T['msd']))):
        D, n = detection_by_bin(L, groups)
        ax.axvspan(-0.5, 2.5, color='#eef3fb', zorder=0)
        ax.text(1.0, 3, T['lowtier'], ha='center', va='bottom', fontsize=6, color=INK2)
        for g in groups:
            d = D[D.group == g]; _, c, ls, mk, lw = style[g]
            ax.plot(d.bin, d['mean'], color=c, ls=ls, marker=mk, ms=4, lw=lw, mec='white', mew=.6, zorder=3, label=names[g])
            ax.vlines(d.bin, d.lo, d.hi, color=c, lw=.8, alpha=.7, zorder=2)
            last = d.iloc[-1]
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
        ax.set_xlim(-0.5, len(labels) - 0.5); ax.set_ylim(0, 105)
        ax.set_xlabel(T['xcnr']); ax.set_title(title, fontsize=7.5, color=INK, pad=4)
        ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)
        for b, k in n.items():
            ax.text(b, 101, f'n={k}', ha='center', va='bottom', fontsize=5.5, color=INK2)
    axes[0].set_ylabel(T['ydet'])
    handles = [Line2D([], [], color=style[g][1], ls=style[g][2], marker=style[g][3], ms=4, lw=style[g][4], label=names[g])
               for g in ('A0', 'A1', 'A14', 'A13b')]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.text(0.075, 0.98, '(a)', fontsize=8, fontweight='bold', va='top'); fig.text(0.53, 0.98, '(b)', fontsize=8, fontweight='bold', va='top')
    save_chart(fig, f'fig6_cnr_curve_{lang}')


# ---------------------------------------------------------------- Fig. 8
def boot_fp(C, a, b):
    C = C.copy(); C['group'] = C.run.str.rsplit('_s', n=1).str[0]
    m = C.groupby(['group', 'case']).fp.mean()
    cases = sorted(C.case.unique())
    d = (m.loc[a].reindex(cases) - m.loc[b].reindex(cases)).values
    idx = np.random.default_rng(SEED_BOOT).integers(0, len(cases), size=(N_BOOT, len(cases)))
    bd = d[idx].mean(1)
    return d.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5)


def fig_forest(lang):
    T = TXT[lang]
    tp = json.load(open(R / 'test_protocol/stats/protocol_results.json'))
    ep = json.load(open(R / 'external_msd08/stats/protocol_results.json'))
    a14 = json.load(open(R / 'a14_pixel2cancer/a14_stats.json'))['test']
    fpm = json.load(open(R / 'external_msd08/fp_analysis/fp_summary.json'))['rows'][0]
    Ct = pd.read_csv(R / 'test_protocol/tables/cases_pp4mm.csv')
    rc = {c['comparison']: c for c in tp['comparisons']}; re_ = {c['comparison']: c for c in ep['comparisons']}
    rows = [
        (T['lits'], T['gated'], (rc['A13b vs A1']['diff_pp'], rc['A13b vs A1']['ci95_low'], rc['A13b vs A1']['ci95_high']), boot_fp(Ct, 'A13b', 'A1'), True),
        (T['lits'], T['base'], (rc['A13b vs A0']['diff_pp'], rc['A13b vs A0']['ci95_low'], rc['A13b vs A0']['ci95_high']), boot_fp(Ct, 'A13b', 'A0'), True),
        (T['lits'], T['p2c'], tuple(a14['A13b - A14']['recall_low']), tuple(a14['A13b - A14']['fp_per_scan']), False),
        (T['msd'], T['gated'], (re_['A13b vs A1']['diff_pp'], re_['A13b vs A1']['ci95_low'], re_['A13b vs A1']['ci95_high']),
         (fpm['A13b-A1'], *fpm['A13b-A1_ci']), True),
        (T['msd'], T['base'], (re_['A13b vs A0']['diff_pp'], re_['A13b vs A0']['ci95_low'], re_['A13b vs A0']['ci95_high']),
         (fpm['A13b-A0'], *fpm['A13b-A0_ci']), True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 58 * MM), sharey=True, gridspec_kw=dict(wspace=0.08))
    ys = [4.6, 3.6, 2.6, 1.0, 0.0]
    blue = STYLE['A13b'][1]
    for ax, k, xl in ((axes[0], 2, T['d_recall']), (axes[1], 3, T['d_fp'])):
        ax.axvline(0, color=INK2, lw=.6, ls=(0, (2, 2)))
        for y, r in zip(ys, rows):
            v, lo, hi = r[k]
            ax.hlines(y, lo, hi, color=blue, lw=1.4)
            ax.plot(v, y, marker='o', ms=5, mfc=blue if r[4] else 'white', mec=blue, mew=1.2, zorder=3)
            ax.text(hi, y + 0.22, f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]', fontsize=5.6, color=INK2, ha='right', va='bottom')
        ax.set_xlabel(xl); ax.grid(axis='x', color=GRID, lw=.5); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([T['vs'] + r[1] for r in rows], fontsize=6.5)
    axes[0].set_ylim(-0.6, 5.5)
    for y, lab in ((5.2, T['lits']), (1.6, T['msd'])):
        axes[0].text(-0.02, y, lab, transform=axes[0].get_yaxis_transform(), ha='right', va='center', fontsize=7, fontweight='bold', color=INK)
    handles = [Line2D([], [], color=blue, marker='o', mfc=blue, mec=blue, ls='-', label=T['conf']),
               Line2D([], [], color=blue, marker='o', mfc='white', mec=blue, ls='-', label=T['desc'])]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.55, -0.02), frameon=False)
    fig.text(0.02, 0.98, '(a)', fontsize=8, fontweight='bold', va='top'); fig.text(0.53, 0.98, '(b)', fontsize=8, fontweight='bold', va='top')
    save_chart(fig, f'fig8_forest_{lang}')


# ---------------------------------------------------------------- Figs. 7 and 10
def lesion_grid(lang, prefixes, cols, row_info, name, pw=24 * MM, left=21 * MM):
    T = TXT[lang]
    D = np.load(HERE / 'figs_extra.npz')
    gap, top = 1.0 * MM, 6 * MM
    W = left + len(cols) * pw + (len(cols) - 1) * gap; H = top + len(prefixes) * pw + (len(prefixes) - 1) * gap
    fig = plt.figure(figsize=(W, H))
    for i, p in enumerate(prefixes):
        ct = window(D[f'{p}_ct']); tgt = D[f'{p}_target']
        for j, (key, title, color, status) in enumerate(cols):
            ax, y0 = grid_axes(fig, len(prefixes), len(cols), pw, gap, left, top, H, W, i, j)
            ax.imshow(ct, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
            if key == 'gt':
                contour(ax, D[f'{p}_gt'], GT_C, .7)
            elif key != 'ct':
                contour(ax, D[f'{p}_{key}'], color, .7)
            arrow(ax, tgt)
            st = status(i) if status else None
            if st is not None:
                tag(ax, *st)
            if i == 0:
                ax.set_title(title, fontsize=6.2, color=INK, pad=2)
            if j == 0:                                   # scale bar in every row (crop sizes differ)
                h = ct.shape[0]; px = float(D[f'{p}_px'])
                ax.plot([h - 5 - 10 / px, h - 5], [h - 5, h - 5], color='white', lw=1.2, solid_capstyle='butt')
                ax.text(h - 5, h - 8, '10 mm', color='white', fontsize=5, ha='right', va='bottom')
        y0 = 1 - (top + i * (pw + gap)) / H
        fig.text(1.2 * MM / W, y0 - 1 * MM / H, f'({chr(97 + i)})', fontsize=7.5, fontweight='bold', va='top', color=INK)
        fig.text(1.2 * MM / W, y0 - 5.5 * MM / H, row_info(i), fontsize=5.6, va='top', color=INK2, linespacing=1.45)
    save_img(fig, name)


def fig_multi(lang):
    T = TXT[lang]
    meta = json.loads(str(np.load(HERE / 'figs_extra.npz')['meta']))['multi']
    st = lambda run: (lambda i: (T['det'], True) if meta[i]['detected'][run] else (T['mis'], False))
    cols = [('ct', T['ct'], None, None), ('gt', T['ref'], None, None),
            ('A0_s0', T['base'], PRED_C, st('A0_s0')), ('A1_s0', T['gated'], PRED_C, st('A1_s0')),
            ('A12_s0', T['cc'], PRED_C, st('A12_s0')), ('A14_s0', T['p2c'], PRED_C, st('A14_s0')),
            ('A13b_s0', T['ours'], PRED_C, st('A13b_s0'))]
    info = lambda i: f"{T['cnr']} {meta[i]['cnr']:.2f}\n{T['diam']} {meta[i]['diameter']:.1f} mm\n{meta[i]['case'].replace('_', ' ')}"
    lesion_grid(lang, [f'multi{i}' for i in range(len(meta))], cols, info, f'fig10_multi_{lang}')


def fig_msd08(lang):
    T = TXT[lang]
    meta = json.loads(str(np.load(HERE / 'figs_extra.npz')['meta']))['msd08']
    def a1(i):
        return None if meta[i]['kind'] == 'fp' else (T['mis'], False)
    def ours(i):
        return (T['fp'], False) if meta[i]['kind'] == 'fp' else (T['det'], True)
    cols = [('ct', T['ct'], None, None), ('gt', T['task08ref'], None, None),
            ('A1', T['gated'], PRED_C, a1), ('A13b', T['ours'], PRED_C, ours)]
    def info(i):
        m = meta[i]; case = m['case'].replace('hepaticvessel_', 'case ')
        if m['kind'] == 'fp':
            return f"{T['fpk']}\n{T['diam']} {m['diameter']:.1f} mm\n{T['contrast']} {m['contrast']:.2f}\n{case}"
        return f"{T['gain']}\n{T['cnr']} {m['cnr']:.2f}\n{T['diam']} {m['diameter']:.1f} mm\n{case}"
    lesion_grid(lang, [f'msd{i}' for i in range(len(meta))], cols, info, f'fig7_msd08_{lang}', pw=33 * MM, left=27 * MM)


if __name__ == '__main__':
    for lang in sys.argv[1:] or ('zh', 'en'):
        fig_samples(lang); fig_cnr_curve(lang); fig_forest(lang); fig_multi(lang); fig_msd08(lang)
