#!/usr/bin/env python3
"""Fig. 5, qualitative comparison on the test set (zh and en).

    python manuscript/figures/src/fig_qualitative.py [zh] [en]

Crops come from qualitative.npz (extract_qualitative.py, server side); CNR,
diameter and detection state from the frozen protocol's lesion table
(results/test_protocol/tables/lesions_pp4mm.csv). Lesions follow the rule
registered in plan §11 (2026-10-14). Writes fig5_qualitative_{zh,en}.{pdf,png}.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_detect import INK, INK2, MM, OUT, ROOT  # noqa: E402  (also sets rcParams)

NPZ = Path(__file__).resolve().parent / 'qualitative.npz'
LESIONS = ROOT / 'results/test_protocol/tables/lesions_pp4mm.csv'
WL, WW = 70, 200                       # liver window: level, width (HU)
GT_C, PRED_C = '#1baf7a', '#eb6834'    # reference, prediction contours
COLS = ('ct', 'gt', 'A1_s0', 'A13b_s0')
TXT = {
    'zh': dict(cols=('CT', '金标准', 'nnU-Net + Gate', 'nnU-Net + DCA + CFC'),
               kind=dict(gain='增益', cost='代价', miss='共同漏检'),
               det='检出', mis='漏检', cnr='CNR', diam='直径'),
    'en': dict(cols=('CT', 'Reference', 'nnU-Net + Gate', 'nnU-Net + DCA + CFC'),
               kind=dict(gain='Gain', cost='Cost', miss='Missed by both'),
               det='detected', mis='missed', cnr='CNR', diam='diameter'),
}


def up(m, f=4):
    return ndi.zoom(m.astype(np.uint8), f, order=0) > 0


def contour(ax, m, color, lw, f=4, ls='-'):
    if m.any():
        h = m.shape[0]
        ax.contour(np.linspace(-.5, h - .5, h * f), np.linspace(-.5, h - .5, h * f), up(m, f),
                   levels=[.5], colors=color, linewidths=lw, linestyles=ls)


def draw(lang):
    T = TXT[lang]
    D = np.load(NPZ)
    L = pd.read_csv(LESIONS)
    n = sum(1 for k in D.files if k.endswith('_meta'))
    pw, gap, left, top = 36 * MM, 1.2 * MM, 22 * MM, 6 * MM
    W, H = left + 4 * pw + 3 * gap, top + n * pw + (n - 1) * gap
    fig = plt.figure(figsize=(W, H))
    for i in range(n):
        kind, case, lesion, _, px = D[f'{i}_meta']
        px = float(px)
        row = L[(L.case == case) & (L.lesion == int(lesion))].set_index('run')
        cnr, diam = row.cnr.iloc[0], row.diameter_mm.iloc[0]
        ct = np.clip((D[f'{i}_ct'] - (WL - WW / 2)) / WW, 0, 1)
        tgt = D[f'{i}_target']
        ty, tx = np.argwhere(tgt).mean(0)
        r = np.sqrt(tgt.sum() / np.pi)
        y0 = 1 - (top + i * (pw + gap) + pw) / H
        for j, col in enumerate(COLS):
            ax = fig.add_axes([(left + j * (pw + gap)) / W, y0, pw / W, pw / H])
            ax.imshow(ct, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col == 'gt':
                contour(ax, D[f'{i}_gt'], GT_C, .9)
            elif col != 'ct':
                contour(ax, D[f'{i}_{col}'], PRED_C, .9)
                hit = bool(row.loc[col, 'detected'])
                ax.text(.96, .04, T['det'] if hit else T['mis'], transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=6, color='white' if hit else '#ffd2c2', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc=(0, 0, 0, .55), ec='none'))
            # arrow to the target lesion, identical in every column
            d = r + 3.0
            ax.annotate('', xy=(tx - d * .62, ty + d * .62), xytext=(tx - d * .62 - 13, ty + d * .62 + 13),
                        arrowprops=dict(arrowstyle='-|>', color='white', lw=1.0, mutation_scale=7,
                                        shrinkA=0, shrinkB=0))
            if i == 0:
                ax.set_title(T['cols'][j], fontsize=7, color=INK, pad=3)
            if i == 0 and j == 0:  # 10-mm scale bar
                h = ct.shape[0]
                ax.plot([h - 6 - 10 / px, h - 6], [h - 6, h - 6], color='white', lw=1.4, solid_capstyle='butt')
                ax.text(h - 6 - 5 / px, h - 9, '10 mm', color='white', fontsize=5.5, ha='center', va='bottom')
        lab = chr(ord('a') + i)
        fig.text(1.5 * MM / W, y0 + pw / H - 1 * MM / H, f'({lab})', fontsize=8, fontweight='bold',
                 ha='left', va='top', color=INK)
        fig.text(1.5 * MM / W, y0 + pw / H - 6.5 * MM / H,
                 f"{T['kind'][kind]}\n{T['cnr']} {cnr:.2f}\n{T['diam']} {diam:.1f} mm\n{case.replace('_', ' ')}",
                 fontsize=6.2, ha='left', va='top', color=INK2, linespacing=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig5_qualitative_{lang}.{ext}', dpi=600, pad_inches=0.0)
    plt.close(fig)
    print('wrote', f'fig5_qualitative_{lang}')


if __name__ == '__main__':
    for lang in sys.argv[1:] or ('zh', 'en'):
        draw(lang)
