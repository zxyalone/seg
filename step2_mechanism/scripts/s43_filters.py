#!/usr/bin/env python3
"""
Step 2 / 4.3 — 滤波器分析（检验 C3 前提）

命题 C3（训练）：PC ranking 监督使可学习 log-Gabor 适配病灶尺度。
证伪：去掉 ranking 后滤波器中心频率、PC gap、路由分布无变化。

本节只做 **C3 的前提检验**（不训练）：
  1. checkpoint 中 pfcr_modules.*.pc.rho_raw / sigma_raw / noise_k 是否相对初始值发生
     显著偏移（"可学习滤波器"是否真的学动过）；
  2. 5 个注入尺度（e0..e4）之间的中心频率是否分化（还是退化为同一组滤波器）；
  3. router 末层权重是否仍为零 —— 若恒为零，softmax 路由在空间上恒为常数，
     所谓"相位条件路由"退化为静态全图带通滤波（这一步是定量证据）。
  4. residual_gate / proxy 的实测量级。

注：validation_plan 4.3 期望的 "A5（无 PC ranking）vs A6（有 ranking）" 对照在本机
尚不存在（现有全部 run 的 PHASER_L_PC_BOUNDARY 均为 1）。脚本会显式报告这一点，
并以"随机初始化 vs 各 checkpoint"以及"Copy2(epoch48) vs 重训(Phase B 已开)"作为
可用的替代对照。
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/Downstream_2/step2_mechanism/scripts")
from common import CKPTS, FIGS, banner, save_json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

MIN_WAVELENGTH = 3.0
MULT = 2.1
SIGMA_ONF = 0.55
NOISE_K = 2.0
N_SKIP = 5


def init_wavelengths(n_scales=3):
    return [MIN_WAVELENGTH * MULT ** s for s in range(n_scales)]


def softplus(x):
    return float(np.log1p(np.exp(x)))


def load_sd(path):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("network_weights", ck.get("state_dict", ck))
    meta = dict(current_epoch=ck.get("current_epoch"), best_ema=ck.get("_best_ema"))
    return sd, meta


def analyse(path):
    sd, meta = load_sd(path)
    out = {"path": str(path), "current_epoch": meta["current_epoch"],
           "best_ema": meta["best_ema"], "modules": []}
    for i in range(N_SKIP):
        pre = f"pfcr_modules.{i}."
        mod = {"index": i}
        for key, name in (("pc.rho_raw", "rho_raw"), ("pc.sigma_raw", "sigma_raw"),
                          ("pc.noise_k", "noise_k_raw")):
            t = sd.get(pre + key)
            mod[name] = None if t is None else t.detach().float().cpu().numpy().tolist()
        for key, name in (("residual_gate", "residual_gate"),
                          ("router.0.weight", "router0_w"),
                          ("router.0.bias", "router0_b"),
                          ("router.2.weight", "router2_w"),
                          ("router.2.bias", "router2_b"),
                          ("proxy.weight", "proxy_w")):
            t = sd.get(pre + key)
            if t is None:
                mod[name] = None
                continue
            a = t.detach().float().cpu().numpy()
            mod[name] = dict(shape=list(a.shape), absmax=float(np.abs(a).max()),
                             absmean=float(np.abs(a).mean()), mean=float(a.mean()))
        if mod["rho_raw"] is not None:
            rho = 0.5 / (1.0 + np.exp(-np.array(mod["rho_raw"])))
            mod["rho"] = rho.tolist()
            mod["wavelength_voxels"] = (1.0 / np.clip(rho, 1e-9, None)).tolist()
            mod["lambda_over_init"] = (np.array(mod["wavelength_voxels"])
                                       / np.array(init_wavelengths())).tolist()
        if mod["sigma_raw"] is not None:
            mod["sigma_onf"] = (0.05 + 0.9 / (1.0 + np.exp(-np.array(mod["sigma_raw"])))).tolist()
        if mod["noise_k_raw"] is not None:
            mod["noise_k"] = softplus(mod["noise_k_raw"])
        if mod["residual_gate"] is not None:
            mod["tanh_residual_gate"] = float(np.tanh(mod["residual_gate"]["mean"]))
        out["modules"].append(mod)
    return out


def main():
    banner("Step 2 / 4.3 滤波器分析 — log-Gabor 是否真的学动了（C3 前提）")
    init_lam = init_wavelengths()
    print(f"初始设定：min_wavelength={MIN_WAVELENGTH}  mult={MULT}  "
          f"→ λ_init = {['%.2f' % v for v in init_lam]} 体素")
    init_rho = [1.0 / v for v in init_lam]
    print(f"初始 ρ_init = {['%.4f' % v for v in init_rho]}   sigma_onf={SIGMA_ONF}   noise_k={NOISE_K}")

    result = {"init": dict(min_wavelength=MIN_WAVELENGTH, mult=MULT,
                           wavelengths=init_lam, rho=init_rho,
                           sigma_onf=SIGMA_ONF, noise_k=NOISE_K),
              "checkpoints": {}}

    for tag, path in CKPTS.items():
        if not path.is_file():
            print(f"\n[skip] {tag}: 不存在")
            continue
        info = analyse(path)
        result["checkpoints"][tag] = info
        ep = info["current_epoch"]
        print(f"\n{'='*74}\n{tag}   epoch={ep}  best_ema={info['best_ema']}\n{'='*74}")
        print(f"{'mod':>3} | {'λ1':>7} {'λ2':>7} {'λ3':>7} | {'λ/λ_init':>20} | "
              f"{'sigma':>17} | {'k':>5} | {'tanh(g)':>8} | {'|W2|max':>8} | {'|W2|mean':>9} | {'bias2':>16}")
        for m in info["modules"]:
            lam = m["wavelength_voxels"]
            lamr = m["lambda_over_init"]
            sig = m["sigma_onf"]
            print(f"{m['index']:>3} | {lam[0]:7.3f} {lam[1]:7.3f} {lam[2]:7.3f} | "
                  f"{lamr[0]:6.3f} {lamr[1]:6.3f} {lamr[2]:6.3f}       | "
                  f"{sig[0]:6.3f} {sig[1]:6.3f} {sig[2]:6.3f} | "
                  f"{m['noise_k']:5.3f} | {m['tanh_residual_gate']:+8.4f} | "
                  f"{m['router2_w']['absmax']:8.2e} | {m['router2_w']['absmean']:9.2e} | "
                  f"{str([round(v,4) for v in (m['router2_b']['mean'],)]):>16}")
        # 跨尺度分化度
        rho = np.array([m["rho"] for m in info["modules"]])          # [5,3]
        lam = np.array([m["wavelength_voxels"] for m in info["modules"]])
        cross = lam.std(axis=0, ddof=0)
        print(f"  跨 5 个注入尺度的 λ 标准差: {np.round(cross, 4)}  "
              f"(跨尺度 λ 均值 {np.round(lam.mean(0), 3)})")
        info["cross_module_lambda_std"] = cross.tolist()
        info["cross_module_lambda_mean"] = lam.mean(0).tolist()
        info["router2_w_absmax"] = [m["router2_w"]["absmax"] for m in info["modules"]]
        info["router2_w_absmean"] = [m["router2_w"]["absmean"] for m in info["modules"]]

    # ---------------- 汇总：是否"学动了" ----------------
    print("\n" + "-" * 74)
    print("汇总：相对初始值的漂移与路由退化程度")
    print(f"{'checkpoint':22s} {'max|Δλ|/λ':>11} {'跨尺度λstd':>11} "
          f"{'|W2|max':>10} {'路由是否空间恒定':>18}")
    summary = {}
    for tag, info in result["checkpoints"].items():
        dev = np.abs(np.array([m["lambda_over_init"] for m in info["modules"]]) - 1.0).max()
        cstd = float(np.max(info["cross_module_lambda_std"]))
        w2 = float(np.max(info["router2_w_absmax"]))
        const = "是（退化）" if w2 < 1e-3 else ("部分" if w2 < 5e-2 else "否")
        summary[tag] = dict(epoch=info["current_epoch"], best_ema=info["best_ema"],
                            max_rel_lambda_dev=float(dev),
                            max_cross_module_lambda_std=cstd,
                            router2_max_absmax=w2,
                            router_spatially_constant=bool(w2 < 1e-3))
        print(f"{tag:22s} {dev:11.4f} {cstd:11.4f} {w2:10.2e} {const:>18}")
    result["summary"] = summary

    # ---------------- 作图 ----------------
    tags = [t for t in result["checkpoints"]]
    if tags:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
        x = np.arange(len(tags))
        w = 0.25

        ax = axes[0]
        for s in range(3):
            vals = [result["checkpoints"][t]["cross_module_lambda_mean"][s] for t in tags]
            ax.bar(x + (s - 1) * w, vals, w, label=f"scale {s+1} (init λ={init_lam[s]:.1f})")
            ax.axhline(init_lam[s], color=f"C{s}", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("learned center wavelength (voxels)")
        ax.set_title("Learned log-Gabor wavelength vs init (dashed)")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

        ax = axes[1]
        devs = [summary[t]["max_rel_lambda_dev"] for t in tags]
        ax.bar(x, devs, 0.5, color="#D85A30")
        ax.axhline(0.1, color="gray", ls="--", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("max |lambda / lambda_init - 1|")
        ax.set_title("Filter drift from initialization")
        ax.grid(alpha=.25, axis="y")

        ax = axes[2]
        w2 = [max(summary[t]["router2_max_absmax"], 1e-12) for t in tags]
        ax.bar(x, w2, 0.5, color="#534AB7")
        ax.axhline(1e-3, color="red", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("max |router[-1].weight|")
        ax.set_title("Router head weight (0 => route is a spatial constant)")
        ax.grid(alpha=.25, axis="y")

        fig.suptitle("4.3 Filter analysis: are the learnable log-Gabor scales actually learned? (C3 precondition)",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        fp = FIGS / "fig43_filter_analysis.png"
        fig.savefig(fp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[fig] {fp}")

    result["note_A5_missing"] = (
        "validation_plan 4.3 要求对比 A5(无 PC ranking) vs A6(有 ranking) 的滤波器分布；"
        "本机现有全部 run 的 PHASER_L_PC_BOUNDARY 均为 1，A5 不存在，故以"
        "随机初始化 + Copy2(epoch48, 仅 Phase A) + 重训(Phase B 已开) 作为替代对照。"
    )
    save_json(result, "s43_filters.json")
    banner("4.3 完成")


if __name__ == "__main__":
    main()
