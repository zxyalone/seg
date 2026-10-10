#!/usr/bin/env python3
"""
End-to-end test of E11 / E12 / E13 on CPU, no real data needed.

Builds a small synthetic LiTS-like dataset (liver + low-contrast lesions in HU),
lets nnU-Net plan it, writes checkpoint_final.pth files in the real format for
A0 / A1 / A6 / A7 (untrained weights; PFCR gates opened so interventions act),
then runs every analysis on two cases and checks:
  * E11 'none' predictions equal nnUNetv2_predict (no TTA) voxel for voxel;
  * interventions change the output, gate_zero differs from none;
  * E12 alpha = 1 equals E11 'none'; realised lesion CNR = alpha * CNR;
  * E13 recovers the initial filter wavelengths of an untrained checkpoint
    and its route / PC self-test passes.

    PYTHONPATH=nnUNet python experiments/analysis/test_analysis.py   (about 10-20 min on CPU)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPACING_XYZ = (0.78, 0.78, 1.0)
SHAPE = (40, 72, 72)  # z, y, x


def make_dataset(raw: Path, n: int = 4):
    ds = raw / "Dataset102_LiTS"
    (ds / "imagesTr").mkdir(parents=True)
    (ds / "labelsTr").mkdir()
    rng = np.random.default_rng(0)
    z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    for i in range(n):
        lab = np.zeros(SHAPE, np.uint8)
        c = np.array(SHAPE) / 2 + rng.normal(0, 2, 3)
        liver = ((z - c[0]) / 15) ** 2 + ((y - c[1]) / 26) ** 2 + ((x - c[2]) / 26) ** 2 < 1
        lab[liver] = 1
        for _ in range(3):
            tc = c + rng.normal(0, [4, 9, 9])
            r = rng.uniform(2.5, 6)
            t = ((z - tc[0]) ** 2 + ((y - tc[1]) * .78) ** 2 + ((x - tc[2]) * .78) ** 2 < r ** 2) & liver
            lab[t] = 2
        ct = np.where(liver, 100.0, -80.0) + np.where(lab == 2, -rng.uniform(15, 30), 0.0)
        ct = (ct + rng.normal(0, 10, SHAPE)).astype(np.float32)
        for arr, path in ((ct, ds / "imagesTr" / f"LiTS_{i:03d}_0000.nii.gz"),
                          (lab, ds / "labelsTr" / f"LiTS_{i:03d}.nii.gz")):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(SPACING_XYZ)
            sitk.WriteImage(img, str(path))
    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "liver": 1, "tumor": 2},
        "numTraining": n, "file_ending": ".nii.gz"}))
    return ds


BUILD = r'''
import json, os, sys, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments/analysis")
import common as C
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_phaser import nnUNetTrainer_phaser
pre, results = sys.argv[2], sys.argv[3]
plans = json.load(open(pre + "/Dataset102_LiTS/nnUNetPlans.json"))
dj = json.load(open(pre + "/Dataset102_LiTS/dataset.json"))
pm = PlansManager(plans); cm = pm.get_configuration("3d_fullres")
for run in sys.argv[4:]:
    g, s = run.split("_s"); C.apply_group_env(g, int(s)); torch.manual_seed(int(s))
    net = nnUNetTrainer_phaser.build_network_architecture(pm, dj, cm, 1, enable_deep_supervision=False)
    if getattr(net, "enable_pfcr", False):
        for m in net.pfcr_modules:
            m.residual_gate.data.fill_(1.0)
            torch.nn.init.normal_(m.router[-1].weight, std=1.0)
    if getattr(net, "enable_pcsf", False):
        for m in net.pcsf_modules:
            m.gamma.data.fill_(1.0)
    folder = C.model_dir(g, int(s), results); (folder / "fold_0").mkdir(parents=True, exist_ok=True)
    json.dump(plans, open(folder / "plans.json", "w")); json.dump(dj, open(folder / "dataset.json", "w"))
    torch.save({"network_weights": net.state_dict(), "trainer_name": "nnUNetTrainer_phaser",
                "init_args": {"configuration": "3d_fullres"}, "inference_allowed_mirroring_axes": (0, 1, 2)},
               folder / "fold_0" / "checkpoint_final.pth")
    print("built", run, sum(p.numel() for p in net.parameters()))
'''

REFPRED = r'''
import sys, torch
sys.path.insert(0, sys.argv[1] + "/experiments/analysis")
import common as C
C.apply_group_env("A6", 0)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                    perform_everything_on_gpu=False, device=torch.device("cpu"), allow_tqdm=False)
p.initialize_from_trained_model_folder(str(C.model_dir("A6", 0, sys.argv[2])), use_folds=(0,))
p.predict_from_files([[sys.argv[3] + f"/{c}_0000.nii.gz"] for c in sys.argv[5:]], sys.argv[4],
                     save_probabilities=False, overwrite=True, num_processes_preprocessing=1,
                     num_processes_segmentation_export=1)
'''


def arr(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw, pre, res = root / "raw", root / "pre", root / "res"
        ds = make_dataset(raw)
        env = dict(os.environ, nnUNet_raw=str(raw), nnUNet_preprocessed=str(pre), nnUNet_results=str(res),
                   RESULTS_ROOT=str(res), RAW_DATASET=str(ds), DATA=str(root),
                   PYTHONPATH=str(REPO / "nnUNet") + os.pathsep + os.environ.get("PYTHONPATH", ""))

        def py(*a, **k):
            subprocess.run([sys.executable, *a], check=True, env=env, **k)

        py("-c", "from nnunetv2.experiment_planning.plan_and_preprocess_api import "
                 "extract_fingerprints, plan_experiments; extract_fingerprints([102], num_processes=1, "
                 "check_dataset_integrity=True); plan_experiments([102])")
        runs = ["A0_s0", "A1_s0", "A6_s0", "A6_s1", "A7_s0"]
        py("-c", BUILD, str(REPO), str(pre), str(res), *runs)
        cases = ["LiTS_000", "LiTS_001"]
        cf = root / "cases.txt"
        cf.write_text("\n".join(cases) + "\n")
        common = ["--results-root", str(res), "--image-dir", str(ds / "imagesTr"),
                  "--label-dir", str(ds / "labelsTr"), "--cases-file", str(cf)]

        # E11 (two shards, as on the server) + summary
        e11 = root / "E11"
        for sh in ("0/2", "1/2"):
            py(str(HERE / "e11_interventions.py"), "run", "--runs", "A6:0,1", "--out", str(e11),
               "--device", "cpu", "--shard", sh, *common)
        py(str(HERE / "e11_interventions.py"), "summarize", "--runs", "A6:0,1", "--out", str(e11),
           "--n-boot", "200", "--workers", "1", *common)
        ref_dir = root / "ref_pred"
        py("-c", REFPRED, str(REPO), str(res), str(ds / "imagesTr"), str(ref_dir), *cases)
        for c in cases:
            a, b = arr(e11 / "pred" / "A6_s0" / "none" / f"{c}.nii.gz"), arr(ref_dir / f"{c}.nii.gz")
            assert np.array_equal(a, b), f"E11 none differs from nnUNetv2_predict for {c}"
        changed = [not np.array_equal(arr(e11 / "pred" / "A6_s0" / cond / f"{cases[0]}.nii.gz"),
                                      arr(e11 / "pred" / "A6_s0" / "none" / f"{cases[0]}.nii.gz"))
                   for cond in ("gate_zero", "route_uniform", "pc_zero@all", "prior_zero")]
        assert changed[0], "gate_zero did not change the prediction"
        util = json.loads((e11 / "utility" / "A6_s0" / f"{cases[0]}.json").read_text())
        assert set(util["loss"]) >= {"none", "gate_zero", "pc_zero@0"} and util["lesions"]
        assert (e11 / "table6_interventions.md").is_file()
        print("E11 ok; conditions that changed the output:", changed)

        # E12
        e12 = root / "E12"
        py(str(HERE / "e12_degradation.py"), "run", "--runs", "A0:0", "A1:0", "A6:0", "A7:0",
           "--out", str(e12), "--device", "cpu", *common)
        py(str(HERE / "e12_degradation.py"), "summarize", "--runs", "A0:0", "A1:0", "A6:0", "A7:0",
           "--out", str(e12), "--n-boot", "200", "--workers", "1", *common)
        for c in cases:
            assert np.array_equal(arr(e12 / "pred" / "A6_s0" / "a1" / f"{c}.nii.gz"),
                                  arr(e11 / "pred" / "A6_s0" / "none" / f"{c}.nii.gz")), "E12 a1 != E11 none"
        chk = json.loads((e12 / "e12_cnr_check.json").read_text())
        for a in (1.0, 0.75, 0.5, 0.25):
            assert abs(chk[f"a{a:g}"] - a) < 0.02, chk
        print("E12 ok; realised CNR / CNR:", chk)

        # E13
        e13 = root / "E13"
        py(str(HERE / "e13_mechanism.py"), "--runs", "A6:0,1", "A7:0", "--out", str(e13), "--device", "cpu",
           "--min-voxels", "20", *common)
        import pandas as pd
        f = pd.read_csv(e13 / "filters.csv")
        assert f.rel_shift.max() < 1e-4 and len(f) == 3 * 3 * f.level.nunique(), f.head()
        st = pd.read_csv(e13 / "lesion_level_stats.csv")
        assert len(st) > 0 and st.sel_l1.notna().all()
        print(f"E13 ok; {len(st)} lesion x level rows, levels {sorted(st.level.unique())}")
    print("test_analysis: all checks passed")


if __name__ == "__main__":
    main()
