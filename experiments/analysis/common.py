"""
Shared helpers for the E11 / E12 / E13 analyses (manuscript Sections 4.8-4.9).

All three scripts work on the 21 fold-0 validation cases with the trained
`checkpoint_final.pth` of each run, no TTA, sliding window with step 0.5, i.e.
the same inference as the end-of-training validation. Each case is read and
preprocessed once and every model / condition is then run on the cached array,
so a condition costs one forward pass per case and nothing else.

Paths follow run.sh: $RESULTS_ROOT/<group>_s<seed>/Dataset102_LiTS/
nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth.
Override with DATA, RESULTS_ROOT, RAW_DATASET (env) or the CLI flags.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "experiments" / "configs"
if str(REPO / "nnUNet") not in sys.path:
    sys.path.insert(0, str(REPO / "nnUNet"))

DATA = Path(os.environ.get("DATA", "/root/autodl-tmp/nnUNet_data"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", DATA / "nnUNet_results_v2"))
RAW_DATASET = Path(os.environ.get("RAW_DATASET", DATA / "nnUNet_raw" / "Dataset102_LiTS"))
TRAINER_DIR = "Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres"
CNR_TERTILES = (1.2785, 1.9208)


# ---------------------------------------------------------------------------
# Runs, environment, models
# ---------------------------------------------------------------------------

def read_env(group: str) -> dict:
    env = {}
    for line in (CONFIGS / f"{group}.env").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            k, v = line.removeprefix("export ").split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def apply_group_env(group: str, seed: int) -> None:
    """Exactly the environment run.sh trains with (build_network_architecture
    reads the module switches from it)."""
    for k in [k for k in os.environ if k.startswith("PHASER_")]:
        del os.environ[k]
    os.environ.update(read_env(group))
    os.environ["PHASER_SEED"] = str(seed)
    # Pretrained weights are irrelevant at inference (the checkpoint is loaded).
    os.environ["VOCOMNI_NNUNET_CKPT"] = ""


@dataclass
class Model:
    name: str          # e.g. "A6_s0"
    group: str
    seed: int
    predictor: object  # nnUNetPredictor

    @property
    def network(self):
        return self.predictor.network

    @property
    def has_pfcr(self) -> bool:
        return bool(getattr(self.network, "enable_pfcr", False))


def parse_runs(specs: list[str]) -> list[tuple[str, int]]:
    """'A6:0,1,2' or 'A6_s0' -> [(group, seed), ...]"""
    out = []
    for spec in specs:
        if ":" in spec:
            g, seeds = spec.split(":", 1)
            out += [(g, int(s)) for s in seeds.split(",")]
        else:
            g, s = spec.split("_s")
            out.append((g, int(s)))
    return out


def model_dir(group: str, seed: int, results_root: Path = None) -> Path:
    return Path(results_root or RESULTS_ROOT) / f"{group}_s{seed}" / TRAINER_DIR


def load_model(group: str, seed: int, device: torch.device, results_root: Path = None,
               model_folder: Path = None) -> Model:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    apply_group_env(group, seed)
    folder = Path(model_folder) if model_folder else model_dir(group, seed, results_root)
    if not (folder / "fold_0" / "checkpoint_final.pth").is_file():
        raise FileNotFoundError(f"no checkpoint_final.pth under {folder}/fold_0")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           perform_everything_on_gpu=device.type == "cuda", device=device,
                           verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(folder), use_folds=(0,),
                                              checkpoint_name="checkpoint_final.pth")
    pred.network.load_state_dict(pred.list_of_parameters[0])
    pred.network.to(device).eval()
    return Model(f"{group}_s{seed}", group, seed, pred)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def read_cases(cases_file: str | None, image_dir: Path) -> list[str]:
    if cases_file:
        return [c for c in Path(cases_file).read_text().split() if c.strip()]
    return sorted(p.name[:-12] for p in image_dir.glob("*_0000.nii.gz"))


def shard(items: list, spec: str | None) -> list:
    """--shard i/n keeps every n-th item starting at i (0-based)."""
    if not spec:
        return items
    i, n = (int(v) for v in spec.split("/"))
    return items[i::n]


def read_raw(predictor, case: str, image_dir: Path, label_dir: Path | None):
    """Raw image (c, z, y, x) in HU, its properties, and the raw label (or None)."""
    rw = predictor.plans_manager.image_reader_writer_class()
    data, props = rw.read_images([str(image_dir / f"{case}_0000.nii.gz")])
    seg = None
    if label_dir is not None and (label_dir / f"{case}.nii.gz").is_file():
        seg, _ = rw.read_seg(str(label_dir / f"{case}.nii.gz"))
    return data, props, seg


def preprocess(predictor, data, props, seg=None):
    """nnU-Net preprocessing of an in-memory raw array. Returns (data, seg, props);
    props gains the cropping/resampling keys needed to export the prediction."""
    import copy

    cm, pm = predictor.configuration_manager, predictor.plans_manager
    pp = cm.preprocessor_class(verbose=False)
    props = copy.deepcopy(props)
    d, s = pp.run_case_npy(data, seg, props, pm, cm, predictor.dataset_json)
    return d, s, props


def predict_logits(model: Model, data: np.ndarray, preserve_dtype: bool = False) -> torch.Tensor:
    """Sliding-window logits in preprocessed space (no TTA)."""
    pred = model.predictor
    with torch.no_grad():
        logits = pred.predict_sliding_window_return_logits(torch.from_numpy(data))
    # End-of-training validation exports prediction.cpu() without promotion
    # (FP16 logits, resampled and cast back to FP16). Preserve its dtype when
    # the export should follow the validation path.
    return logits.cpu() if preserve_dtype else logits.float().cpu()


def export_seg(model: Model, logits: torch.Tensor, props: dict, out_file: Path) -> None:
    from nnunetv2.inference.export_prediction import \
        convert_predicted_logits_to_segmentation_with_correct_shape

    pred = model.predictor
    seg = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, pred.plans_manager, pred.configuration_manager, pred.label_manager, props)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pred.plans_manager.image_reader_writer_class().write_seg(seg, str(out_file), props)


def set_intervention(model: Model, mode: str, levels=None) -> None:
    """mode in PhaseFrequencyCoherenceRouting.INTERVENTIONS; levels = skip
    indices (0 = shallowest) or None for all."""
    if not model.has_pfcr:
        if mode != "none":
            raise ValueError(f"{model.name} has no PFCR modules")
        return
    for i, m in enumerate(model.network.pfcr_modules):
        m.intervene = mode if (levels is None or i in levels) else "none"


def evaluate(pred_dir: Path, out_dir: Path, label_dir: Path, image_dir: Path | None,
             cases_file: str | None, workers: int = 4) -> dict:
    cmd = [sys.executable, str(REPO / "experiments" / "eval" / "evaluate.py"),
           "--pred-dir", str(pred_dir), "--gt-dir", str(label_dir), "--out-dir", str(out_dir),
           "--workers", str(workers)]
    if image_dir is not None:
        cmd += ["--image-dir", str(image_dir)]
    if cases_file:
        cmd += ["--cases-file", str(cases_file)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((out_dir / "summary.json").read_text())


# ---------------------------------------------------------------------------
# Lesions and CNR in preprocessed space
# ---------------------------------------------------------------------------

def lesion_cnr_preprocessed(data: np.ndarray, seg: np.ndarray, min_voxels: int = 1):
    """Lesions (6-connected components of label 2) of a preprocessed case and
    their CNR with the rule of s42_real_pc.py: liver ring at Euclidean voxel
    distance 2..6 (fallback 1..6 if < 20 voxels; undefined if < 10), other
    lesions excluded. Preprocessed data is on the plan spacing, and nnU-Net's CT
    normalization is a global affine map, which leaves CNR unchanged (the only
    difference from the frozen-threshold computation is spline instead of linear
    resampling of the image).
    Returns (label map, [dict(label, voxels, cnr, stratum, bbox)])."""
    img = data[0]
    s = seg[0] if seg.ndim == 4 else seg
    tumor, liver = s == 2, s == 1
    lab, n = ndi.label(tumor)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m_full = lab == i
        nvox = int(m_full.sum())
        if nvox < min_voxels:
            continue
        lo = [max(x.start - 10, 0) for x in sl]
        hi = [min(x.stop + 10, d) for x, d in zip(sl, lab.shape)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        m, liv, oth = m_full[box], liver[box], (tumor & ~m_full)[box]
        d_out = ndi.distance_transform_edt(~m)
        ring = (~m) & liv & (d_out >= 2) & (d_out <= 6) & ~oth
        if ring.sum() < 20:
            ring = (~m) & liv & (d_out >= 1) & (d_out <= 6) & ~oth
        cnr = np.nan
        if ring.sum() >= 10:
            im = img[box]
            cnr = abs(float(im[m].mean()) - float(im[ring].mean())) / (float(im[ring].std()) + 1e-8)
        out.append(dict(label=i, voxels=nvox, cnr=cnr, stratum=cnr_stratum(cnr),
                        bbox=[(x.start, x.stop) for x in sl]))
    return lab, out


def cnr_stratum(c: float) -> str:
    if not np.isfinite(c):
        return "na"
    return "low" if c < CNR_TERTILES[0] else ("mid" if c < CNR_TERTILES[1] else "high")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float, ensure_ascii=False))
