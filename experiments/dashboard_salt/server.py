#!/usr/bin/env python3
"""Live progress dashboard for the SALT runs (docs/SALT模型设计.md).

Reads only: nnU-Net training logs, results/salt/ runner logs and decisions, nvidia-smi, /proc.
Standard library only, so it runs with any python3 on the server.

    python3 experiments/dashboard_salt/server.py --host 127.0.0.1 --port 6008
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "salt"
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "/root/autodl-tmp/nnUNet_data/nnUNet_results_v2"))
EXT_CASES = Path(os.environ.get("EXT", "/root/autodl-tmp/ext_data/msd08/Dataset104_MSD08")) / "cases.txt"
DISK = os.environ.get("DASH_DISK", "/root/autodl-tmp")
TRAINER = "Dataset102_LiTS/nnUNetTrainer_phaser__nnUNetPlans__3d_fullres/fold_0"
GROUPS = [  # (group, description), in queue order of server_salt_all.sh
    ("Q4", "SALT 完整：查询解码器 + SMD + FAF-3D"),
    ("Q1", "病灶查询解码器（32 个自由查询）"),
    ("F1", "只加 FAF-3D 频率感知融合"),
    ("Q3", "Q1 + SMD 谱矩形状描述子（Q4 通过后才训练）"),
]
REFERENCE = "A0_s0"
SEED = 0

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
RE_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _ts(line: str) -> float | None:
    m = RE_TS.match(line)
    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) if m else None


def parse_training(fold: Path) -> dict:
    """Per-epoch records from all training_log_*.txt of a fold (merged, so restarts are covered)."""
    epochs: dict[int, dict] = {}
    cur = None
    nonfinite = 0
    last_ts = None
    for f in sorted(fold.glob("training_log_*.txt")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            t = _ts(line)
            if t:
                last_ts = t
            body = line.split(": ", 1)[1].strip() if ": " in line else line.strip()
            if m := re.fullmatch(r"Epoch (\d+)", body):
                cur = int(m.group(1))
                epochs.setdefault(cur, {"epoch": cur, "start": t})
                epochs[cur]["start"] = t
            elif cur is None:
                continue
            elif body.startswith("Current learning rate:"):
                epochs[cur]["lr"] = float(re.findall(FLOAT, body)[-1])
            elif body.startswith("train_loss"):
                epochs[cur]["train_loss"] = float(re.findall(FLOAT, body)[-1])
            elif body.startswith("val_loss"):
                epochs[cur]["val_loss"] = float(re.findall(FLOAT, body)[-1])
            elif body.startswith("Pseudo dice"):
                vals = [float(v) for v in re.findall(FLOAT, body.replace("float32", "").replace("float64", ""))]
                epochs[cur]["dice"] = vals
            elif body.startswith("Epoch time"):
                epochs[cur]["time"] = float(re.findall(FLOAT, body)[0])
                epochs[cur]["end"] = t
            elif "non-finite" in body:
                nonfinite += 1
                epochs[cur]["nonfinite"] = epochs[cur].get("nonfinite", 0) + 1
    done = [e for e in sorted(epochs.values(), key=lambda e: e["epoch"]) if "time" in e]
    ema = None
    for e in done:  # nnU-Net's EMA of the mean foreground pseudo Dice, and the same EMA for the tumour alone
        d = e.get("dice") or []
        if d:
            m = sum(d) / len(d)
            ema = m if ema is None else 0.9 * ema + 0.1 * m
            e["ema"] = ema
    t_ema = None
    for e in done:
        d = e.get("dice") or []
        if len(d) >= 2:
            t_ema = d[1] if t_ema is None else 0.9 * t_ema + 0.1 * d[1]
            e["tumour_ema"] = t_ema
    return {"epochs": done, "current": cur, "nonfinite": nonfinite, "last_log": last_ts}


def num_epochs(group: str) -> int:
    cfg = REPO / "experiments" / "configs" / f"{group}.env"
    try:
        for line in cfg.read_text().splitlines():
            if line.startswith("PHASER_NUM_EPOCHS="):
                return int(line.split("=", 1)[1])
    except OSError:
        pass
    return 200


def tail(path: Path, n: int) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def run_state(group: str) -> dict:
    run = f"{group}_s{SEED}"
    fold = RESULTS_ROOT / run / TRAINER
    tr = parse_training(fold)
    total = num_epochs(group)
    eps = tr["epochs"]
    runner = tail(OUT / f"runner_{run}.log", 50)
    joined = "\n".join(runner)
    now = time.time()
    decision_path = OUT / f"pilot_msd08_{run}" / "decision.json"
    n_cases = len(tail(EXT_CASES, 100000)) or 303
    pred_dir = fold / "external_msd08_pred"
    n_pred = len(glob.glob(str(pred_dir / "*.nii.gz")))

    stage, detail = "queued", "排队中"
    if "FAILED" in joined or "refusing" in joined or "mismatch" in joined:
        stage, detail = "failed", next(l for l in reversed(runner) if re.search("FAILED|refusing|mismatch", l))
    elif decision_path.exists():
        stage, detail = "done", "已完成"
    elif "MSD08 predicted" in joined:
        stage, detail = "eval", "MSD08 评测与判定中"
    elif "trained and validated" in joined:
        stage, detail = "predict", f"MSD08 推理 {n_pred}/{n_cases}"
    elif "pilot start" in joined:
        if (fold / "checkpoint_final.pth").exists():
            stage, detail = "validate", "训练完成，LiTS 验证集推理与评测中"
        else:
            stage, detail = "train", "训练中"
    if group == "Q3" and stage == "queued":
        detail = "等待 Q4 通过继续条件（§4）"

    stale = tr["last_log"] is not None and stage == "train" and now - tr["last_log"] > 1200
    times = [e["time"] for e in eps[-5:]]
    avg = sum(times) / len(times) if times else None
    done_ep = len(eps)
    eta = avg * (total - done_ep) if (avg and stage == "train") else None
    last = eps[-1] if eps else {}
    best_t = max((e["dice"][1] for e in eps if len(e.get("dice") or []) >= 2), default=None)

    decision = None
    if decision_path.exists():
        try:
            d = json.loads(decision_path.read_text())
            decision = {"continue": d.get("runs", {}).get(run, {}).get("CONTINUE"),
                        "md": (decision_path.parent / "DECISION.md").read_text(errors="replace")
                        if (decision_path.parent / "DECISION.md").exists() else ""}
        except (OSError, ValueError):
            pass

    return {
        "group": group, "run": run, "stage": stage, "detail": detail, "stale": stale,
        "epochs_done": done_ep, "epochs_total": total, "avg_epoch_s": avg, "eta_s": eta,
        "started": eps[0]["start"] if eps else None,
        "last": {k: last.get(k) for k in ("epoch", "train_loss", "val_loss", "dice", "lr", "ema", "tumour_ema")},
        "best_tumour": best_t, "nonfinite": tr["nonfinite"],
        "n_pred": n_pred, "n_cases": n_cases,
        "series": [{"e": e["epoch"], "tl": e.get("train_loss"), "vl": e.get("val_loss"),
                    "d": e.get("dice"), "te": e.get("tumour_ema"), "ema": e.get("ema")} for e in eps],
        "runner_tail": runner[-6:], "decision": decision,
    }


def reference_series() -> list[dict]:
    tr = parse_training(RESULTS_ROOT / REFERENCE / TRAINER)
    return [{"e": e["epoch"], "te": e.get("tumour_ema"), "tl": e.get("train_loss"), "vl": e.get("val_loss")}
            for e in tr["epochs"]]


def gpus() -> list[dict]:
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    res = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        res.append({"index": int(p[0]), "name": p[1], "util": num(p[2]), "mem_used": num(p[3]),
                    "mem_total": num(p[4]), "temp": num(p[5]), "power": num(p[6])})
    return res


def system() -> dict:
    load = os.getloadavg()
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            mem[k] = int(v.split()[0]) / 1048576
    except OSError:
        pass
    du = shutil.disk_usage(DISK)
    tmux = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout
    return {"load": load, "ncpu": os.cpu_count(), "mem_total": mem.get("MemTotal"),
            "mem_avail": mem.get("MemAvailable"), "disk_path": DISK, "disk_total": du.total / 2**30,
            "disk_free": du.free / 2**30, "runner_alive": any(l.startswith("salt:") for l in tmux.splitlines())}


def g0() -> dict | None:
    try:
        return json.loads((OUT / "g0" / "summary.json").read_text())
    except (OSError, ValueError):
        return None


def head_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


_cache: dict = {"t": 0.0, "data": None}
_lock = threading.Lock()


def status() -> dict:
    with _lock:
        if _cache["data"] is None or time.time() - _cache["t"] > 10:
            _cache["data"] = {
                "now": time.time(), "head": head_commit(), "runs": [run_state(g) for g, _ in GROUPS],
                "desc": dict(GROUPS), "reference": {"run": REFERENCE, "series": reference_series()},
                "gpus": gpus(), "system": system(), "g0": g0(), "all_log": tail(OUT / "all.log", 25),
            }
            _cache["t"] = time.time()
        return _cache["data"]


INDEX = (Path(__file__).parent / "index.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            try:
                self._send(200, json.dumps(status()).encode(), "application/json")
            except Exception as e:  # keep serving; show the error on the page
                self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6008)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"SALT dashboard on {a.host}:{a.port}, repo {REPO}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
