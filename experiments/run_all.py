"""Reproduce every result of the study: runs, tables, figures.

Usage:
    python experiments/run_all.py            # full study (about 10-15 minutes on a laptop CPU)
    python experiments/run_all.py --quick    # smoke test (< 1 minute)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from fisherfed.federated import Aggregator, FedConfig, run_federated  # noqa: E402

# Base operating point: heterogeneous noisy clients, few shots per circuit.
BASE = FedConfig(shots=8, p_med=0.08, sigma=1.2, rounds=40)
SEEDS = [0, 1, 2, 3, 4]

# method key -> (mode, corrected, alpha, display name)
METHODS: dict[str, tuple[str, bool, float, str]] = {
    "fedavg": ("quantum", False, 0.0, "FedAvg (data-size weights)"),
    "reliability": ("quantum", False, 1.0, "Reliability-weighted (alpha=1)"),
    "corrected": ("quantum", True, 0.0, "Corrected, data-size weights"),
    "fisher": ("quantum", False, 2.0, "Fisher-weighted only (alpha=2)"),
    "proposed": ("quantum", True, 2.0, "Corrected + Fisher-weighted (alpha=2)"),
    "oracle": ("oracle", False, 0.0, "Ideal devices, exact gradients"),
    "classical": ("classical", False, 0.0, "Classical MLP, 19 parameters"),
}


def alpha_key(corrected: bool, alpha: float) -> str:
    """Method key for an arbitrary (corrected, alpha) pair, reusing the named methods when they match."""
    for k, (mode, c, a, _) in METHODS.items():
        if mode == "quantum" and c == corrected and a == alpha:
            return k
    return f"{'corr' if corrected else 'raw'}_a{alpha:g}"


def to_aggregator(key: str) -> Aggregator:
    if key in METHODS:
        mode, c, a, name = METHODS[key]
        return Aggregator(name, mode=mode, corrected=c, alpha=a)
    corrected, alpha = key.startswith("corr"), float(key.split("_a")[1])
    return Aggregator(f"{'Corrected' if corrected else 'Uncorrected'}, alpha={alpha:g}", corrected=corrected, alpha=alpha)


def canon(cfg: FedConfig, key: str) -> FedConfig:
    """Drop fields that a method ignores so that identical runs are executed once."""
    mode = METHODS[key][0] if key in METHODS else "quantum"
    if mode == "classical":
        return replace(cfg, shots=None, p_med=0.0, sigma=0.0, noise_kind="depolarizing", cal_error=0.0)
    if mode == "oracle":
        return replace(cfg, shots=None, cal_error=0.0)
    return cfg


def build_plan(quick: bool) -> list[tuple[str, FedConfig, str]]:
    """List of (experiment, config, method key) triples; seeds are added later."""
    base = replace(BASE, rounds=4, eval_every=2, batch=2, cal_probes=4, cal_shots=32) if quick else BASE
    plan: list[tuple[str, FedConfig, str]] = []
    shots = [8, None] if quick else [8, 32, None]
    main = ["fedavg", "reliability", "corrected", "proposed"]
    for s in shots:
        plan += [("shots", replace(base, shots=s), m) for m in main]
    plan += [("shots", base, m) for m in ("oracle", "classical", "fisher")]
    alphas = [0.0, 2.0] if quick else [0.0, 1.0, 2.0, 4.0]
    for a in alphas:
        plan += [("alpha", base, alpha_key(c, a)) for c in (False, True)]
    for sg in [0.0, 1.2] if quick else [0.0, 0.6, 1.2]:
        plan += [("sigma", replace(base, sigma=sg), m) for m in main]
    for ce in [0.0, 0.5] if quick else [0.0, 0.2, 0.5]:
        plan += [("calerror", replace(base, cal_error=ce), m) for m in ("corrected", "proposed")]
    plan += [("noisekind", replace(base, noise_kind="amplitude_damping"), m) for m in ("fedavg", "corrected", "proposed")]
    for beta in [1.0, 100.0] if quick else [0.3, 1.0, 100.0]:
        plan += [("beta", replace(base, beta=beta), m) for m in ("fedavg", "corrected", "proposed")]
    return plan


def _job(args: tuple[FedConfig, str, int]) -> dict:
    cfg, key, seed = args
    torch.set_num_threads(1)
    r = run_federated(cfg, to_aggregator(key), seed)
    return {"cfg": cfg, "key": key, "seed": seed, "result": r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run with tiny sizes (< 1 minute)")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from results/*.csv")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.quick:  # smoke runs never overwrite the published results
        tmp = Path(tempfile.mkdtemp(prefix="fisherfed_quick_"))
        out, fig_dir = tmp / "results", tmp / "figures"
    else:
        out, fig_dir = ROOT / "results", ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    seeds = SEEDS[:1] if args.quick else SEEDS
    if args.figures_only:
        import plots
        runs = pd.read_csv(out / "runs.csv")
        runs["shots"] = runs["shots"].fillna(-1)
        plots.make_all(runs, pd.read_csv(out / "curves.csv"), pd.read_csv(out / "summary.csv"), fig_dir)
        return

    plan = build_plan(args.quick)
    unique: dict[tuple[str, str], tuple[FedConfig, str]] = {}
    for _, cfg, key in plan:
        unique.setdefault((repr(canon(cfg, key)), key), (canon(cfg, key), key))
    jobs = [(cfg, key, s) for (cfg, key) in unique.values() for s in seeds]
    print(f"{len(plan)} plan entries -> {len(jobs)} unique runs", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        done = pool.map(_job, jobs, chunksize=1)
    print(f"runs finished in {time.time() - t0:.0f} s", flush=True)

    lookup = {(repr(d["cfg"]), d["key"], d["seed"]): d for d in done}

    def find(cfg: FedConfig, key: str, seed: int) -> dict:
        return lookup[(repr(canon(cfg, key)), key, seed)]

    rows, curves = [], []
    for exp, cfg, key in plan:
        for s in seeds:
            d = find(cfg, key, s)
            r = d["result"]
            rows.append({"experiment": exp, "method_key": key, "method": r["method"], "seed": s, **asdict(cfg),
                         "alpha": to_aggregator(key).alpha, "corrected": to_aggregator(key).corrected,
                         "final_mse": r["final_mse"], "final_acc": r["final_acc"],
                         "lam_true_mean": float(np.mean(r.get("lam_true", [1.0]))),
                         "lam_true_min": float(np.min(r.get("lam_true", [1.0]))),
                         "lam_hat_mean": float(np.mean(r.get("lam_hat", [1.0]))),
                         "shots_total": r.get("shots_total", 0), "calibration_shots": r.get("calibration_shots", 0)})
            for rd, mse, acc in r["curve"]:
                curves.append({"experiment": exp, "method_key": key, "seed": s, "shots": cfg.shots,
                               "sigma": cfg.sigma, "round": rd, "test_mse": mse, "test_acc": acc})
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)

    group = ["experiment", "method_key", "method", "shots", "sigma", "p_med", "cal_error", "noise_kind", "beta"]
    runs["shots"] = runs["shots"].fillna(-1)
    summary = (runs.groupby(group, dropna=False)
               .agg(mse_mean=("final_mse", "mean"), mse_std=("final_mse", "std"),
                    acc_mean=("final_acc", "mean"), acc_std=("final_acc", "std"),
                    lam_mean=("lam_true_mean", "mean"), n_seeds=("seed", "nunique")).reset_index())
    summary.to_csv(out / "summary.csv", index=False)

    paired = paired_tests(runs)
    (out / "paired_tests.json").write_text(json.dumps(paired, indent=2))
    meta = {"seeds": seeds, "rounds": plan[0][1].rounds, "base_config": asdict(BASE), "wall_clock_s": time.time() - t0,
            "torch": torch.__version__, "quick": args.quick}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    import plots  # noqa: E402  (needs sys.path set above)
    plots.make_all(runs, pd.DataFrame(curves), summary, fig_dir)
    print(f"results in {out}, figures in {fig_dir}", flush=True)


def paired_tests(runs: pd.DataFrame) -> list[dict]:
    """Seed-paired differences in final test loss against FedAvg for every shared configuration."""
    out = []
    cfg_cols = ["experiment", "shots", "sigma", "p_med", "cal_error", "noise_kind", "beta"]
    for cfg_vals, g in runs.groupby(cfg_cols, dropna=False):
        base = g[g.method_key == "fedavg"].set_index("seed")["final_mse"]
        if base.empty:
            continue
        for key, h in g[g.method_key != "fedavg"].groupby("method_key"):
            m = h.set_index("seed")["final_mse"]
            common = base.index.intersection(m.index)
            if len(common) < 2:
                continue
            d = (m.loc[common] - base.loc[common]).to_numpy()
            p = float(stats.ttest_rel(m.loc[common], base.loc[common]).pvalue) if np.std(d) > 0 else float("nan")
            out.append({**dict(zip(cfg_cols, [None if (isinstance(v, float) and np.isnan(v)) else v for v in cfg_vals])),
                        "method_key": key, "delta_mse_mean": float(d.mean()), "delta_mse_std": float(d.std(ddof=1)),
                        "seeds_improved": int((d < 0).sum()), "n_seeds": int(len(d)), "paired_t_p": p})
    return out


if __name__ == "__main__":
    main()
