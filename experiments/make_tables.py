"""Write results/tables.md (markdown tables used in the README) from results/*.csv and paired_tests.json."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
s = pd.read_csv(R / "summary.csv")
p = pd.DataFrame(json.load(open(R / "paired_tests.json")))
runs = pd.read_csv(R / "runs.csv")
f = lambda m, sd: f"{m:.3f} ± {sd:.3f}"  # noqa: E731
out = []


def table(df: pd.DataFrame, lead: str, lead_fmt) -> None:
    out.append(f"| {lead} | Method | Test MSE | Test accuracy |\n|---|---|---|---|")
    for _, r in df.iterrows():
        out.append(f"| {lead_fmt(r)} | {r.method} | {f(r.mse_mean, r.mse_std)} | {f(r.acc_mean, r.acc_std)} |")
    out.append("")


out.append("## Shots sweep (sigma=1.2, beta=1)")
d = s[(s.experiment == "shots")].copy()
d["o"] = d.shots.replace(-1, 1e9)
table(d.sort_values(["o", "method_key"]), "Shots", lambda r: "exact" if r.shots == -1 else f"{int(r.shots)}")
for e, col, name in [("sigma", "sigma", "Noise heterogeneity sigma"), ("calerror", "cal_error", "Injected calibration error"),
                     ("beta", "beta", "Dirichlet beta")]:
    out.append(f"## {name}")
    table(s[s.experiment == e].sort_values([col, "method_key"]), name, lambda r, c=col: f"{r[c]:g}")
out.append("## Amplitude damping instead of depolarising")
table(s[s.experiment == "noisekind"], "Noise", lambda r: r.noise_kind)
out.append("## Weight exponent")
a = runs[runs.experiment == "alpha"].groupby(["corrected", "alpha"]).final_mse.agg(["mean", "std"]).reset_index()
out.append("| Corrected | alpha | Test MSE |\n|---|---|---|")
for _, r in a.iterrows():
    out.append(f"| {bool(r.corrected)} | {r.alpha:g} | {f(r['mean'], r['std'])} |")
out.append("\n## Paired differences vs FedAvg (negative = better)")
out.append("| Experiment | Setting | Method | Δ MSE (mean ± std) | Seeds improved | Paired t p |\n|---|---|---|---|---|---|")
for _, r in p.iterrows():
    if r.method_key in ("classical",):
        continue
    setting = f"S={'exact' if r.shots == -1 else int(r.shots)}, sigma={r.sigma:g}, cal_err={r.cal_error:g}, beta={r.beta:g}, {r.noise_kind}"
    out.append(f"| {r.experiment} | {setting} | {r.method_key} | {r.delta_mse_mean:+.3f} ± {r.delta_mse_std:.3f} | {r.seeds_improved}/{r.n_seeds} | {r.paired_t_p:.2f} |")
q = runs[(runs.experiment == "shots") & (runs.method_key == "proposed") & (runs.shots == 8)]
out.append(f"\nShot cost at S=8 (per run, mean): total {q.shots_total.mean():.0f}, of which calibration {q.calibration_shots.mean():.0f} "
           f"({100 * q.calibration_shots.mean() / q.shots_total.mean():.1f} %). Mean true attenuation {q.lam_true_mean.mean():.3f}, "
           f"mean minimum client attenuation {q.lam_true_min.mean():.3f}.")
(R / "tables.md").write_text("\n".join(out), encoding="utf-8")
