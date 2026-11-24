"""Run the bandit tournament: 10 strategies x 6 scenario settings x 200 runs, in parallel.

Run:    .venv/bin/python tournament.py [--runs 200] [--procs 8]
Output: results/summary.json, results/curves.npz, results/leaderboards.md, figures/*.png
Seeds:  every (scenario, strategy) task gets its own stream from numpy SeedSequence(2026) -> reproducible.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bandits as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["eps0.05", "eps0.1", "eps0.2", "decay", "ucb1", "ts", "dts", "etc100", "etc1000", "etc5000"]
COLORS = {"eps0.05": "#9ecae1", "eps0.1": "#4292c6", "eps0.2": "#08519c", "decay": "#6a51a3", "ucb1": "#e6550d",
          "ts": "#31a354", "dts": "#a1d99b", "etc100": "#fdae6b", "etc1000": "#d6616b", "etc5000": "#843c39"}


def task(args):
    scn_key, strat_key, runs, seed = args
    t0 = time.time()
    times, reg, pulls = B.simulate(B.SCENARIOS[scn_key], B.STRATEGIES[strat_key], R=runs, seed=seed)
    return scn_key, strat_key, times, reg, pulls.mean(axis=0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    keys = [(s, k) for s in B.SCENARIOS for k in ORDER]
    seeds = np.random.SeedSequence(2026).spawn(len(keys))
    jobs = [(s, k, a.runs, int(sq.generate_state(1)[0])) for (s, k), sq in zip(keys, seeds)]
    jobs.sort(key=lambda j: -B.SCENARIOS[j[0]]["T"])          # longest first for better load balance
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(task, jobs)
    wall, cpu = time.time() - t0, sum(r[-1] for r in res)
    print("%d tasks, %d runs each: %.1f s wall on %d processes (%.1f s of CPU, %.1fx speed-up)" % (len(jobs), a.runs, wall, a.procs, cpu, cpu / wall))

    summary = {"runs": a.runs, "wall_seconds": wall, "cpu_seconds": cpu, "procs": a.procs, "scenarios": {}}
    curves = {}
    for scn_key, scn in B.SCENARIOS.items():
        rows = []
        for s, k, times, reg, pulls, _ in res:
            if s != scn_key:
                continue
            final = reg[:, -1]
            m, sd = final.mean(), final.std(ddof=1)
            half = 1.96 * sd / np.sqrt(len(final))
            rows.append({"key": k, "name": B.STRATEGIES[k](1, B.n_arms(scn), scn["T"], np.random.default_rng()).name,
                         "mean": float(m), "lo": float(m - half), "hi": float(m + half), "sd": float(sd),
                         "median": float(np.median(final)), "p90": float(np.percentile(final, 90)),
                         "pulls": pulls.round(1).tolist()})
            curves[f"{scn_key}|{k}|t"] = times
            curves[f"{scn_key}|{k}|mean"] = reg.mean(0)
            curves[f"{scn_key}|{k}|half"] = 1.96 * reg.std(0, ddof=1) / np.sqrt(reg.shape[0])
        rows.sort(key=lambda r: r["mean"])
        best_hi = rows[0]["hi"]
        for i, r in enumerate(rows):
            r["rank"] = i + 1
            r["tied_with_best"] = bool(r["lo"] <= best_hi)          # 95% CIs overlap the leader's
        summary["scenarios"][scn_key] = {"title": scn["title"], "desc": scn["desc"], "T": scn["T"],
                                         "p": None if scn["p"] is None else scn["p"].round(4).tolist(), "leaderboard": rows}
    with open(os.path.join(HERE, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(HERE, "results", "curves.npz"), **curves)
    write_leaderboards(summary)
    plot_all(summary, curves)
    print("wrote results/ and figures/")


def write_leaderboards(summary):
    lines = ["# Leaderboards", "", f"{summary['runs']} runs per strategy per scenario. Final expected regret (lost conversions vs always "
             "pulling the best arm), mean with 95% CI. ★ = 95% CI overlaps the leader's.", ""]
    for key, s in summary["scenarios"].items():
        lines += [f"## {s['title']} (T = {s['T']:,})", "", s["desc"] + ".", "",
                  "| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |", "|---|---|---|---|---|---|"]
        for r in s["leaderboard"]:
            lines.append(f"| {r['rank']}{' ★' if r['tied_with_best'] else ''} | {r['name']} | {r['mean']:.1f} | [{r['lo']:.1f}, {r['hi']:.1f}] | {r['median']:.1f} | {r['p90']:.1f} |")
        lines.append("")
    with open(os.path.join(HERE, "results", "leaderboards.md"), "w") as f:
        f.write("\n".join(lines))


def plot_all(summary, curves):
    plt.rcParams.update({"font.size": 10})
    for key, s in summary["scenarios"].items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in sorted(s["leaderboard"], key=lambda r: ORDER.index(r["key"])):
            k = r["key"]
            t, m, h = curves[f"{key}|{k}|t"], curves[f"{key}|{k}|mean"], curves[f"{key}|{k}|half"]
            lw = 2.4 if r["rank"] <= 3 else 1.2
            ax.plot(t, m, color=COLORS[k], lw=lw, label=f"{r['name']} ({r['mean']:.0f})")
            ax.fill_between(t, m - h, m + h, color=COLORS[k], alpha=0.18, lw=0)
        ax.set_title(f"{s['title']}: {s['desc']}", fontsize=10.5)
        ax.set_xlabel("visitors (time step)")
        ax.set_ylabel("cumulative expected regret (lost conversions)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
        top = max(r["hi"] for r in s["leaderboard"])
        ax.set_ylim(0, top * 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", f"regret_{key}.png"), dpi=150)
        plt.close(fig)
    # overview: rank heatmap
    scn_keys = list(summary["scenarios"])
    M = np.array([[next(r["rank"] for r in summary["scenarios"][s]["leaderboard"] if r["key"] == k) for s in scn_keys] for k in ORDER])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=len(ORDER), aspect="auto")
    names = [next(r["name"] for r in summary["scenarios"][scn_keys[0]]["leaderboard"] if r["key"] == k) for k in ORDER]
    ax.set_yticks(range(len(ORDER)), names)
    ax.set_xticks(range(len(scn_keys)), [summary["scenarios"][s]["title"] for s in scn_keys], rotation=20, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9)
    ax.set_title("Rank by scenario (1 = lowest regret)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "ranks.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
