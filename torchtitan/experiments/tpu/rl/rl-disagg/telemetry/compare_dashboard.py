# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline vs. time-slicing TPU duty-cycle comparison dashboard.

Reads the per-pod CSVs produced by ``tpu_duty_cycle.py`` (real tpu-info /
RuntimeMetricService readings, 1s resolution) and renders a static HTML
dashboard with matplotlib PNGs embedded as base64, following the layout of
tpu-rl-jax-poc/telemetry/scraper_dashboard.py:

  * side-by-side (Baseline | Time-slicing) duty-cycle panels per node
    (sampler/inference node and trainer/training node)
  * per-chip time series
  * summary stats table (avg duty, % time active, avg HBM)

Expected files per run dir:
  baseline dir:   tpu_duty_cycle_sampler_a.csv, tpu_duty_cycle_trainer_a.csv
  timeslice dir:  the same plus tpu_duty_cycle_sampler_b.csv,
                  tpu_duty_cycle_trainer_b.csv

Usage:
  python3 compare_dashboard.py --baseline-dir runs/X/baseline \
      --timeslice-dir runs/X/timeslice --output runs/X/dashboard.html
"""

import argparse
import base64
import csv
import io
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_A = "#4285F4"  # job A
COLOR_B = "#EA4335"  # job B
COLOR_NODE = "#34A853"  # combined node utilization
CHIP_COLORS = [
    "#4285F4",
    "#EA4335",
    "#FBBC05",
    "#34A853",
    "#8E24AA",
    "#F57C00",
    "#0097A7",
    "#546E7A",
]

NODES = [("sampler", "Sampler node (inference)"), ("trainer", "Trainer node (training)")]


def load_pod_csv(path):
    """Return {"ts": [...], "mean_duty": [...], "mean_mem_gib": [...], "chips": {chip: {"ts": [], "duty": []}}}."""
    if not os.path.exists(path):
        return None
    by_ts = defaultdict(list)
    chips = defaultdict(lambda: {"ts": [], "duty": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = float(row["ts"])
            duty = float(row["duty_cycle_pct"])
            mem = float(row["mem_used_mib"])
            by_ts[ts].append((duty, mem))
            chip = int(row["chip"])
            chips[chip]["ts"].append(ts)
            chips[chip]["duty"].append(duty)
    ts_sorted = sorted(by_ts)
    mean_duty = [sum(d for d, _ in by_ts[t]) / len(by_ts[t]) for t in ts_sorted]
    mean_mem = [
        sum(m for _, m in by_ts[t]) / len(by_ts[t]) / 1024.0 for t in ts_sorted
    ]
    return {
        "ts": ts_sorted,
        "mean_duty": mean_duty,
        "mean_mem_gib": mean_mem,
        "chips": chips,
    }


def load_phase(run_dir, jobs):
    """Return {node_role: {job: pod_data}} and the phase t0."""
    data = {}
    t0 = None
    for role, _ in NODES:
        data[role] = {}
        for job in jobs:
            pod = load_pod_csv(
                os.path.join(run_dir, f"tpu_duty_cycle_{role}_{job}.csv")
            )
            if pod and pod["ts"]:
                data[role][job] = pod
                first = pod["ts"][0]
                t0 = first if t0 is None else min(t0, first)
    return data, t0


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def save_png(fig_b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(fig_b64))


def combined_node_series(pods):
    """Element-wise sum of per-job mean duty on a shared 1s grid (only one job
    is attached at a time, so the sum is the node's utilization)."""
    grid = defaultdict(float)
    for pod in pods.values():
        for t, d in zip(pod["ts"], pod["mean_duty"]):
            grid[round(t)] = grid[round(t)] + d
    ts = sorted(grid)
    return ts, [min(grid[t], 100.0) for t in ts]


def plot_node_comparison(baseline, base_t0, timeslice, ts_t0):
    """2x2: rows = node, cols = Baseline | Time-slicing. Mean duty across 8 chips."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 9), sharey=True)
    for i, (role, node_label) in enumerate(NODES):
        # Baseline column
        ax = axes[i][0]
        pod = baseline.get(role, {}).get("a")
        if pod:
            x = [(t - base_t0) / 60 for t in pod["ts"]]
            ax.fill_between(x, pod["mean_duty"], color=COLOR_A, alpha=0.35)
            ax.plot(x, pod["mean_duty"], color=COLOR_A, lw=0.9, label="job A (dedicated)")
        ax.set_title(f"Baseline — {node_label}", fontsize=12)
        # Time-slicing column
        ax = axes[i][1]
        pods = timeslice.get(role, {})
        for job, color in (("a", COLOR_A), ("b", COLOR_B)):
            pod = pods.get(job)
            if pod:
                x = [(t - ts_t0) / 60 for t in pod["ts"]]
                ax.fill_between(x, pod["mean_duty"], color=color, alpha=0.35)
                ax.plot(x, pod["mean_duty"], color=color, lw=0.9, label=f"job {job.upper()}")
        ax.set_title(f"Time-slicing — {node_label}", fontsize=12)
    for ax in axes.flat:
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Elapsed (minutes)")
        ax.set_ylabel("Duty cycle (%)")
    fig.suptitle(
        "TPU duty cycle (mean of 8 chips, 1s tpu-info/RuntimeMetricService samples)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_to_b64(fig)


def plot_node_utilization_overlay(baseline, base_t0, timeslice, ts_t0):
    """1x2 money shot: total node utilization (sum of jobs) per node, baseline vs timeslice."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
    for col, (label, phase, t0) in enumerate(
        [("Baseline", baseline, base_t0), ("Time-slicing", timeslice, ts_t0)]
    ):
        ax = axes[col]
        for (role, node_label), color in zip(NODES, (COLOR_A, COLOR_B)):
            pods = phase.get(role, {})
            if not pods:
                continue
            ts, duty = combined_node_series(pods)
            x = [(t - t0) / 60 for t in ts]
            ax.fill_between(x, duty, color=color, alpha=0.4, label=node_label)
        ax.set_title(f"{label} — total TPU utilization per node", fontsize=12)
        ax.set_xlabel("Elapsed (minutes)")
        ax.set_ylabel("Duty cycle (%)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_per_chip(phase, t0, phase_label):
    """2x1 per-chip duty (all pods of a node on one axes, solid=job A, dashed=job B)."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for i, (role, node_label) in enumerate(NODES):
        ax = axes[i]
        for job, ls in (("a", "-"), ("b", "--")):
            pod = phase.get(role, {}).get(job)
            if not pod:
                continue
            for chip, series in sorted(pod["chips"].items()):
                x = [(t - t0) / 60 for t in series["ts"]]
                ax.plot(
                    x,
                    series["duty"],
                    ls,
                    lw=0.8,
                    alpha=0.7,
                    color=CHIP_COLORS[chip % 8],
                    label=f"job {job.upper()} chip {chip}",
                )
        ax.set_title(f"{phase_label} — {node_label} (per chip)", fontsize=11)
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.set_xlabel("Elapsed (minutes)")
        ax.set_ylabel("Duty cycle (%)")
    fig.tight_layout()
    return fig_to_b64(fig)


CR_EVENT_RE = re.compile(
    r"(\d\d:\d\d:\d\d),\d+ INFO \[(job-[ab])_(sampler|trainer)\] "
    r"(checkpoint|restore) elapsed: (\d+)ms"
)


def load_cr_events(log_path):
    """Parse checkpoint/restore durations from the orchestrator log.

    Returns a list of (wall_hhmmss, job, pool, op, ms) in log order.
    """
    if not log_path or not os.path.exists(log_path):
        return []
    events = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = CR_EVENT_RE.search(line)
            if m:
                events.append(
                    (m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5)))
                )
    return events


def plot_cr_times(events):
    """1x2 bars: per-event checkpoint and restore durations, colored by pool."""
    colors = {"sampler": COLOR_A, "trainer": COLOR_B}
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)
    for ax, op in zip(axes, ("checkpoint", "restore")):
        evs = [(w, pool, ms) for w, _, pool, o, ms in events if o == op]
        xs = range(len(evs))
        ax.bar(
            xs,
            [ms / 1000 for _, _, ms in evs],
            color=[colors[p] for _, p, _ in evs],
            alpha=0.8,
        )
        for pool, c in colors.items():
            vals = [ms / 1000 for _, p, ms in evs if p == pool]
            if vals:
                ax.axhline(
                    sum(vals) / len(vals),
                    color=c,
                    ls="--",
                    lw=1,
                    label=f"{pool} mean {sum(vals) / len(vals):.2f}s",
                )
        ax.set_xticks(list(xs))
        ax.set_xticklabels([w for w, _, _ in evs], rotation=45, fontsize=7)
        ax.set_title(f"{op} duration per event")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def cr_table(events):
    stats = defaultdict(list)
    for _, _, pool, op, ms in events:
        stats[(op, pool)].append(ms)
    for op in ("checkpoint", "restore"):
        combined = [ms for (o, _), v in stats.items() if o == op for ms in v]
        if combined:
            stats[(op, "all pools")] = combined
    rows = []
    for op in ("checkpoint", "restore"):
        for pool in ("sampler", "trainer", "all pools"):
            v = stats.get((op, pool))
            if not v:
                continue
            bold = pool == "all pools"
            b = ("<b>", "</b>") if bold else ("", "")
            rows.append(
                f"<tr><td>{b[0]}{op}{b[1]}</td><td>{b[0]}{pool}{b[1]}</td>"
                f"<td>{len(v)}</td><td>{b[0]}{sum(v) / len(v):.0f} ms{b[1]}</td>"
                f"<td>{min(v)} ms</td><td>{max(v)} ms</td></tr>"
            )
    return (
        "<table><tr><th>Operation</th><th>Pool</th><th>Events</th>"
        "<th>Mean</th><th>Min</th><th>Max</th></tr>" + "".join(rows) + "</table>"
    )


def phase_stats(phase):
    """Per node: avg duty over run, avg duty while active, % time active, avg HBM."""
    stats = {}
    for role, _ in NODES:
        pods = phase.get(role, {})
        if not pods:
            continue
        ts, node_duty = combined_node_series(pods)
        active = [d for d in node_duty if d > 1.0]
        mems = []
        for pod in pods.values():
            mems.extend(m for m in pod["mean_mem_gib"] if m > 0.05)
        stats[role] = {
            "samples": len(node_duty),
            "duration_min": (ts[-1] - ts[0]) / 60 if len(ts) > 1 else 0.0,
            "avg_duty": sum(node_duty) / len(node_duty) if node_duty else 0.0,
            "avg_duty_active": sum(active) / len(active) if active else 0.0,
            "active_pct": 100.0 * len(active) / len(node_duty) if node_duty else 0.0,
            "avg_mem_gib": sum(mems) / len(mems) if mems else 0.0,
        }
    return stats


def summary_table(base_stats, ts_stats):
    rows = []
    metrics = [
        ("Avg node duty cycle (whole run)", "avg_duty", "{:.1f}%"),
        ("Avg duty cycle while active", "avg_duty_active", "{:.1f}%"),
        ("% of time node active (duty > 1%)", "active_pct", "{:.1f}%"),
        ("Avg HBM used per chip (active)", "avg_mem_gib", "{:.1f} GiB"),
        ("Run duration", "duration_min", "{:.1f} min"),
        ("Samples (1s)", "samples", "{:.0f}"),
    ]
    for role, node_label in NODES:
        rows.append(
            f'<tr class="section"><td colspan="3">{node_label}</td></tr>'
        )
        for label, key, fmt in metrics:
            b = base_stats.get(role)
            t = ts_stats.get(role)
            bval = fmt.format(b[key]) if b else "—"
            tval = fmt.format(t[key]) if t else "—"
            rows.append(
                f"<tr><td>{label}</td><td>{bval}</td><td>{tval}</td></tr>"
            )
    return (
        '<table><tr><th>Metric</th><th>Baseline<br>(1 job, dedicated TPUs)</th>'
        "<th>Time-slicing<br>(2 jobs, shared TPUs)</th></tr>"
        + "".join(rows)
        + "</table>"
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TPU Duty Cycle: Baseline vs Time-slicing</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f7; margin: 0; padding: 24px; }}
.container {{ max-width: 1500px; margin: 0 auto; }}
.card {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 24px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; color: #333; }}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; font-size: 14px; min-width: 640px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
th {{ background: #f0f0f0; }}
tr.section td {{ background: #e8f0fe; font-weight: 600; }}
.note {{ color: #666; font-size: 13px; }}
</style></head><body><div class="container">
<div class="card"><h1>TPU Duty Cycle — Baseline vs Time-slicing (rl-disagg GRPO, libtpu v3)</h1>
<p class="note">Real 1-second samples of <code>tpu.runtime.tensorcore.dutycycle.percent</code>
from libtpu's RuntimeMetricService (the same source tpu-info reads); v5e 2x4, Qwen3-0.6B.
Baseline: one GRPO job with a dedicated sampler node and trainer node.
Time-slicing: two GRPO jobs sharing both nodes via libtpu pipe-based checkpoint/restore.
A detached (checkpointed) process reports 0%.</p>
{summary}</div>
<div class="card"><h2>1. Total node utilization — Baseline vs Time-slicing</h2>{overlay}</div>
<div class="card"><h2>2. Per-node duty cycle by job (side by side)</h2>{nodes}</div>
<div class="card"><h2>3. Checkpoint / restore latency (time-slicing run)</h2>
<p class="note">Per-event durations of libtpu pipe-based checkpoint (yield) and restore
(acquire), parsed from the orchestrator log; 8 ranks per pool, DMA pipeline,
32 MiB chunks. Trainer C/R carries more live state (optimizer + gradients)
than sampler C/R (weights + KV cache).</p>
{cr}</div>
<div class="card"><h2>4. Per-chip detail — Baseline</h2>{chips_base}</div>
<div class="card"><h2>5. Per-chip detail — Time-slicing</h2>{chips_ts}</div>
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--timeslice-dir", required=True)
    ap.add_argument("--output", default="dashboard.html")
    ap.add_argument(
        "--orchestrator-log",
        default=None,
        help="orchestrator log with C/R events "
        "(default: <timeslice-dir>/orchestrator.log)",
    )
    args = ap.parse_args()

    baseline, base_t0 = load_phase(args.baseline_dir, jobs=("a",))
    timeslice, ts_t0 = load_phase(args.timeslice_dir, jobs=("a", "b"))
    if base_t0 is None or ts_t0 is None:
        raise SystemExit("missing CSVs in one of the run dirs")

    overlay = plot_node_utilization_overlay(baseline, base_t0, timeslice, ts_t0)
    nodes = plot_node_comparison(baseline, base_t0, timeslice, ts_t0)
    chips_base = plot_per_chip(baseline, base_t0, "Baseline")
    chips_ts = plot_per_chip(timeslice, ts_t0, "Time-slicing")

    orch_log = args.orchestrator_log or os.path.join(
        args.timeslice_dir, "orchestrator.log"
    )
    cr_events = load_cr_events(orch_log)
    if cr_events:
        cr_fig = plot_cr_times(cr_events)
        cr_html = cr_table(cr_events) + f'<img src="data:image/png;base64,{cr_fig}">'
    else:
        print(f"warning: no C/R events found in {orch_log}; skipping that section")
        cr_html = '<p class="note">No checkpoint/restore events found.</p>'

    out_dir = os.path.dirname(os.path.abspath(args.output))
    save_png(overlay, os.path.join(out_dir, "duty_overlay.png"))
    save_png(nodes, os.path.join(out_dir, "duty_by_node.png"))
    if cr_events:
        save_png(cr_fig, os.path.join(out_dir, "cr_times.png"))

    html = HTML_TEMPLATE.format(
        summary=summary_table(phase_stats(baseline), phase_stats(timeslice)),
        overlay=f'<img src="data:image/png;base64,{overlay}">',
        nodes=f'<img src="data:image/png;base64,{nodes}">',
        cr=cr_html,
        chips_base=f'<img src="data:image/png;base64,{chips_base}">',
        chips_ts=f'<img src="data:image/png;base64,{chips_ts}">',
    )
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output}")

    for label, stats in (("baseline", phase_stats(baseline)), ("timeslice", phase_stats(timeslice))):
        for role, s in stats.items():
            print(
                f"{label:9s} {role:8s} avg_duty={s['avg_duty']:5.1f}% "
                f"active={s['active_pct']:5.1f}% avg_active_duty={s['avg_duty_active']:5.1f}% "
                f"hbm={s['avg_mem_gib']:.1f}GiB dur={s['duration_min']:.1f}min"
            )


if __name__ == "__main__":
    main()
