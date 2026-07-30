# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Poll libtpu's RuntimeMetricService and append per-chip duty cycle / HBM rows to a CSV.

Runs inside a TPU pod (copied in via ``kubectl cp``). Uses ``grpcurl`` (present in
the rl-disagg image) against the runtime metric ports, so it works for pods on
custom ``TPU_RUNTIME_METRICS_PORTS`` where ``tpu-info`` cannot reach.

A connection-refused metric port means the process is checkpointed/detached
(libtpu tears the service down while detached); rows are written with
duty 0 / mem 0 so the CSV reflects real 0% utilization by this pod.

Env vars:
  PORTS          comma-separated metric ports to try (default "8431")
  PHASE          run label, e.g. "baseline" or "timeslice" (default "baseline")
  CSV_FILE       output path (default /tmp/tpu_duty_cycle.csv)
  POLL_INTERVAL  seconds between polls (default 1)
  NUM_CHIPS      chips to report as 0 when detached (default 8)

CSV schema matches tpu-rl-jax-poc/telemetry/tpu_duty_cycle.py:
  ts,wall,phase,chip,duty_cycle_pct,mem_used_mib,mem_total_mib,mem_pct
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

PORTS = [p.strip() for p in os.environ.get("PORTS", "8431").split(",") if p.strip()]
PHASE = os.environ.get("PHASE", "baseline")
CSV_FILE = os.environ.get("CSV_FILE", "/tmp/tpu_duty_cycle.csv")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1"))
NUM_CHIPS = int(os.environ.get("NUM_CHIPS", "8"))

FIELDS = [
    "ts",
    "wall",
    "phase",
    "chip",
    "duty_cycle_pct",
    "mem_used_mib",
    "mem_total_mib",
    "mem_pct",
]

DUTY_METRIC = "tpu.runtime.tensorcore.dutycycle.percent"
MEM_USED_METRIC = "tpu.runtime.hbm.memory.usage.bytes"
MEM_TOTAL_METRIC = "tpu.runtime.hbm.memory.total.bytes"

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


def get_metric(port, name):
    """Return {device_id: value} for a metric, or None if the port is down."""
    try:
        out = subprocess.run(
            [
                "grpcurl",
                "-plaintext",
                "-max-time",
                "3",
                "-d",
                json.dumps({"metric_name": name}),
                f"localhost:{port}",
                "tpu.monitoring.runtime.RuntimeMetricService/GetRuntimeMetric",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    try:
        resp = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    values = {}
    for m in resp.get("metric", {}).get("metrics", []):
        device = int(m.get("attribute", {}).get("value", {}).get("int_attr", "0"))
        gauge = m.get("gauge", {})
        if "as_double" in gauge:
            values[device] = float(gauge["as_double"])
        elif "as_int" in gauge:
            values[device] = float(gauge["as_int"])
    return values or None


def poll_once():
    """Query the first responsive port; None means detached / not initialized."""
    for port in PORTS:
        duty = get_metric(port, DUTY_METRIC)
        if duty is None:
            continue
        used = get_metric(port, MEM_USED_METRIC) or {}
        total = get_metric(port, MEM_TOTAL_METRIC) or {}
        return duty, used, total
    return None


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    new_file = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        while not _stop:
            start = time.time()
            ts = round(start, 3)
            wall = datetime.now(timezone.utc).strftime("%H:%M:%S")
            result = poll_once()
            if result is None:
                rows = [
                    {
                        "ts": ts,
                        "wall": wall,
                        "phase": PHASE,
                        "chip": chip,
                        "duty_cycle_pct": 0.0,
                        "mem_used_mib": 0,
                        "mem_total_mib": 0,
                        "mem_pct": 0.0,
                    }
                    for chip in range(NUM_CHIPS)
                ]
            else:
                duty, used, total = result
                rows = []
                for chip in sorted(duty):
                    used_mib = int(used.get(chip, 0) / (1024 * 1024))
                    total_mib = int(total.get(chip, 0) / (1024 * 1024))
                    rows.append(
                        {
                            "ts": ts,
                            "wall": wall,
                            "phase": PHASE,
                            "chip": chip,
                            "duty_cycle_pct": round(duty[chip], 1),
                            "mem_used_mib": used_mib,
                            "mem_total_mib": total_mib,
                            "mem_pct": round(100.0 * used_mib / total_mib, 1)
                            if total_mib
                            else 0.0,
                        }
                    )
            writer.writerows(rows)
            f.flush()
            elapsed = time.time() - start
            time.sleep(max(0.0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    sys.exit(main())
