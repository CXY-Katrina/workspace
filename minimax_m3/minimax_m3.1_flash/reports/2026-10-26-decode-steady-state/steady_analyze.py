import argparse
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
from ais_bench.benchmark.calculators.stable_perf_metric_calculator import StablePerfMetricCalculator

ROOT = Path("/work/decode-steady-reanalysis-20261026")
SESSIONS = {
    "old-single-d": Path("/work/pd-pure-perf-20261026-0022"),
    "new-dp4": Path("/work/decode-dp4-tp4-ep16-20261026"),
}


def analyze_run(run, group):
    original = json.loads((run / "result.json").read_text())
    params = original.get("parameters", {})
    if params.get("role") != "decode" or not original.get("validity", {}).get("data_valid"):
        return None
    detail = sorted((run / "aisbench/outputs").glob("*/performances/*/gsm8k_details.jsonl"))[-1]
    rows = [json.loads(line) for line in detail.open() if line.strip()]
    assert len(rows) == params["data_num"] and all(
        row["success"] and row["output_tokens"] == params["output_len"] for row in rows
    )
    database = {}
    times = []
    for row in rows:
        dbname = row["db_name"]
        if dbname not in database:
            db = next(detail.parent.rglob(dbname))
            connection = sqlite3.connect(db)
            try:
                database[dbname] = {
                    key: np.load(io.BytesIO(blob), allow_pickle=False)
                    for key, blob in connection.execute("SELECT id,arr_blob FROM numpy_store")
                }
            finally:
                connection.close()
        times.append(database[dbname][row["time_points"]["__db_ref__"]])
    starts = [float(a[0]) for a in times]
    ends = [float(a[-1]) for a in times]
    maximum = int(original["requests"]["max_concurrency"])
    calculator = StablePerfMetricCalculator()
    calculator.max_concurrency = maximum
    ids = calculator._get_requests_id({"id": list(range(len(rows))), "start_time": starts, "end_time": ends})
    left, right = map(float, calculator.stage_section)
    duration = right - left
    assert duration > 0
    values = {
        "tpot_ms": [(times[i][-1] - times[i][1]) * 1000 / (rows[i]["output_tokens"] - 1) for i in ids],
        "ttft_s": [times[i][1] - times[i][0] for i in ids],
        "e2el_s": [times[i][-1] - times[i][0] for i in ids],
    }
    metrics = {}
    for name, arr in values.items():
        prefix, unit = name.rsplit("_", 1)
        for stat, number in [("avg", np.mean(arr)), ("p90", np.percentile(arr, 90)), ("p99", np.percentile(arr, 99))]:
            metrics[prefix + "_" + stat + "_" + unit] = float(number)
    metrics["output_token_throughput"] = sum(rows[i]["output_tokens"] for i in ids) / duration
    metrics["qps"] = len(ids) / duration
    metrics["benchmark_duration_s"] = duration
    events = sorted([(starts[i], 1) for i in range(len(rows))] + [(ends[i], -1) for i in range(len(rows))])
    active = 0
    last = events[0][0]
    area = 0.0
    full_time = 0.0
    minimum = maximum
    for when, delta in events:
        overlap = max(0.0, min(when, right) - max(last, left))
        if overlap:
            area += active * overlap
            full_time += overlap * (active == maximum)
            minimum = min(minimum, active)
        active += delta
        last = when
    metrics["client_concurrency_time_weighted"] = area / duration
    result = {
        "run_id": run.name,
        "group": group,
        "method": "AISBench StablePerfMetricCalculator (request-start cohort)",
        "parameters": params,
        "source": {"run_path": str(run), "measured_outputs": str(detail.parents[2]), "original_result": original},
        "window": {
            "absolute_start_monotonic_s": left,
            "absolute_end_monotonic_s": right,
            "relative_start_s": left - min(starts),
            "relative_end_s": right - min(starts),
            "duration_s": duration,
            "max_concurrency": maximum,
            "time_at_max_concurrency_fraction": full_time / duration,
            "min_concurrency": minimum,
            "selected_request_count": len(ids),
            "original_request_count": len(rows),
        },
        "selected_requests": [
            {
                "row_index": i,
                "id": rows[i]["id"],
                "uuid": rows[i].get("uuid"),
                "start_relative_s": starts[i] - min(starts),
                "end_relative_s": ends[i] - min(starts),
            }
            for i in ids
        ],
        "metrics": metrics,
        "validity": {
            "data_valid": True,
            "slo_pass": {
                str(x): metrics["tpot_avg_ms"] <= x + 1
                for x in ([20, 50] if params["input_len"] == 16384 else [30, 50])
            },
            "tolerance_ms": 1,
        },
        "interpretation": "AISBench stable-stage counts full output tokens of requests starting in the selected interval and divides by interval duration; request latency retains the full lifetime. It is not an exact token-event time slice or a guarantee of a constant server decode batch.",
    }
    out = ROOT / group / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "steady-result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group")
    parser.add_argument("--run")
    args = parser.parse_args()
    results = []
    for group, root in SESSIONS.items():
        if args.group and group != args.group:
            continue
        for run in sorted((root / "runs").iterdir()):
            if args.run and run.name != args.run:
                continue
            if not (run / "result.json").exists():
                continue
            result = analyze_run(run, group)
            if result:
                results.append(result)
                print(
                    json.dumps(
                        {
                            "group": group,
                            "run": run.name,
                            "cc": result["parameters"]["concurrency"],
                            "window": result["window"],
                            "metrics": result["metrics"],
                        }
                    ),
                    flush=True,
                )
    if not args.run:
        (ROOT / "all-steady-results.json").write_text(json.dumps(results, indent=2))
