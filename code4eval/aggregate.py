#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate <results_root>/scores/<model>__<mode>/run*_<judge>/eval/accuracy.json into mean ± std per (model, mode, judge).
Usage: python code4eval/aggregate.py [--results_root results] [--out <results_root>/summary]
Writes <out>.json and <out>.md. std is the sample standard deviation over the runs."""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out", default=None, help="output prefix, default <results_root>/summary")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.results_root, "summary")

    runs = defaultdict(list)   # (model, mode, judge) -> [accuracy dict]
    for acc in sorted(glob.glob(os.path.join(args.results_root, "scores", "*__*", "run*_*", "eval", "accuracy.json"))):
        parts = Path(acc).parts
        model, mode = parts[-4].split("__", 1)
        m = re.match(r"run(\d+)_(.+)", parts[-3])
        judge = m.group(2) if m else "?"
        d = json.load(open(acc, encoding="utf-8"))
        d["_run"] = int(m.group(1)) if m else -1
        runs[(model, mode, judge)].append(d)

    metrics = ["CSR", "ISR"]
    breakdown_keys = ["by_constraint_presentation_type", "by_constraint_type"]
    summary = {}
    lines = ["| model | mode | judge | n_runs | CSR mean±std | ISR mean±std | per-run CSR | per-run ISR | samples (ISR total) | pred-null instr | judge-null % |",
             "|---|---|---|---:|---|---|---|---|---|---|---|"]
    for (model, mode, judge), ds in sorted(runs.items()):
        ds.sort(key=lambda x: x["_run"])
        row = {"n_runs": len(ds), "runs": [d["_run"] for d in ds]}
        cells = []
        for k in metrics:
            vals = [d.get(k, {}).get("accuracy") for d in ds]
            m, sd, n = mean_std(vals)
            row[k] = {"mean": m, "std": sd, "values": vals}
            cells.append(f"{m*100:.2f} ± {sd*100:.2f}" if m is not None else "-")
        for bk in breakdown_keys:
            dims = defaultdict(list)
            for d in ds:
                for dim, v in (d.get(bk) or d.get(bk + " (dimension)") or {}).items():   # tolerate the key name of older result files
                    if isinstance(v, dict) and "accuracy" in v:
                        dims[dim].append(v["accuracy"])
            row[bk] = {dim: dict(zip(("mean", "std", "n"), mean_std(v))) for dim, v in dims.items()}
        row["samples"] = [d.get("ISR", {}).get("total") for d in ds]   # instructions with at least one scored constraint, per run
        fs = [d.get("failure_stats") or {} for d in ds]
        row["prediction_null_instructions"] = [(f.get("prediction_null") or {}).get("null_instructions") for f in fs]
        row["judge_null_rate"] = [(f.get("judge_null") or {}).get("rate_over_judged_constraints") for f in fs]
        summary[f"{model}__{mode}__{judge}"] = row
        per_csr = ", ".join(f"{(d.get('CSR', {}).get('accuracy') or 0)*100:.2f}" for d in ds)
        per_isr = ", ".join(f"{(d.get('ISR', {}).get('accuracy') or 0)*100:.2f}" for d in ds)
        pn = ", ".join("-" if x is None else str(x) for x in row["prediction_null_instructions"])
        jn = ", ".join("-" if x is None else f"{x*100:.2f}" for x in row["judge_null_rate"])
        lines.append(f"| {model} | {mode} | {judge} | {len(ds)} | {cells[0]} | {cells[1]} | {per_csr} | {per_isr} | {row['samples']} | {pn} | {jn} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {args.out}.json / {args.out}.md")


if __name__ == "__main__":
    main()
