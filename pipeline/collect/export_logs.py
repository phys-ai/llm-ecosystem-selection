#!/usr/bin/env python3
"""Export the raw logs for publication: drop the Reddit excerpt fields that the collector copies into every content row
(the prompts themselves stay) and gzip each timestep file.  Resumable; one archive per run directory.

    python3 pipeline/collect/export_logs.py [--runs runs] [--out ../runs_public] [--seconds 0] [--tar]
    python3 pipeline/collect/export_logs.py --archive ../runs_testbed_tool.tar.gz --out ../runs_public/runs_testbed_tool [--tar]

The second form streams through a testbed archive (whose members are <dir>/run_popseed_<s>.checkpoints/*.json) without
extracting it; --match restricts it to members containing a substring (to run several seeds in parallel).
"""
from __future__ import annotations

import argparse, gzip, json, os, re, tarfile, time
from pathlib import Path

DROP = {"selftext_excerpt", "top_comments_excerpt", "raw_topic_text", "normalized_from", "topic_title", "title"}
LOCAL_PATH = re.compile(r"^/(home|Users|sessions|private|tmp)/")   # machine-specific paths in config fields -> basename


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in DROP}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    if isinstance(obj, str) and LOCAL_PATH.match(obj):
        return os.path.basename(obj.rstrip("/"))
    return obj


def export_archive(a, out: Path, t0: float):
    n = 0; done_runs = set()
    with tarfile.open(a.archive, "r|gz") as tar:           # streaming: one pass, no seeking
        for m in tar:
            if not m.isfile() or a.match not in m.name:
                continue
            parts = Path(m.name).parts                     # <dir>/run_popseed_<s>.checkpoints/<file>.json
            if len(parts) < 3 or not parts[-2].endswith(".checkpoints") or not parts[-1].endswith(".json"):
                continue
            dst = out / parts[-2]; dst.mkdir(parents=True, exist_ok=True)
            if parts[-1] == "metadata.json":
                g = dst / "metadata.json"
                if a.refresh or not g.exists():
                    json.dump(strip(json.load(tar.extractfile(m))), open(g, "w", encoding="utf-8"), ensure_ascii=False); n += 1
                continue
            if not parts[-1].startswith("timestep_"):
                continue
            g = dst / (parts[-1] + ".gz")
            if g.exists():
                continue
            if a.seconds and time.time() - t0 > a.seconds:
                print(f"paused after {n} files ({m.name}); rerun to resume"); return
            tmp = g.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as h:
                json.dump(strip(json.load(tar.extractfile(m))), h, ensure_ascii=False)
            os.replace(tmp, g); n += 1; done_runs.add(dst)
    if a.tar:
        for dst in sorted(p for p in out.iterdir() if p.is_dir()):
            tf = out / (dst.name + ".tar.gz")
            if a.refresh or not tf.exists():
                with tarfile.open(tf, "w:gz") as t2:
                    t2.add(dst, arcname=f"{out.name}/{dst.name}")
                print(f"wrote {tf} ({tf.stat().st_size/1e6:.0f} MB)")
    print(f"done: {n} files converted in {time.time()-t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="../runs_public")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run to the end); rerun to resume")
    ap.add_argument("--match", default="", help="only run directories whose name contains this")
    ap.add_argument("--refresh", action="store_true", help="rewrite metadata.json / summary.json and existing archives")
    ap.add_argument("--tar", action="store_true", help="also write <out>/<run>.tar.gz for every finished run directory")
    ap.add_argument("--archive", default="", help="process the members of this .tar.gz instead of --runs")
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); n = 0
    if a.archive:
        return export_archive(a, out, t0)
    runs = sorted(p.parent for p in Path(a.runs).rglob("metadata.json") if a.match in str(p.parent) and any(p.parent.glob("timestep_*.json")))
    for run in runs:                                       # every directory holding metadata.json + timestep_*.json
        dst = out / run.relative_to(a.runs); dst.mkdir(parents=True, exist_ok=True)
        for small in run.glob("*.json"):                   # metadata.json, summary.json, ...: stripped, not compressed
            if not small.name.startswith("timestep_") and (a.refresh or not (dst / small.name).exists()):
                json.dump(strip(json.load(open(small, encoding="utf-8"))), open(dst / small.name, "w", encoding="utf-8"), ensure_ascii=False)
        for f in sorted(run.glob("timestep_*.json")):
            g = dst / (f.name + ".gz")
            if g.exists():
                continue
            if a.seconds and time.time() - t0 > a.seconds:
                print(f"paused after {n} files ({run.name}); rerun to resume"); return
            tmp = g.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as h:
                json.dump(strip(json.load(open(f, encoding="utf-8"))), h, ensure_ascii=False)
            os.replace(tmp, g); n += 1
        if a.tar and dst.parent == out and len(list(dst.glob("timestep_*.json.gz"))) == len(list(run.glob("timestep_*.json"))):
            tf = out / (run.name + ".tar.gz")             # one archive per top-level run directory
            if a.refresh or not tf.exists():
                with tarfile.open(tf, "w:gz") as tar:
                    tar.add(dst, arcname=run.name)
                print(f"wrote {tf} ({tf.stat().st_size/1e6:.0f} MB)")
    print(f"done: {n} files converted in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
