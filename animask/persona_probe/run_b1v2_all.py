"""Run the five-label decision-point labelling (b1v2) for every pack that
has a v1 b1_decisions file for the given run tags.

One subprocess per pack (each resets its call statistics), N packs at a
time, resumable: packs whose b1v2_decisions file is complete are skipped.

Usage (from the repository root):
    python3 -m animask.persona_probe.run_b1v2_all <tag> [<tag> ...] [--parallel N]
"""
import argparse
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BW = ROOT / "results" / "runs"
LOG = ROOT / "results" / "reports" / "b1v2_run.log"


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def pending(tags):
    jobs = []
    for tag in tags:
        for p in sorted(BW.glob(f"*/b1_decisions_{tag}.json")):
            out = p.parent / f"b1v2_decisions_{tag}.json"
            if out.exists():
                try:
                    if json.loads(out.read_text()).get("complete"):
                        continue
                except ValueError:
                    pass
            jobs.append((p.parent.name, tag))
    return jobs


def run(job):
    pack, tag = job
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "animask.persona_probe.b1v2_decisions", pack, tag],
                       cwd=str(ROOT), capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    log(f"{pack} {tag} rc={r.returncode} {time.time()-t:.0f}s {tail[:160]}"
        + (f" ERR {r.stderr.strip()[-200:]}" if r.returncode else ""))
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="animask.persona_probe.run_b1v2_all",
        description="Five-label decision-point labelling over finished runs.")
    ap.add_argument("tags", nargs="+",
                    help="run tags, e.g. mybatch_gpt55_r1 (one per actor arm)")
    ap.add_argument("--parallel", type=int, default=2,
                    help="packs labelled concurrently (default 2)")
    args = ap.parse_args(argv)
    jobs = pending(args.tags)
    log(f"start: {len(jobs)} pack-runs pending, {args.parallel} in parallel")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        rcs = list(ex.map(run, jobs))
    log(f"done: {sum(1 for c in rcs if c == 0)}/{len(rcs)} ok")


if __name__ == "__main__":
    main()
