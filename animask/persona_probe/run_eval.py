"""One-command evaluate driver for a finished (or partially finished)
batch — the v2 chain:

  per run:  story-curve questionnaire (sim side)  +  decision-point labels
  per sid:  canon-side story curves
  batch:    A1 facet profile -> t0 leak audit -> profile (gate) ->
            b1 aggregate -> curve_compare (dispersion side-by-side +
            paired deviations; no ratio)

    python3 -m animask.persona_probe.run_eval <batchtag> [--judge M] [--no-b1]
        [--no-curves] [--legacy]

--legacy additionally runs the superseded endpoint chain (C3 outcome
cards + C0) — kept only as a cross-check for the curve endpoints.
Every step is artifact-gated (a finished sub-result is never recomputed),
so the command is safe to re-run as more books finish simulating. Per-book
failures are reported and skipped, never fatal.
"""
import argparse
import sys
import traceback
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import (b1_decisions, c0_convergence, c3_scoreboard,
                           curve_compare, facet_profile, leak_audit,
                           story_curves)


def _reset_call_stats():
    for k in pp.CALL_STATS:
        pp.CALL_STATS[k] = 0


def discover_done_runs(batchtag: str):
    """[(pack, tag)] for finished simulations matching the batch tag."""
    base = pp.get_root() / "results" / "runs"
    out = []
    for meta_p in sorted(base.glob("*/run_meta_*.json")):
        tag = meta_p.name[len("run_meta_"):-len(".json")]
        if not (tag == batchtag or tag.startswith(batchtag + "_")):
            continue
        try:
            if pp.load_json(meta_p).get("status") == "done":
                out.append((meta_p.parent.name, tag))
        except (OSError, ValueError):
            continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.run_eval")
    ap.add_argument("batchtag")
    ap.add_argument("--judge", default=pp.DEEPSEEK)
    ap.add_argument("--no-b1", action="store_true")
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="also run superseded C3/C0 endpoint chain")
    args = ap.parse_args(argv)

    runs = discover_done_runs(args.batchtag)
    print(f"[run_eval] {len(runs)} finished runs for {args.batchtag}")
    failures = []

    def step(label, fn):
        _reset_call_stats()
        try:
            fn()
            return True
        except SystemExit as e:
            failures.append(f"{label}: {e}")
        except Exception as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")
            traceback.print_exc()
        return False

    for pack, tag in runs:
        if not args.no_b1:
            out_p = pp.run_dir(pack) / f"b1_decisions_{tag}.json"
            done = False
            if out_p.exists():
                try:
                    done = pp.load_json(out_p).get("complete", False)
                except (OSError, ValueError):
                    done = False
            if not done:
                step(f"b1 {pack}", lambda p=pack, t=tag:
                     b1_decisions.run_b1(p, t, judge=args.judge))
        if not args.no_curves:
            step(f"curves {pack}", lambda p=pack, t=tag:
                 story_curves.run_story_curves(p, t, judge=args.judge))
            sid = pack.split("__")[0]
            step(f"canon curves {sid}", lambda s=sid:
                 story_curves.canon_curves(s, judge=args.judge))
        if args.legacy:
            step(f"c3 {pack}", lambda p=pack, t=tag:
                 c3_scoreboard.run_c3(p, t, judge=args.judge))
            sid = pack.split("__")[0]
            step(f"canon card {sid}", lambda s=sid:
                 c3_scoreboard.canon_card(s, judge=args.judge))

    # A1 profile -> leak audit -> profile again (audit feeds the gate)
    step("facet_profile", lambda: facet_profile.main([args.batchtag]))
    step("leak_audit", lambda: leak_audit.main([args.batchtag,
                                                "--judge", args.judge]))
    step("facet_profile+audit",
         lambda: facet_profile.main([args.batchtag]))

    if not args.no_b1:
        step("b1 aggregate",
             lambda: b1_decisions.main(["--aggregate", args.batchtag]))
    if not args.no_curves:
        step("curve_compare",
             lambda: curve_compare.aggregate(args.batchtag))
    if args.legacy:
        step("c3 aggregate",
             lambda: c3_scoreboard.main(["--aggregate", args.batchtag]))
        step("c0", lambda: c0_convergence.main([args.batchtag]))
    if failures:
        print(f"[run_eval] {len(failures)} failures:")
        for f in failures:
            print("  -", f)
    else:
        print("[run_eval] all steps clean")
    return failures


if __name__ == "__main__":
    main()
