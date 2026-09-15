"""Cross-run aggregation of persona_probe outputs.

    python3 -m animask.persona_probe.aggregate <batchtag> [--all]
    python3 -m animask.persona_probe.aggregate <batchtag> --runs pack:tag pack:tag ...

Collects results/runs/<pack>/persona_probe_<tag>.json files, groups by
pack, pools per-agent trajectories across runs (cluster = run, matching the
clustered-bootstrap protocol), applies the interpretation rules
at the aggregate level, and writes
results/persona_probe/aggregate_<batchtag>.json + .md.
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import stats
from animask.persona_probe.run_persona_probe import (AFC_DISTINGUISH, AFC_INERT,
                                             AFC_MIN_VOTES_DISTINGUISH,
                                             AFC_MIN_VOTES_INERT,
                                             AFC_INERT_MAX_GAP, L1_DECLINE_MIN,
                                             MK_ALPHA)


def discover_runs():
    """[(pack, tag, path)] for every final probe output on disk."""
    base = pp.get_root() / "results" / "runs"
    out = []
    if not base.exists():
        return out
    for path in sorted(base.glob("*/persona_probe_*.json")):
        name = path.name
        if name.endswith((".l1.json", ".l2.json", ".l3.json")):
            continue  # layer partials
        tag = name[len("persona_probe_"):-len(".json")]
        out.append((path.parent.name, tag, path))
    return out


def _layer_ok(layer):
    return isinstance(layer, dict) and "skipped" not in layer


def _run_declining(agent_l1_stats):
    st = agent_l1_stats
    return (st["n"] >= 3 and st["mk"]["p"] < MK_ALPHA
            and st["total_change"] is not None
            and st["total_change"] <= L1_DECLINE_MIN)


def aggregate_runs(runs, seed=0):
    """runs: [(pack, tag, loaded_result)] -> aggregate dict."""
    packs = {}
    for pack, tag, res in runs:
        packs.setdefault(pack, []).append((tag, res))
    agg = {"packs": {}, "n_runs": len(runs), "generated_at": pp.now_str()}
    for pack, items in sorted(packs.items()):
        agents = {}
        run_verdicts = {}
        degradations = set()
        for tag, res in items:
            run_verdicts[tag] = res.get("verdict", {}).get("run_verdict")
            degradations |= set(res.get("verdict", {}).get("degradations",
                                                           []))
            l1, l2, l3 = res.get("l1"), res.get("l2"), res.get("l3")
            for role in res.get("roles", []):
                a = agents.setdefault(role, {
                    "l1_clusters": [], "l1_slopes": [], "l1_declining": 0,
                    "l1_runs": 0, "l2_means": [], "l2_slopes": [],
                    "l3_persona_votes": 0, "l3_votes": 0, "l3_gaps": [],
                    "verdicts": {}})
                a["verdicts"][tag] = res.get("verdict", {}).get(
                    "per_agent", {}).get(role, {}).get("verdict")
                if _layer_ok(l1) and role in l1.get("per_agent", {}):
                    ag = l1["per_agent"][role]
                    scores = [s["score"] for s in ag["series"]
                              if s["score"] is not None]
                    if scores:
                        a["l1_clusters"].append(scores)
                    a["l1_runs"] += 1
                    if ag["stats"]["theil_sen_slope"] is not None:
                        a["l1_slopes"].append(ag["stats"]["theil_sen_slope"])
                    if _run_declining(ag["stats"]):
                        a["l1_declining"] += 1
                if _layer_ok(l2) and role in l2.get("per_agent", {}):
                    st = l2["per_agent"][role]["stats"]["normative"]
                    if st["mean"] is not None:
                        a["l2_means"].append(st["mean"])
                    if st["theil_sen_slope"] is not None:
                        a["l2_slopes"].append(st["theil_sen_slope"])
                if _layer_ok(l3) and role in l3.get("per_agent", {}):
                    ag3 = l3["per_agent"][role]
                    a["l3_persona_votes"] += ag3.get("persona_votes", 0)
                    a["l3_votes"] += ag3.get("votes", 0)
                    if ag3.get("mean_drift_gap") is not None:
                        a["l3_gaps"].append(ag3["mean_drift_gap"])

        per_agent = {}
        for role, a in sorted(agents.items()):
            rate = (a["l3_persona_votes"] / a["l3_votes"]
                    if a["l3_votes"] else None)
            # the accumulator key is l3_gaps; the mean must be computed here —
            # reading a["l3_mean_drift_gap"] (an output-dict key) was always
            # None and made the aggregate inert verdict unreachable
            mean_gap = stats.mean(a["l3_gaps"]) if a["l3_gaps"] else None
            declining = (a["l1_runs"] > 0
                         and a["l1_declining"] * 2 >= a["l1_runs"])
            distinguishable = (rate is not None
                               and a["l3_votes"] >= AFC_MIN_VOTES_DISTINGUISH
                               and rate >= AFC_DISTINGUISH)
            inert = (rate is not None
                     and a["l3_votes"] >= AFC_MIN_VOTES_INERT
                     and mean_gap is not None
                     and mean_gap <= AFC_INERT_MAX_GAP
                     and rate <= AFC_INERT)
            if inert:
                verdict = "persona_inert"
            elif declining:
                verdict = "drift_associated"
            elif a["l1_runs"] and distinguishable:
                verdict = "persona_stable_model_trend"
            elif a["l1_runs"]:
                verdict = "persona_stable_attribution_pending"
            else:
                verdict = "insufficient_evidence"
            per_agent[role] = {
                "n_runs": a["l1_runs"],
                "l1_mean_ci": stats.bootstrap_ci_clustered(a["l1_clusters"],
                                                           seed=seed),
                "l1_slope_median": stats.median(a["l1_slopes"]),
                "l1_declining_runs": a["l1_declining"],
                "l2_normative_mean": stats.mean(a["l2_means"]),
                "l2_slope_median": stats.median(a["l2_slopes"]),
                "l3_persona_rate": rate, "l3_votes": a["l3_votes"],
                "l3_mean_drift_gap": mean_gap,
                "l3_gap_vs_zero": stats.paired_permutation(
                    a["l3_gaps"], [0.0] * len(a["l3_gaps"]), seed=seed)
                if a["l3_gaps"] else None,
                "aggregate_verdict": verdict,
                "per_run_verdicts": a["verdicts"]}
        # Holm across agents on the gap-vs-zero tests
        tested = [r for r in per_agent
                  if per_agent[r]["l3_gap_vs_zero"]
                  and per_agent[r]["l3_gap_vs_zero"]["p"] is not None]
        if tested:
            adj = stats.holm([per_agent[r]["l3_gap_vs_zero"]["p"]
                              for r in tested])
            for r, p_adj in zip(tested, adj):
                per_agent[r]["l3_gap_vs_zero"]["p_holm"] = p_adj
        agg["packs"][pack] = {"runs": [t for t, _ in items],
                              "run_verdicts": run_verdicts,
                              "degradations": sorted(degradations),
                              "per_agent": per_agent}
    return agg


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def to_markdown(agg, batchtag):
    lines = [f"# persona_probe aggregate — {batchtag}",
             f"generated: {agg['generated_at']}  runs: {agg['n_runs']}", ""]
    for pack, node in agg["packs"].items():
        lines += [f"## {pack}",
                  f"runs: {', '.join(node['runs'])}",
                  f"run verdicts: {node['run_verdicts']}"]
        if node["degradations"]:
            lines.append(f"degradations: {', '.join(node['degradations'])}")
        lines += ["",
                  "| agent | runs | L1 mean [95% CI] | L1 slope (med) | "
                  "declining runs | L2 norm mean | 2AFC persona rate | "
                  "drift gap | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for role, a in node["per_agent"].items():
            ci = a["l1_mean_ci"]
            ci_txt = (f"{_fmt(ci['stat'])} [{_fmt(ci['lo'])}, "
                      f"{_fmt(ci['hi'])}]" if ci["stat"] is not None else "—")
            gap = a["l3_mean_drift_gap"]
            gap_txt = _fmt(gap, 1)
            if a.get("l3_gap_vs_zero") and \
                    a["l3_gap_vs_zero"].get("p_holm") is not None:
                gap_txt += f" (p_holm={_fmt(a['l3_gap_vs_zero']['p_holm'])})"
            lines.append(
                f"| {role} | {a['n_runs']} | {ci_txt} | "
                f"{_fmt(a['l1_slope_median'], 4)} | "
                f"{a['l1_declining_runs']} | "
                f"{_fmt(a['l2_normative_mean'], 1)} | "
                f"{_fmt(a['l3_persona_rate'])} ({a['l3_votes']} votes) | "
                f"{gap_txt} | {a['aggregate_verdict']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.aggregate")
    ap.add_argument("batchtag")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="pack:tag pairs; default: discover all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.runs:
        triples = []
        for spec in args.runs:
            pack, _, tag = spec.partition(":")
            path = pp.probe_out_path(pack, tag)
            if not path.exists():
                print(f"[aggregate] WARNING: missing {path}", file=sys.stderr)
                continue
            triples.append((pack, tag, path))
    else:
        triples = discover_runs()
    if not triples:
        raise SystemExit("no persona_probe outputs found")
    runs = [(pack, tag, pp.load_json(path)) for pack, tag, path in triples]
    agg = aggregate_runs(runs, seed=args.seed)
    out_dir = pp.aggregate_dir()
    out_json = out_dir / f"aggregate_{args.batchtag}.json"
    pp.save_json_atomic(out_json, agg)
    (out_dir / f"aggregate_{args.batchtag}.md").write_text(
        to_markdown(agg, args.batchtag), encoding="utf-8")
    print(f"[aggregate] {len(runs)} runs -> {out_json}")
    return agg


if __name__ == "__main__":
    main()
