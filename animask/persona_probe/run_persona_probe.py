"""CLI — run the persona-consistency probe on one finished resim run.

    python3 -m animask.persona_probe.run_persona_probe <pack> <tag> \
        [--layers l1,l2,l3] [--actor-model gpt-5.6-sol] \
        [--l1-judge gpt-5.6-luna] [--main-judge deepseek-v4-pro] \
        [--lang auto] [--l2-checkpoints 5] [--l2-paraphrases 2] \
        [--max-utterances-l1 0] [--dry-run]

Zero-intrusion: reads saved artifacts only; writes
results/runs/<pack>/persona_probe_<tag>[.l1|.l2|.l3].json and the atom
cache. --dry-run makes no LLM call and writes nothing: it parses, groups, and
prints the planned call counts. Old runs without llm_calls_<tag>.jsonl degrade
L2/L3 to {"skipped": "call_log_missing"} and record the degradation in the
output's verdict node.
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import atomize, l1_stream, l2_checkpoint, l3_ablation, stats

# --- interpretation-rule thresholds ---
MK_ALPHA = 0.05
L1_DECLINE_MIN = -0.05    # total Theil-Sen change over the series, 0..1 scale
L2_DECLINE_MIN = -5.0     # total change over the checkpoint span, 0..100
# 2AFC evidence floors. A positive claim (distinguishable) at rate 1.0 on
# n=6 has binomial p=0.016 vs chance; the null-like claim (inert) needs more
# votes AND a small probe-score gap — absence of evidence on 4 votes is not
# evidence of absence.
AFC_MIN_VOTES_DISTINGUISH = 6
AFC_MIN_VOTES_INERT = 8
AFC_DISTINGUISH = 0.75    # persona-arm pick rate: behavior distinguishable
AFC_INERT = 0.60          # at/below this: not distinguishable from default
AFC_INERT_MAX_GAP = 10.0  # persona-vs-ablated probe-score gap (0..100)

VERDICTS = ("persona_stable_model_trend", "drift_associated", "persona_inert",
            "persona_stable_attribution_pending", "mixed",
            "insufficient_evidence")


def _layer_ok(layer):
    return isinstance(layer, dict) and "skipped" not in layer


def build_verdict(l1, l2, l3, degradations):
    """Per-run reading of the interpretation rules.

    - flat L1/L2 trajectories + actual behavior distinguishable from the
      default arm -> trend attributed to the LLM's own tendency
      (persona held): persona_stable_model_trend
    - significant downward fidelity trajectory -> drift_associated
      (association only, no causal language)
    - actual behavior indistinguishable from the default arm -> persona_inert
      (reported separately)
    """
    roles = set()
    for layer in (l1, l2, l3):
        if _layer_ok(layer):
            roles |= set(layer.get("per_agent", {}))
    per_agent = {}
    for role in sorted(roles):
        ev = {}
        declining = False
        has_traj = False
        if _layer_ok(l1) and role in l1["per_agent"]:
            st = l1["per_agent"][role]["stats"]
            ev.update({"l1_mean": st["mean"], "l1_slope": st["theil_sen_slope"],
                       "l1_total_change": st["total_change"],
                       "l1_mk_p": st["mk"]["p"], "l1_n": st["n"]})
            if st["n"] >= 3:
                has_traj = True
                if (st["mk"]["p"] < MK_ALPHA
                        and st["total_change"] is not None
                        and st["total_change"] <= L1_DECLINE_MIN):
                    declining = True
        if _layer_ok(l2) and role in l2["per_agent"]:
            st = l2["per_agent"][role]["stats"]["normative"]
            ev.update({"l2_normative_mean": st["mean"],
                       "l2_slope": st["theil_sen_slope"],
                       "l2_total_change": st["total_change"],
                       "l2_mk_p": st["mk"]["p"], "l2_n": st["n"]})
            if st["n"] >= 3:
                has_traj = True
                if (st["mk"]["p"] < MK_ALPHA
                        and st["total_change"] is not None
                        and st["total_change"] <= L2_DECLINE_MIN):
                    declining = True
        rate = votes = None
        if _layer_ok(l3) and role in l3["per_agent"]:
            a = l3["per_agent"][role]
            rate, votes = a["persona_rate"], a["votes"]
            ev.update({"l3_persona_rate": rate, "l3_votes": votes,
                       "l3_mean_drift_gap": a["mean_drift_gap"]})
        gap = ev.get("l3_mean_drift_gap")
        distinguishable = (rate is not None
                           and votes >= AFC_MIN_VOTES_DISTINGUISH
                           and rate >= AFC_DISTINGUISH)
        inert = (rate is not None and votes >= AFC_MIN_VOTES_INERT
                 and rate <= AFC_INERT
                 and gap is not None and gap <= AFC_INERT_MAX_GAP)
        if inert:
            verdict = "persona_inert"
        elif declining:
            verdict = "drift_associated"
        elif has_traj and distinguishable:
            verdict = "persona_stable_model_trend"
        elif has_traj:
            verdict = "persona_stable_attribution_pending"
        else:
            verdict = "insufficient_evidence"
        per_agent[role] = {"verdict": verdict, "evidence": ev}

    counts = {v: 0 for v in VERDICTS}
    for a in per_agent.values():
        counts[a["verdict"]] += 1
    evaluable = [a for a in per_agent.values()
                 if a["verdict"] != "insufficient_evidence"]
    n_eval = len(evaluable)
    if n_eval == 0:
        run_verdict = "insufficient_evidence"
    elif counts["drift_associated"] * 2 >= n_eval:
        run_verdict = "drift_associated"
    elif counts["persona_inert"] == n_eval:
        run_verdict = "persona_inert"
    elif counts["persona_inert"] > 0:
        run_verdict = "mixed"
    elif counts["persona_stable_model_trend"] == n_eval:
        run_verdict = "persona_stable_model_trend"
    elif counts["persona_stable_attribution_pending"] > 0:
        run_verdict = "persona_stable_attribution_pending"
    else:
        run_verdict = "mixed"
    return {"run_verdict": run_verdict, "per_agent": per_agent,
            "verdict_counts": counts,
            "inert_agents": [r for r, a in per_agent.items()
                             if a["verdict"] == "persona_inert"],
            "degradations": sorted(set(degradations)),
            "thresholds": {"mk_alpha": MK_ALPHA,
                           "l1_decline_min": L1_DECLINE_MIN,
                           "l2_decline_min": L2_DECLINE_MIN,
                           "afc_min_votes_distinguish": AFC_MIN_VOTES_DISTINGUISH,
                           "afc_min_votes_inert": AFC_MIN_VOTES_INERT,
                           "afc_distinguish": AFC_DISTINGUISH,
                           "afc_inert": AFC_INERT},
            "rules_version": "v1"}


def build_stats_summary(l1, l2, l3, seed=0):
    """Run-level roll-up: clustered bootstrap CIs (cluster = agent) and the
    paired permutation test on the L3 arm gap (Holm over agents)."""
    out = {}
    if _layer_ok(l1):
        clusters = [[s["score"] for s in a["series"] if s["score"] is not None]
                    for a in l1["per_agent"].values()]
        out["l1_mean_score_ci"] = stats.bootstrap_ci_clustered(clusters,
                                                               seed=seed)
    if _layer_ok(l2):
        clusters = [[c["normative_score"] for c in a["checkpoints"]
                     if c["normative_score"] is not None]
                    for a in l2["per_agent"].values()]
        out["l2_normative_ci"] = stats.bootstrap_ci_clustered(clusters,
                                                              seed=seed)
    if _layer_ok(l3):
        perms, pvals, order = {}, [], []
        for role, a in l3["per_agent"].items():
            pairs = [(c["persona_probe_score"], c["ablated_probe_score"])
                     for c in a["checkpoints"]
                     if c.get("persona_probe_score") is not None
                     and c.get("ablated_probe_score") is not None]
            if not pairs:
                continue
            res = stats.paired_permutation([p[0] for p in pairs],
                                           [p[1] for p in pairs], seed=seed)
            perms[role] = res
            order.append(role)
            pvals.append(res["p"])
        if pvals:
            adj = stats.holm(pvals)
            for role, p_adj in zip(order, adj):
                perms[role]["p_holm"] = p_adj
        out["l3_arm_gap_permutation"] = perms
    return out


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="persona_probe", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("tag")
    ap.add_argument("--layers", default="l1,l2,l3")
    ap.add_argument("--actor-model", default=pp.GPT)
    ap.add_argument("--l1-judge", default=pp.MINI)
    ap.add_argument("--main-judge", default=pp.DEEPSEEK)
    ap.add_argument("--second-judge", default=pp.MINI)
    ap.add_argument("--lang", default="auto", choices=["auto", "zh", "en"])
    ap.add_argument("--l2-checkpoints", type=int, default=5)
    ap.add_argument("--l2-paraphrases", type=int, default=2,
                    help="variants per question INCLUDING the original")
    ap.add_argument("--l3-checkpoints", type=int, default=2)
    ap.add_argument("--max-utterances-l1", type=int, default=0,
                    help="0 = all utterances (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--no-bfi", action="store_true")
    ap.add_argument("--force-atomize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def resolve_roles(pack):
    """role_agent_codes from the preset, kept only if a role card exists."""
    preset = pp.load_json(pp.preset_path(pack))
    codes = preset.get("role_agent_codes", [])
    return [c for c in codes
            if (pp.roles_dir(pack) / c / "role_info.json").exists()]


def main(argv=None):
    args = _parse_args(argv)
    pack, tag = args.pack, args.tag
    preset = pp.preset_path(pack)
    if not preset.exists():
        raise SystemExit(f"pack {pack!r} not found: no preset at {preset} "
                         "(build it with python3 -m animask.resim.build_world_pack)")
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    for x in layers:
        if x not in ("l1", "l2", "l3"):
            raise SystemExit(f"unknown layer {x!r} (expected l1,l2,l3)")
    lang = pp.resolve_lang(pack, args.lang)
    roles = resolve_roles(pack)
    if not roles:
        raise SystemExit(f"no role cards found for pack {pack!r}")
    max_utt = args.max_utterances_l1 or None
    if not pp.transcript_path(pack, tag).exists():
        raise SystemExit(f"transcript not found: "
                         f"{pp.transcript_path(pack, tag)}")
    call_log_present = pp.call_log_path(pack, tag).exists()
    degradations = []
    if not call_log_present and ("l2" in layers or "l3" in layers):
        degradations.append("call_log_missing")

    # ---------------- dry run: parse + group + print counts ----------------
    if args.dry_run:
        cached_atoms = {r: (pp.load_json(atomize.atom_cache_path(pack, r))
                            ["atoms"]
                            if atomize.atom_cache_path(pack, r).exists()
                            else None) for r in roles}
        plan = {"pack": pack, "tag": tag, "lang": lang, "roles": roles,
                "layers": layers, "call_log_present": call_log_present,
                "degradations": degradations,
                "atomize": atomize.plan_atomize(pack, roles)}
        if "l1" in layers:
            plan["l1"] = l1_stream.plan_l1(pack, tag, roles, max_utt=max_utt)
        if "l2" in layers:
            plan["l2"] = l2_checkpoint.plan_l2(
                pack, tag, roles, cached_atoms,
                n_checkpoints=args.l2_checkpoints,
                n_variants=args.l2_paraphrases, bfi=not args.no_bfi)
        if "l3" in layers:
            plan["l3"] = l3_ablation.plan_l3(
                pack, tag, roles, cached_atoms,
                n_checkpoints=args.l3_checkpoints)
        total = plan["atomize"]["n_calls"]
        for key in ("l1", "l2", "l3"):
            node = plan.get(key)
            if isinstance(node, dict):
                total += node.get("total_calls", 0)
        plan["total_planned_calls"] = total
        print(f"[dry-run] pack={pack} tag={tag} lang={lang} "
              f"roles={len(roles)} layers={','.join(layers)}")
        print(f"[dry-run] call log present: {call_log_present}"
              + ("  -> L2/L3 degrade: call_log_missing"
                 if not call_log_present else ""))
        print(f"[dry-run] atomize: {plan['atomize']['n_calls']} calls "
              f"(cache misses: {plan['atomize']['cache_misses']})")
        if "l1" in plan:
            p = plan["l1"]
            print(f"[dry-run] L1: {p['n_utterances']} utterances "
                  f"{p['per_agent_utterances']} -> {p['judge_batches']} judge "
                  f"batches + {p['calib_batches']} calibration batches "
                  f"= {p['total_calls']} calls")
        for key in ("l2", "l3"):
            node = plan.get(key)
            if node is None:
                continue
            if "skipped" in node:
                print(f"[dry-run] {key.upper()}: skipped "
                      f"({node['skipped']})")
            else:
                print(f"[dry-run] {key.upper()}: {node['total_calls']} calls "
                      f"{{per agent: "
                      f"{ {r: v['total_calls'] for r, v in node['per_agent'].items()} }}}")
        print(f"[dry-run] TOTAL planned LLM calls: {total}")
        return plan

    # ---------------- real run ----------------
    # per-phase call accounting: <<ERROR results used to be skipped
    # silently, so a credit outage produced empty layers marked "done"
    phase_stats = {}

    def _phase(name, before):
        phase_stats[name] = {k: pp.CALL_STATS[k] - before[k]
                             for k in ("ok", "error", "parse_skip")}

    snap = dict(pp.CALL_STATS)
    atoms = atomize.atomize_pack(pack, roles, lang, model=args.main_judge,
                                 force=args.force_atomize)
    _phase("atomize", snap)
    l1 = l2 = l3 = {"skipped": "not_requested"}
    if "l1" in layers:
        snap = dict(pp.CALL_STATS)
        l1 = l1_stream.run_l1(pack, tag, roles, atoms, lang,
                              l1_judge=args.l1_judge,
                              calib_judge=args.main_judge,
                              max_utt=max_utt, seed=args.seed,
                              flush_every=args.flush_every)
        _phase("l1", snap)
    if "l2" in layers:
        snap = dict(pp.CALL_STATS)
        l2 = l2_checkpoint.run_l2(pack, tag, roles, atoms, lang,
                                  actor_model=args.actor_model,
                                  main_judge=args.main_judge,
                                  second_judge=args.second_judge,
                                  n_checkpoints=args.l2_checkpoints,
                                  n_variants=args.l2_paraphrases,
                                  seed=args.seed, bfi=not args.no_bfi,
                                  flush_every=args.flush_every)
        _phase("l2", snap)
    if "l3" in layers:
        snap = dict(pp.CALL_STATS)
        l3 = l3_ablation.run_l3(pack, tag, roles, atoms, lang,
                                actor_model=args.actor_model,
                                main_judge=args.main_judge,
                                n_checkpoints=args.l3_checkpoints,
                                flush_every=args.flush_every)
        _phase("l3", snap)

    calls_ok = sum(v["ok"] for v in phase_stats.values())
    calls_err = sum(v["error"] for v in phase_stats.values())
    calls_skip = sum(v.get("parse_skip", 0) for v in phase_stats.values())
    if calls_err and not calls_ok:
        # nothing succeeded (credit outage, provider down): layer stores keep
        # any earlier cached work, but a final json here would be an empty
        # result that the batch runner skips forever
        print(f"[persona_probe] ABORT: {calls_err} LLM calls, all failed "
              f"({phase_stats}) — final json NOT written")
        sys.exit(2)
    result = {
        "pack": pack, "tag": tag, "lang": lang, "roles": roles,
        "generated_at": pp.now_str(),
        "params": vars(args),
        "atoms": {r: {"n_atoms": len(a),
                      "by_type": {t: sum(1 for x in a if x["type"] == t)
                                  for t in atomize.ATOM_TYPES},
                      "cache_path": str(atomize.atom_cache_path(pack, r))}
                  for r, a in atoms.items()},
        "l1": l1, "l2": l2, "l3": l3,
        "call_stats": {"by_phase": phase_stats,
                       "ok": calls_ok, "error": calls_err,
                       "parse_skip": calls_skip},
        # parse-skips count too: those items are missing from the layer
        # stores and only a re-run (gated on complete=false) retries them
        "complete": calls_err == 0 and calls_skip == 0,
        "stats_summary": build_stats_summary(l1, l2, l3, seed=args.seed),
        "verdict": build_verdict(l1, l2, l3, degradations)}
    out = pp.probe_out_path(pack, tag)
    pp.save_json_atomic(out, result)
    print(f"[persona_probe] verdict={result['verdict']['run_verdict']} "
          f"-> {out}")
    if calls_err or calls_skip:
        print(f"[persona_probe] WARNING: {calls_err} failed calls, "
              f"{calls_skip} unparseable-judge skips ({phase_stats}) — "
              f"marked complete=false; a re-run resumes from the layer "
              f"stores")
        sys.exit(2)
    return result


if __name__ == "__main__":
    main()
