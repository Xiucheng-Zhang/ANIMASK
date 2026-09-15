"""A1 — facet-level presence profile + gate, from stored L2 scores.

Pure reprocessing: reads persona_probe_<tag>.json (probe_set, checkpoint
fracs) and the .l2.json layer store (per-question judge scores). No LLM
calls.

    python3 -m animask.persona_probe.facet_profile <batchtag> [--runs pack:tag ...]

Facet mapping (from each probe_set question, in priority order):
  expectation should_know / should_not_know -> knowledge   (QK*/F* qids)
  atom_id R* -> relations ; atom_id K* -> knowledge
  everything else (identity I1/I2, conditional C*, trait T*) -> traits_values
The fourth paper facet (style) is passive-layer by design (style is
measured on the normal run, not the interview fork);
each role carries its L1 series + rule style features under "l1_passive".

Gate (thresholds fixed before analysis; a role-run must pass
to enter the B1 attribution pool, failing is itself a reportable result):
  - traits_values and relations facet: checkpoint mean >= GATE_FACET_MIN at
    >= GATE_CP_PASS_FRAC of that facet's scored checkpoints. 80 on the
    penalty scale (score = 100 - 5*sum(severity)) means on average one
    severity-4 violation (or equivalent) per answer — persona materially
    broken at that checkpoint.
  - knowledge leak: t0 possession audit (persona_probe.leak_audit). The raw
    "any violation on a should_not_know answer" count is NOT a leak measure
    — inspection of real runs showed it conflates (a) foreign atoms that are
    ABOUT the probed character (asking Lincoln what model of machine he is),
    (b) legitimate in-story knowledge propagation after the freeze, (c) the
    character disclosing their OWN secrets to the interviewer, and (d)
    ordinary trait violations on that answer. So: FK samples whose source
    atom/question mentions the probed character are excluded as invalid
    (rule, `fk_validity`); valid samples are judged at t0 only (nothing has
    propagated yet) for possession of the specific foreign fact; the gate
    uses that audited verdict. Raw violation counts stay as the
    "fk_violations" diagnostic. Audit missing -> component untested.
  - A component with no questions in this pack (e.g. single-role packs have
    no should_not_know source) passes vacuously and is listed in
    "untested" — packs are not punished for battery gaps.

Output: results/persona_probe/facets_<batchtag>.json + .md — per-run
per-role facet series (fig:drift data, pooled by nominal checkpoint slot
0/.25/.5/.75/1) and per actor_model x persona_level group rollups with
clustered-bootstrap CIs (cluster = pack) for tab:presence say-side columns.
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import stats

FACETS = ("knowledge", "relations", "traits_values")

GATE_FACET_MIN = 80.0
GATE_CP_PASS_FRAC = 0.8
GATE_FACETS = ("traits_values", "relations")

SLOTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fk_validity(pack: str, role: str, q: dict) -> dict:
    """Rule check of one should_not_know sample: the foreign atom must not be
    about the probed character themselves (else "does the character know it"
    is trivially yes and the sample measures disclosure, not boundary).
    Returns {valid, reason, source_role, atom_id, statement}."""
    src = q.get("source_role")
    atom_id = (q.get("qid") or "")[1:] or None   # "FK01" -> "K01"
    stmt = None
    if src and atom_id:
        cache = pp.cache_dir(pack) / f"{src}.atoms.json"
        if cache.exists():
            atoms = pp.load_json(cache)["atoms"]
            stmt = next((a["statement"] for a in atoms
                         if a["id"] == atom_id), None)
    out = {"source_role": src, "atom_id": atom_id, "statement": stmt}
    names = {pp.role_display_name(pack, role), role.split("-")[0]}
    hay = " ".join([stmt or ""] + list(q.get("variants", [])))
    hit = next((n for n in names if n and n.lower() in hay.lower()), None)
    if stmt is None:
        return dict(out, valid=False, reason="source_atom_missing")
    if hit:
        return dict(out, valid=False, reason=f"self_referential({hit})")
    return dict(out, valid=True, reason=None)


def facet_of(q: dict) -> str:
    """Facet of one probe_set question (see module docstring)."""
    if q.get("expectation") not in (None, "none"):
        return "knowledge"
    aid = q.get("atom_id") or ""
    if aid.startswith("R"):
        return "relations"
    if aid.startswith("K"):
        return "knowledge"
    return "traits_values"


def nearest_slot(frac: float) -> float:
    return min(SLOTS, key=lambda s: abs(s - frac))


# --------------------------------------------------------------------------
# per-run profile
# --------------------------------------------------------------------------
def role_profile(role: str, l2_agent: dict, items: dict,
                 fk_valid: dict = None, audit: dict = None) -> dict:
    """Facet series + leak + gate for one role of one run.

    l2_agent: probe json l2.per_agent[role]; items: .l2 store items;
    fk_valid: {qid: fk_validity dict} (missing -> FK samples treated valid);
    audit: this role's leak_audit entry ({leaked_t0, ...}) or None."""
    pset = l2_agent.get("probe_set", [])
    cps = l2_agent.get("checkpoints", [])
    fk_valid = fk_valid or {}

    def _fk_ok(q):
        v = fk_valid.get(q["qid"])
        return v is None or v["valid"]

    by_facet = {f: [] for f in FACETS}
    fk_viol = fk_n = 0
    excluded = sorted(q["qid"] for q in pset
                      if q.get("expectation") == "should_not_know"
                      and not _fk_ok(q))
    for cp in cps:
        ci = cp["call_index"]
        scores = {f: [] for f in FACETS}
        for q in pset:
            if q.get("expectation") == "should_not_know" and not _fk_ok(q):
                continue    # invalid sample: out of scores AND diagnostics
            f = facet_of(q)
            for vi in range(len(q.get("variants", []) or [None])):
                rec = items.get(f"score|{role}|{ci}|{q['qid']}|{vi}")
                if rec is None:
                    continue
                scores[f].append(rec["score"])
                if q.get("expectation") == "should_not_know":
                    fk_n += 1
                    if rec.get("violations"):
                        fk_viol += 1
        for f in FACETS:
            by_facet[f].append({
                "call_index": ci, "frac": cp["frac"],
                "slot": nearest_slot(cp["frac"]),
                "score": stats.mean(scores[f]),
                "n_scored": len(scores[f])})

    facets = {}
    for f in FACETS:
        pts = [p for p in by_facet[f] if p["score"] is not None]
        facets[f] = {"series": by_facet[f],
                     "stats": stats.series_stats([p["frac"] for p in pts],
                                                 [p["score"] for p in pts])}

    # ---- gate ----
    components, untested = {}, []
    for f in GATE_FACETS:
        pts = [p for p in facets[f]["series"] if p["score"] is not None]
        if not pts:
            components[f] = {"tested": False, "pass": True}
            untested.append(f)
            continue
        ok = sum(1 for p in pts if p["score"] >= GATE_FACET_MIN)
        components[f] = {"tested": True,
                         "pass": ok >= GATE_CP_PASS_FRAC * len(pts),
                         "cp_pass": ok, "cp_total": len(pts)}
    has_fk = any(q.get("expectation") == "should_not_know" for q in pset)
    valid_fk = [q for q in pset if q.get("expectation") == "should_not_know"
                and _fk_ok(q)]
    if not has_fk:
        leak = {"tested": False, "pass": True, "reason": "no_fk_question"}
    elif not valid_fk:
        leak = {"tested": False, "pass": True,
                "reason": "fk_invalid: " + ", ".join(
                    fk_valid[q]["reason"] for q in excluded
                    if q in fk_valid)}
    elif audit is None or audit.get("leaked_t0") is None:
        leak = {"tested": False, "pass": True,
                "reason": (audit or {}).get("reason") or "audit_pending"}
    else:
        leak = {"tested": True, "pass": not audit["leaked_t0"],
                "leaked_t0": audit["leaked_t0"]}
    components["knowledge_leak"] = leak
    if not leak["tested"]:
        untested.append("knowledge_leak")
    return {"facets": facets,
            "fk_violations": {"n": fk_n, "violated": fk_viol,
                              "rate": (fk_viol / fk_n) if fk_n else None,
                              "excluded_invalid": excluded},
            "leak": leak,
            "gate": {"pass": all(c["pass"] for c in components.values()),
                     "components": components, "untested": untested}}


def run_profile(pack: str, tag: str, audit: dict = None) -> dict:
    """All roles of one run; {"skipped": reason} when L2 is unusable.
    audit: leak_audit "roles" mapping keyed "<pack>:<tag>:<role>"."""
    res = pp.load_json(pp.probe_out_path(pack, tag))
    l2 = res.get("l2")
    if not isinstance(l2, dict) or "skipped" in l2 or "per_agent" not in l2:
        return {"skipped": "l2_missing_or_skipped"}
    store_path = pp.layer_path(pack, tag, "l2")
    if not store_path.exists():
        return {"skipped": "l2_store_missing"}
    items = pp.load_json(store_path)["items"]

    meta_p = pp.run_dir(pack) / f"run_meta_{tag}.json"
    actor = persona = None
    if meta_p.exists():
        meta = pp.load_json(meta_p)
        actor = meta.get("models", {}).get("actor") \
            or meta.get("args", {}).get("actor_model")
        persona = meta.get("args", {}).get("persona_level") or "P3"
    out = {"pack": pack, "tag": tag, "actor_model": actor,
           "persona_level": persona, "complete": res.get("complete", True),
           "per_role": {}}
    for role in res.get("roles", []):
        if role not in l2["per_agent"]:
            continue
        agent = l2["per_agent"][role]
        fk_valid = {q["qid"]: fk_validity(pack, role, q)
                    for q in agent.get("probe_set", [])
                    if q.get("expectation") == "should_not_know"}
        prof = role_profile(role, agent, items, fk_valid=fk_valid,
                            audit=(audit or {}).get(f"{pack}:{tag}:{role}"))
        prof["fk_validity"] = fk_valid
        l1 = res.get("l1")
        if isinstance(l1, dict) and role in l1.get("per_agent", {}):
            a1 = l1["per_agent"][role]
            prof["l1_passive"] = {"stats": a1.get("stats"),
                                  "feature_means": a1.get("feature_means")}
        out["per_role"][role] = prof
    return out


# --------------------------------------------------------------------------
# batch rollup
# --------------------------------------------------------------------------
def discover_runs(batchtag: str):
    base = pp.get_root() / "results" / "runs"
    out = []
    if not base.exists():
        return out
    for path in sorted(base.glob("*/persona_probe_*.json")):
        if path.name.endswith((".l1.json", ".l2.json", ".l3.json")):
            continue
        tag = path.name[len("persona_probe_"):-len(".json")]
        if tag == batchtag or tag.startswith(batchtag + "_"):
            out.append((path.parent.name, tag))
    return out


def _group_key(run: dict) -> str:
    return f"{run.get('actor_model') or '?'}|{run.get('persona_level') or '?'}"


def rollup(runs: list, seed: int = 0) -> dict:
    """runs: [run_profile dicts (non-skipped)] -> per-group aggregates."""
    groups = {}
    for run in runs:
        groups.setdefault(_group_key(run), []).append(run)
    out = {}
    for gkey, gruns in sorted(groups.items()):
        # cluster = pack; one value per role-run
        facet_clusters = {f: {} for f in FACETS}
        slot_clusters = {f: {s: {} for s in SLOTS} for f in FACETS}
        slopes = {f: [] for f in FACETS}
        n_roles = passed = audited = leaked = 0
        leak_untested = {}
        failed_roles = []
        for run in gruns:
            for role, prof in run["per_role"].items():
                n_roles += 1
                if prof["gate"]["pass"]:
                    passed += 1
                else:
                    failed_roles.append(f"{run['pack']}/{role}")
                lk = prof["leak"]
                if lk["tested"]:
                    audited += 1
                    leaked += 1 if lk["leaked_t0"] else 0
                else:
                    r = (lk.get("reason") or "?").split(":")[0].split("(")[0]
                    leak_untested[r] = leak_untested.get(r, 0) + 1
                for f in FACETS:
                    st = prof["facets"][f]["stats"]
                    if st["mean"] is not None:
                        facet_clusters[f].setdefault(run["pack"],
                                                     []).append(st["mean"])
                    if st["theil_sen_slope"] is not None:
                        slopes[f].append(st["theil_sen_slope"])
                    for p in prof["facets"][f]["series"]:
                        if p["score"] is not None:
                            slot_clusters[f][p["slot"]].setdefault(
                                run["pack"], []).append(p["score"])
        facet_means = {f: stats.bootstrap_ci_clustered(
            list(facet_clusters[f].values()), seed=seed) for f in FACETS}
        drift_curve = {}
        for f in FACETS:
            drift_curve[f] = {}
            for s in SLOTS:
                clusters = list(slot_clusters[f][s].values())
                if not clusters:
                    continue
                drift_curve[f][str(s)] = dict(
                    stats.bootstrap_ci_clustered(clusters, seed=seed),
                    n=sum(len(c) for c in clusters))
        out[gkey] = {
            "n_runs": len(gruns), "n_role_runs": n_roles,
            "facet_means": facet_means,
            "facet_slope_medians": {f: stats.median(slopes[f])
                                    for f in FACETS},
            "drift_curve": drift_curve,
            "gate_pass": passed, "gate_pass_rate": (passed / n_roles
                                                    if n_roles else None),
            "gate_failed_roles": sorted(failed_roles),
            "leak_t0": {"audited": audited, "leaked": leaked,
                        "rate": (leaked / audited) if audited else None,
                        "untested": leak_untested}}
    return out


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def _fmt(x, nd=1):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def to_markdown(result: dict) -> str:
    lines = [f"# A1 facet presence profile — {result['batchtag']}",
             f"generated: {result['generated_at']}  "
             f"runs: {len(result['runs'])}  "
             f"(gate: facet>={GATE_FACET_MIN} at >={GATE_CP_PASS_FRAC:.0%} "
             f"checkpoints; no audited t0 knowledge leak)", ""]
    for gkey, g in result["groups"].items():
        lt = g["leak_t0"]
        lines += [f"## {gkey}",
                  f"role-runs: {g['n_role_runs']}  gate pass: "
                  f"{g['gate_pass']}/{g['n_role_runs']}  t0 leak: "
                  f"{lt['leaked']}/{lt['audited']} audited"
                  f" (untested: {lt['untested'] or 'none'})", ""]
        lines += ["| facet | mean [95% CI] | slope (med) | "
                  + " | ".join(f"@{s:g}" for s in SLOTS) + " |",
                  "|---|---|---|" + "---|" * len(SLOTS)]
        for f in FACETS:
            ci = g["facet_means"][f]
            ci_txt = (f"{_fmt(ci['stat'])} [{_fmt(ci['lo'])}, "
                      f"{_fmt(ci['hi'])}]" if ci["stat"] is not None else "—")
            curve = [_fmt(g["drift_curve"][f].get(str(s), {}).get("stat"))
                     for s in SLOTS]
            lines.append(f"| {f} | {ci_txt} | "
                         f"{_fmt(g['facet_slope_medians'][f], 3)} | "
                         + " | ".join(curve) + " |")
        if g["gate_failed_roles"]:
            lines += ["", "gate FAILED: " + ", ".join(g["gate_failed_roles"])]
        lines.append("")
    lines += ["## per role-run", "",
              "| pack | role | knowledge | relations | traits_values | "
              "t0 leak | fk viol | gate |", "|---|---|---|---|---|---|---|---|"]
    for run in result["runs"]:
        for role, prof in run["per_role"].items():
            lk = prof["leak"]
            if lk["tested"]:
                leak_txt = "LEAK" if lk["leaked_t0"] else "clean"
            else:
                leak_txt = (lk.get("reason") or "—").split(":")[0]
            fv = prof["fk_violations"]
            fv_txt = (f"{fv['violated']}/{fv['n']}" if fv["n"] else "—")
            lines.append(
                "| " + " | ".join([
                    run["pack"], role]
                    + [_fmt(prof["facets"][f]["stats"]["mean"])
                       for f in FACETS]
                    + [leak_txt, fv_txt,
                       "pass" if prof["gate"]["pass"] else "**FAIL**"])
                + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.facet_profile")
    ap.add_argument("batchtag")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="pack:tag pairs; default: discover by batchtag")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-incomplete", action="store_true",
                    help="also profile runs whose probe json has "
                         "complete=false (their scored subset is valid but "
                         "checkpoints may be missing)")
    args = ap.parse_args(argv)

    if args.runs:
        pairs = [tuple(spec.split(":", 1)) for spec in args.runs]
    else:
        pairs = discover_runs(args.batchtag)
    if not pairs:
        raise SystemExit("no probe outputs found")
    audit_p = pp.aggregate_dir() / f"leak_audit_{args.batchtag}.json"
    audit = pp.load_json(audit_p)["roles"] if audit_p.exists() else {}
    runs, skipped = [], {}
    for pack, tag in pairs:
        prof = run_profile(pack, tag, audit=audit)
        if "skipped" in prof:
            skipped[f"{pack}:{tag}"] = prof["skipped"]
        elif not prof["complete"] and not args.include_incomplete:
            skipped[f"{pack}:{tag}"] = "probe_incomplete"
        else:
            runs.append(prof)
    result = {"batchtag": args.batchtag, "generated_at": pp.now_str(),
              "constants": {"GATE_FACET_MIN": GATE_FACET_MIN,
                            "GATE_CP_PASS_FRAC": GATE_CP_PASS_FRAC,
                            "GATE_FACETS": list(GATE_FACETS),
                            "leak_gate": "t0_possession_audit"},
              "leak_audit_used": audit_p.name if audit else None,
              "skipped_runs": skipped, "runs": runs,
              "groups": rollup(runs, seed=args.seed)}
    out_dir = pp.aggregate_dir()
    out_json = out_dir / f"facets_{args.batchtag}.json"
    pp.save_json_atomic(out_json, result)
    (out_dir / f"facets_{args.batchtag}.md").write_text(
        to_markdown(result), encoding="utf-8")
    print(f"[facet_profile] {len(runs)} runs ({len(skipped)} skipped) "
          f"-> {out_json}")
    return result


if __name__ == "__main__":
    main()
