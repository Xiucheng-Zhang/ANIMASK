"""curve_compare — cross-story comparisons over the §4.1 score curves.

Pure computation, no LLM calls. Consumes story_curves_<tag>.json (simulation
side) and data/cache/canon_curves/<sid>.json (canon side) and emits, per
(actor model | persona level) group:

  - dispersion curves per score, both sides read side by side (std over
    stories per checkpoint, seed-merged) — no ratio: a strong genre
    convention collapses the canon-side dispersion and blows a ratio up,
    so the gap (canon - sim) is reported with a bootstrap CI instead
  - paired deviations per score: per-story mean(sim - canon), sign
    agreement across stories, paired permutation p, bootstrap CI

    python3 -m animask.persona_probe.curve_compare <batchtag>

Outputs: results/persona_probe/curves_<batchtag>.{json,md}
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import stats
from animask.persona_probe.story_curves import SCORES, N_CHECKPOINTS, discover_runs

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _actor_model(pack: str, tag: str) -> str:
    meta_p = pp.run_dir(pack) / f"run_meta_{tag}.json"
    try:
        return pp.load_json(meta_p).get("models", {}).get("actor") or "?"
    except (OSError, ValueError):
        return "?"


def load_sides(batchtag: str):
    """{group: {sid: [curves, ...]}} sim side (curves per seed) and
    {sid: curves} canon side."""
    sim = {}
    for pack, tag in discover_runs(batchtag):
        p = pp.run_dir(pack) / f"story_curves_{tag}.json"
        if not p.exists():
            continue
        d = pp.load_json(p)
        sid = pack[:-len("__oracle")] if pack.endswith("__oracle") else pack
        group = _actor_model(pack, tag)
        sim.setdefault(group, {}).setdefault(sid, []).append(d["curves"])
    canon = {}
    base = pp.get_root() / "data" / "cache" / "canon_curves"
    for p in sorted(base.glob("*.json")):
        if p.name.endswith(".store.json"):
            continue
        d = pp.load_json(p)
        canon[d["sid"]] = d["curves"]
    return sim, canon


def seed_merge(curve_list: list) -> dict:
    """Average the curves of several seeds of one story."""
    return {s: [_mean([c[s][k] for c in curve_list])
                for k in range(N_CHECKPOINTS)] for s in SCORES}


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------
def compare_group(story_curves: dict, canon: dict, seed: int = 0) -> dict:
    """story_curves: {sid: merged curves}; canon: {sid: curves}."""
    paired = {sid: (story_curves[sid], canon[sid])
              for sid in story_curves if sid in canon}
    n = len(paired)
    out = {"n_stories": len(story_curves), "n_paired": n}
    if n < 2:
        return out

    # dispersion curves per score, both sides + bootstrap CI on the gap
    import random
    rng = random.Random(seed)
    sids = list(paired)
    disp, gap = {}, {}
    for s in SCORES:
        disp[s], gap[s] = [], []
        for k in range(N_CHECKPOINTS):
            sim_d = _std([paired[i][0][s][k] for i in sids])
            can_d = _std([paired[i][1][s][k] for i in sids])
            disp[s].append({"sim": round(sim_d, 3),
                            "canon": round(can_d, 3)})
            boots = []
            for _ in range(1000):
                bs = [sids[rng.randrange(len(sids))] for _ in sids]
                boots.append(_std([paired[i][1][s][k] for i in bs])
                             - _std([paired[i][0][s][k] for i in bs]))
            boots.sort()
            gap[s].append({"gap": round(can_d - sim_d, 3),
                           "lo": round(boots[24], 3),
                           "hi": round(boots[974], 3)})
    out["dispersion"] = disp
    out["dispersion_gap"] = gap

    # paired deviations per score
    dev = {}
    for s in SCORES:
        per_story = {sid: _mean([sim_c[s][k] - can_c[s][k]
                                 for k in range(N_CHECKPOINTS)])
                     for sid, (sim_c, can_c) in paired.items()}
        diffs = list(per_story.values())
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        ci = stats.bootstrap_ci_clustered([[d] for d in diffs], seed=seed)
        p_perm = stats.paired_permutation(
            [_mean(sim_c[s]) for sim_c, _ in paired.values()],
            [_mean(can_c[s]) for _, can_c in paired.values()])
        dev[s] = {"mean_diff": round(_mean(diffs), 3),
                  "ci": ci, "n_pos": pos, "n_neg": neg,
                  "p_permutation": p_perm.get("p"),
                  "per_checkpoint_diff": [
                      round(_mean([sim_c[s][k] - can_c[s][k]
                                   for sim_c, can_c in paired.values()]), 3)
                      for k in range(N_CHECKPOINTS)]}
    # one permutation p per score; Holm step-down across the four scores
    pvals = [dev[s]["p_permutation"] for s in SCORES]
    if all(v is not None for v in pvals):
        for s, adj in zip(SCORES, stats.holm(pvals)):
            dev[s]["p_holm"] = round(adj, 4)
    out["deviation"] = dev
    return out


def aggregate(batchtag: str, seed: int = 0) -> dict:
    sim, canon = load_sides(batchtag)
    out = {"batchtag": batchtag, "generated_at": pp.now_str(),
           "n_canon": len(canon), "groups": {}}
    for group, stories in sorted(sim.items()):
        merged = {sid: seed_merge(cl) for sid, cl in stories.items()}
        out["groups"][group] = compare_group(merged, canon, seed=seed)
    out_p = pp.aggregate_dir() / f"curves_{batchtag}.json"
    pp.save_json_atomic(out_p, out)
    (pp.aggregate_dir() / f"curves_{batchtag}.md").write_text(
        to_markdown(out), encoding="utf-8")
    print(f"[curve_compare] {sum(len(v) for v in sim.values())} sim "
          f"stories, {len(canon)} canons -> {out_p}")
    return out


# --------------------------------------------------------------------------
# markdown summary
# --------------------------------------------------------------------------
def to_markdown(res: dict) -> str:
    lines = [f"# story-curve comparisons — {res['batchtag']}",
             f"generated: {res['generated_at']}  canons: {res['n_canon']}",
             ""]
    for group, g in res["groups"].items():
        lines.append(f"## {group}  ({g.get('n_paired', 0)} paired stories)")
        if "dispersion" not in g:
            lines.append("(not enough paired stories)")
            continue
        lines.append("")
        lines.append("| score | sim std c1..c5 | canon std c1..c5 | "
                     "gap@c5 (canon-sim) [95% CI] |")
        lines.append("|---|---|---|---|")
        for s in SCORES:
            d = g["dispersion"][s]
            sim = " ".join(f"{x['sim']:.2f}" for x in d)
            can = " ".join(f"{x['canon']:.2f}" for x in d)
            e = g["dispersion_gap"][s][-1]
            lines.append(f"| {s} | {sim} | {can} | "
                         f"{e['gap']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}] |")
        lines.append("")
        lines.append("| score | mean sim-canon | 95% CI | sign | p(perm) | p(Holm) |")
        lines.append("|---|---|---|---|---|---|")
        for s in SCORES:
            d = g["deviation"][s]
            ci = d["ci"]
            lines.append(
                f"| {s} | {d['mean_diff']:+.2f} | "
                f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}] | "
                f"{d['n_pos']}+/{d['n_neg']}- | "
                f"{d['p_permutation'] if d['p_permutation'] is not None else '--'} | "
                f"{d.get('p_holm', '--')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.curve_compare")
    ap.add_argument("batchtag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    aggregate(args.batchtag, seed=args.seed)


if __name__ == "__main__":
    main()
