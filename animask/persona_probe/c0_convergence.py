"""Superseded by curve_compare.py: the story-level comparison runs on
checkpoint score curves, without a ratio. Kept as a cross-check of curve
endpoints; run via `run_eval --legacy`.

C0 — convergence ratio: did the model act out fewer story developments
than the authors wrote? Pure computation over C3 outcome cards (simulation
side) and canon cards — no LLM calls; both sides were coded by the same
encoder (c3_scoreboard), so the ratio compares like with like.

    python3 -m animask.persona_probe.c0_convergence <batchtag>

Per (actor_model | persona_level) group, over the stories that have BOTH a
simulation card and a canon card:
  - per-story sim card: majority tension mode / conflict curve across that
    story's runs (seed-averaged), mean
    decision-action distribution, mean goal-achieved rate.
  - dispersion, entropy version:  H(tension modes) in bits.
  - dispersion, pairwise version: mean pairwise card distance, where
    distance = mean of [mode differs, curve differs, JSD(action dists),
    |goal rate diff|] — each component in [0, 1].
  - C0 = sim dispersion / canon dispersion (per version; < 1 means the
    simulations are more alike than the authors' endings).
  - attractor: the modal (mode, curve) pattern among sim stories, its sim
    share, and that same pattern's share among canon stories — a mode the
    simulation collapses to that is NOT dominant in canon is model-added.

Output: results/persona_probe/c0_<batchtag>.{json,md}
"""
import argparse
import math
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe.b1_decisions import CATEGORY_KEYS
from animask.persona_probe.c3_scoreboard import discover_c3


def entropy(values: list) -> float:
    """Shannon entropy (bits) of the value distribution."""
    if not values:
        return 0.0
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalize(dist: dict) -> dict:
    total = sum(dist.values())
    if not total:
        return {}
    return {k: v / total for k, v in dist.items()}


def jsd(d1: dict, d2: dict) -> float:
    """Jensen–Shannon divergence (base 2, in [0,1]) between two count/prob
    dicts. Two empty dists -> 0; one empty -> 1 (maximally unlike)."""
    p, q = _normalize(d1), _normalize(d2)
    if not p and not q:
        return 0.0
    if not p or not q:
        return 1.0
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}

    def kl(a):
        return sum(a[k] * math.log2(a[k] / m[k]) for k in a if a[k] > 0)
    return (kl(p) + kl(q)) / 2


def card_distance(a: dict, b: dict) -> float:
    parts = [
        0.0 if a["mode"] == b["mode"] else 1.0,
        0.0 if a["curve"] == b["curve"] else 1.0,
        jsd(a["actions"], b["actions"]),
    ]
    if a["goal_rate"] is not None and b["goal_rate"] is not None:
        parts.append(abs(a["goal_rate"] - b["goal_rate"]))
    return sum(parts) / len(parts)


def mean_pairwise(cards: list) -> float:
    if len(cards) < 2:
        return 0.0
    ds = [card_distance(a, b) for i, a in enumerate(cards)
          for b in cards[i + 1:]]
    return sum(ds) / len(ds)


def _majority(values: list):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def slim_card(card: dict) -> dict:
    goals = list(card["goals"].values())
    return {"mode": card["tension_resolution"]["mode"],
            "curve": card["conflict_curve"],
            "actions": dict(card["decision_actions"]),
            "goal_rate": (sum(1 for g in goals if g == "achieved")
                          / len(goals)) if goals else None}


def story_card(cards: list) -> dict:
    """Seed-average several runs of one story into one card."""
    actions = {}
    for c in cards:
        for k, v in c["actions"].items():
            actions[k] = actions.get(k, 0) + v / len(cards)
    rates = [c["goal_rate"] for c in cards if c["goal_rate"] is not None]
    return {"mode": _majority([c["mode"] for c in cards]),
            "curve": _majority([c["curve"] for c in cards]),
            "actions": actions,
            "goal_rate": (sum(rates) / len(rates)) if rates else None}


def dispersion(cards: list) -> dict:
    return {"n": len(cards),
            "mode_entropy_bits": round(entropy([c["mode"] for c in cards]),
                                       4),
            "mode_dist": {m: [c["mode"] for c in cards].count(m)
                          for m in {c["mode"] for c in cards}},
            "mean_pairwise_distance": round(mean_pairwise(cards), 4)}


def attractor(sim_cards: list, canon_cards: list) -> dict:
    patterns = [(c["mode"], c["curve"]) for c in sim_cards]
    if not patterns:
        return {}
    modal = _majority(patterns)
    canon_patterns = [(c["mode"], c["curve"]) for c in canon_cards]
    return {"pattern": {"mode": modal[0], "curve": modal[1]},
            "sim_share": patterns.count(modal) / len(patterns),
            "canon_share": (canon_patterns.count(modal)
                            / len(canon_patterns)) if canon_patterns
            else None}


def compute(batchtag: str) -> dict:
    triples = discover_c3(batchtag)
    if not triples:
        raise SystemExit("no c3_scoreboard outputs found")
    canon_dir = pp.get_root() / "data" / "cache" / "canon_cards"
    groups = {}
    canon = {}
    for pack, tag, path in triples:
        sid = pack.split("__")[0]
        cpath = canon_dir / f"{sid}.json"
        if not cpath.exists():
            continue
        canon[sid] = slim_card(pp.load_json(cpath)["card"])
        res = pp.load_json(path)
        meta_p = pp.run_dir(pack) / f"run_meta_{tag}.json"
        meta = pp.load_json(meta_p) if meta_p.exists() else {}
        gkey = (f"{meta.get('models', {}).get('actor') or '?'}|"
                f"{meta.get('args', {}).get('persona_level') or 'P3'}")
        groups.setdefault(gkey, {}).setdefault(sid, []).append(
            slim_card(res["card"]))

    out = {"batchtag": batchtag, "generated_at": pp.now_str(),
           "groups": {}}
    for gkey, by_sid in sorted(groups.items()):
        paired = sorted(set(by_sid) & set(canon))
        sim_cards = [story_card(by_sid[s]) for s in paired]
        canon_cards = [canon[s] for s in paired]
        sim_d, canon_d = dispersion(sim_cards), dispersion(canon_cards)
        ratios = {}
        for key in ("mode_entropy_bits", "mean_pairwise_distance"):
            ratios[key] = (round(sim_d[key] / canon_d[key], 4)
                           if canon_d[key] else None)
        out["groups"][gkey] = {
            "paired_stories": paired,
            "simulation": sim_d, "canon": canon_d,
            "convergence_ratio": ratios,
            "attractor": attractor(sim_cards, canon_cards),
            "action_dist_sim": _merge([c["actions"] for c in sim_cards]),
            "action_dist_canon": _merge([c["actions"]
                                         for c in canon_cards])}
    return out


def _merge(dists: list) -> dict:
    total = {}
    for d in dists:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return {k: round(total[k], 2) for k in CATEGORY_KEYS if k in total}


def to_markdown(res: dict) -> str:
    lines = [f"# C0 convergence ratio — {res['batchtag']}",
             f"generated: {res['generated_at']}", ""]
    for gkey, g in res["groups"].items():
        s, c, r = g["simulation"], g["canon"], g["convergence_ratio"]
        att = g["attractor"]
        lines += [
            f"## {gkey}  ({s['n']} paired stories)",
            "",
            "| dispersion | simulation | canon | ratio (sim/canon) |",
            "|---|---|---|---|",
            f"| tension-mode entropy (bits) | {s['mode_entropy_bits']} | "
            f"{c['mode_entropy_bits']} | {r['mode_entropy_bits']} |",
            f"| mean pairwise card distance | "
            f"{s['mean_pairwise_distance']} | "
            f"{c['mean_pairwise_distance']} | "
            f"{r['mean_pairwise_distance']} |",
            "",
            f"mode dist — sim: {s['mode_dist']}  canon: {c['mode_dist']}",
            f"attractor: {att.get('pattern')} — sim share "
            f"{att.get('sim_share'):.2f}, canon share "
            + (f"{att['canon_share']:.2f}"
               if att.get("canon_share") is not None else "—"),
            f"action dist — sim: {g['action_dist_sim']}",
            f"action dist — canon: {g['action_dist_canon']}", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.c0_convergence")
    ap.add_argument("batchtag")
    args = ap.parse_args(argv)
    res = compute(args.batchtag)
    out_p = pp.aggregate_dir() / f"c0_{args.batchtag}.json"
    pp.save_json_atomic(out_p, res)
    (pp.aggregate_dir() / f"c0_{args.batchtag}.md").write_text(
        to_markdown(res), encoding="utf-8")
    for gkey, g in res["groups"].items():
        print(f"[c0] {gkey}: ratio(entropy)="
              f"{g['convergence_ratio']['mode_entropy_bits']} "
              f"ratio(pairwise)="
              f"{g['convergence_ratio']['mean_pairwise_distance']} "
              f"({g['simulation']['n']} stories)")
    print("saved", out_p)
    return res


if __name__ == "__main__":
    main()
