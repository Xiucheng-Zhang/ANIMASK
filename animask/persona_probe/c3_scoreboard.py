"""Superseded as the story-level instrument by story_curves.py (the
checkpoint questionnaire). The endpoint cards remain a cross-check of the
curve endpoints; run via `run_eval --legacy`. story_curves imports its
canon-segmentation helpers from here.

C3 — social-outcome scoreboard (simulation vs canon), and the outcome
cards C0 consumes.

Rows: tension resolution / per-role goal outcomes /
relation valence start-end / conflict curve / transgression counts. The
tension and goal rows also exist as pure rules over the engine ledgers
(terminator verdicts + timeline), kept alongside the coder's reading as a
cross-check. The remaining rows come from ONE outcome-coder LLM call per
text.

Symmetry: the SAME coder call runs on the simulation transcript and on the
canon post-freeze text (cached once per story), so C0's dispersion ratio
compares like with like. The coder also extracts a decision-action
distribution over the B1 taxonomy — C0 uses this symmetric version, NOT the
B1 pipeline's labels (which only exist on the simulation side).

    python3 -m animask.persona_probe.c3_scoreboard <pack> <tag> [--judge M]
    python3 -m animask.persona_probe.c3_scoreboard --canon <sid>
    python3 -m animask.persona_probe.c3_scoreboard --aggregate <batchtag>

Artifacts: results/runs/<pack>/c3_scoreboard_<tag>.json ;
data/cache/canon_cards/<sid>.json ; results/persona_probe/c3_<batchtag>.*
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe import atomize
from animask.persona_probe.b1_decisions import CATEGORY_KEYS, _taxonomy

TEXT_HEAD, TEXT_TAIL = 8000, 14000
TENSION_MODES = ("resolved_as_posed", "dissolved", "unresolved")
CURVES = ("escalates", "sustains", "de_escalates", "none")
GOAL_OUTCOMES = ("achieved", "partial", "failed", "abandoned", "unclear")
VALENCES = ("positive", "neutral", "negative", "mixed")
TRANSGRESSIONS = ("theft", "assault", "coercion", "deception", "betrayal")

_CODER_EN = """You are coding the OUTCOME of a story segment for a research \
scoreboard. Base every judgment ONLY on the text below.

[CENTRAL TENSION at the story's freeze point]
{tension}

[CHARACTERS and their goals]
{roles}

[TEXT]
{text}

Code the following:
1. tension_resolution.mode — how the central tension stands at the END of \
this text: "resolved_as_posed" (settled one way or another), "dissolved" \
(the tension evaporates without being settled — parties drift into \
agreement, distraction or inaction), "unresolved" (still open and live). \
Also "how": one sentence.
2. conflict_curve — the overall trajectory of interpersonal conflict \
across the text: "escalates" | "sustains" | "de_escalates" | "none".
3. goals — for each listed character: "achieved" | "partial" | "failed" | \
"abandoned" | "unclear".
4. relations — for each pair of listed characters that meaningfully \
interacts: valence toward each other at the START and at the END of this \
text: "positive" | "neutral" | "negative" | "mixed".
5. transgressions — count of DISTINCT events in the text: theft, assault, \
coercion, deception, betrayal (0 when none).
6. decision_actions — find every consequential choice by a listed \
character in the text and classify each into exactly one category; report \
counts per category (omit zero categories). Categories:
{taxonomy}

Output STRICT JSON only:
{{"tension_resolution": {{"mode": "...", "how": "..."}},
"conflict_curve": "...",
"goals": {{"<character>": "..."}},
"relations": [{{"pair": "<A>|<B>", "start": "...", "end": "..."}}],
"transgressions": {{"theft": 0, "assault": 0, "coercion": 0, \
"deception": 0, "betrayal": 0}},
"decision_actions": {{"<category>": <count>}}}}"""

_CODER_ZH = """你在为一项研究记分板给一段故事文本的「结局」编码。所有判断只基于\
下面的文本。

[故事冻结点的中心张力]
{tension}

[角色及其目标]
{roles}

[文本]
{text}

编码以下各项:
1. tension_resolution.mode——文本结束时中心张力的状态:"resolved_as_posed"\
(以某种方式了结)、"dissolved"(张力未被了结而自行消散——各方漂移到一致、\
分心或不了了之)、"unresolved"(仍然悬而未决)。另加 "how":一句话。
2. conflict_curve——文本中人际冲突的总体走势:"escalates" | "sustains" | \
"de_escalates" | "none"。
3. goals——对每个列出的角色:"achieved" | "partial" | "failed" | \
"abandoned" | "unclear"。
4. relations——对每对有实质互动的列出角色:在文本开始与结束时彼此的态度价态:\
"positive" | "neutral" | "negative" | "mixed"。
5. transgressions——文本中「不同事件」的计数:theft(偷窃)、assault(袭击)、\
coercion(胁迫)、deception(欺骗)、betrayal(背叛),没有则为 0。
6. decision_actions——找出文本中列出角色的每个有后果的选择,各归入一个类别;\
报告每类计数(为 0 的类别省略)。类别:
{taxonomy}

只输出严格 JSON:
{{"tension_resolution": {{"mode": "...", "how": "..."}},
"conflict_curve": "...",
"goals": {{"<角色>": "..."}},
"relations": [{{"pair": "<A>|<B>", "start": "...", "end": "..."}}],
"transgressions": {{"theft": 0, "assault": 0, "coercion": 0, \
"deception": 0, "betrayal": 0}},
"decision_actions": {{"<类别>": <计数>}}}}"""


def squeeze_text(text: str, head: int = TEXT_HEAD,
                 tail: int = TEXT_TAIL) -> str:
    text = str(text)
    if len(text) <= head + tail + 40:
        return text
    return (text[:head] + "\n\n[... middle omitted ...]\n\n"
            + text[-tail:])


def _norm(s: str) -> str:
    return (s.replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"'))


def canon_post_freeze(sid: str):
    """(post-freeze canon text, method) — freeze located by quote (curly
    quotes normalized), else by the freeze_word/total_words fraction."""
    meta = pp.load_json(pp.get_root() / "data" / "meta"
                        / f"resim_{sid}.json")
    path = pp.get_root() / "data" / "clean" / f"{sid}.txt"
    text = path.read_text(encoding="utf-8")
    quote = meta.get("freeze", {}).get("freeze_quote", "")
    if quote:
        i = _norm(text).find(_norm(quote))
        if i >= 0:
            return text[i + len(quote):], "quote"
    frac = meta.get("freeze_word", 0) / max(1, meta.get("total_words", 1))
    return text[int(len(text) * frac):], "word_fraction"


def _validate_card(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("coder reply not a dict")
    tr = parsed.get("tension_resolution") or {}
    mode = str(tr.get("mode", "")).strip()
    if mode not in TENSION_MODES:
        raise ValueError(f"bad tension mode {mode!r}")
    curve = str(parsed.get("conflict_curve", "")).strip()
    if curve not in CURVES:
        raise ValueError(f"bad conflict curve {curve!r}")
    goals = {str(k): v for k, v in (parsed.get("goals") or {}).items()
             if v in GOAL_OUTCOMES}
    rels = []
    for r in parsed.get("relations") or []:
        if (isinstance(r, dict) and r.get("start") in VALENCES
                and r.get("end") in VALENCES and "|" in str(r.get("pair"))):
            rels.append({"pair": str(r["pair"]), "start": r["start"],
                         "end": r["end"]})
    trans = {}
    for k in TRANSGRESSIONS:
        v = (parsed.get("transgressions") or {}).get(k, 0)
        trans[k] = max(0, int(v)) if isinstance(v, (int, float)) else 0
    acts = {}
    for k, v in (parsed.get("decision_actions") or {}).items():
        k = str(k).strip().lower()
        if k in CATEGORY_KEYS and isinstance(v, (int, float)) and v > 0:
            acts[k] = int(v)
    return {"tension_resolution": {"mode": mode,
                                   "how": str(tr.get("how", ""))[:300]},
            "conflict_curve": curve, "goals": goals, "relations": rels,
            "transgressions": trans, "decision_actions": acts}


def code_outcome(text: str, tension: str, roles_block: str, lang: str,
                 judge: str) -> dict:
    prompt = pp.L(lang, _CODER_ZH, _CODER_EN).format(
        tension=tension or "(not stated)", roles=roles_block,
        text=squeeze_text(text), taxonomy=_taxonomy(lang))
    return _validate_card(pp.parse_json_reply(
        pp.llm_query(prompt, model=judge)))


# --------------------------------------------------------------------------
# rule rows from engine ledgers
# --------------------------------------------------------------------------
def ledger_rows(pack: str, tag: str) -> dict:
    out = {"termination": None, "tension_status": None, "goal_status": {},
           "n_goal_updates": {}}
    meta_p = pp.run_dir(pack) / f"run_meta_{tag}.json"
    if meta_p.exists():
        out["termination"] = pp.load_json(meta_p).get("termination")
    tl_p = pp.run_dir(pack) / f"timeline_{tag}.json"
    ledger = []
    if tl_p.exists():
        tl = pp.load_json(tl_p)
        ledger = tl.get("resolution_ledger") or []
        for u in tl.get("goal_updates") or []:
            r = u.get("role")
            out["n_goal_updates"][r] = out["n_goal_updates"].get(r, 0) + 1
    if not ledger:
        v_p = pp.run_dir(pack) / f"verdicts_{tag}.json"
        if v_p.exists():
            v = pp.load_json(v_p)
            if isinstance(v, list) and v:
                ledger = v[-1].get("ledger") or []
    for e in ledger:
        if e.get("kind") == "tension" and e.get("id") == "central_tension":
            out["tension_status"] = e.get("status")
        elif e.get("kind") == "goal":
            out["goal_status"][e.get("id")] = e.get("status")
    return out


# --------------------------------------------------------------------------
# per-run / canon entry points
# --------------------------------------------------------------------------
def _roles_block(pack: str, roles: list) -> str:
    lines = []
    for role in roles:
        try:
            info = pp.load_json(atomize.role_info_path(pack, role))
        except (OSError, ValueError):
            info = {}
        name = info.get("role_name") or role.split("-")[0]
        lines.append(f"- {name}: {str(info.get('motivation', ''))[:200]}")
    return "\n".join(lines) or "-"


def sim_roles(pack: str) -> list:
    try:
        preset = pp.load_json(pp.preset_path(pack))
        return preset.get("role_agent_codes") or []
    except (OSError, ValueError):
        return []


def transcript_text(pack: str, tag: str) -> str:
    transcript = pp.load_json(pp.transcript_path(pack, tag))
    return "\n".join(f"[{m.get('code') or m.get('actor_type')}] {m['text']}"
                     for m in transcript
                     if m.get("actor_type") != "error")


def tension_of(pack: str) -> str:
    p = pp.run_dir(pack) / "freeze_validation.json"
    if p.exists():
        try:
            return pp.load_json(p).get("central_tension", "")
        except (OSError, ValueError):
            pass
    return ""


def run_c3(pack: str, tag: str, judge: str = pp.DEEPSEEK) -> dict:
    out_p = pp.run_dir(pack) / f"c3_scoreboard_{tag}.json"
    if out_p.exists():
        return pp.load_json(out_p)
    lang = pp.resolve_lang(pack)
    roles = sim_roles(pack)
    card = code_outcome(transcript_text(pack, tag), tension_of(pack),
                        _roles_block(pack, roles), lang, judge)
    result = {"pack": pack, "tag": tag, "generated_at": pp.now_str(),
              "judge": judge, "lang": lang,
              "rule": ledger_rows(pack, tag), "card": card}
    pp.save_json_atomic(out_p, result)
    return result


def canon_card(sid: str, judge: str = pp.DEEPSEEK,
               force: bool = False) -> dict:
    cache = pp.get_root() / "data" / "cache" / "canon_cards" / f"{sid}.json"
    if cache.exists() and not force:
        return pp.load_json(cache)
    pack = sid + "__oracle"
    lang = pp.resolve_lang(pack)
    text, method = canon_post_freeze(sid)
    card = code_outcome(text, tension_of(pack),
                        _roles_block(pack, sim_roles(pack)), lang, judge)
    result = {"sid": sid, "generated_at": pp.now_str(), "judge": judge,
              "lang": lang, "freeze_method": method,
              "text_chars": len(text), "card": card}
    pp.save_json_atomic(cache, result)
    return result


# --------------------------------------------------------------------------
# aggregate: simulation vs canon
# --------------------------------------------------------------------------
def discover_c3(batchtag: str):
    base = pp.get_root() / "results" / "runs"
    out = []
    for path in sorted(base.glob("*/c3_scoreboard_*.json")):
        tag = path.name[len("c3_scoreboard_"):-len(".json")]
        if tag == batchtag or tag.startswith(batchtag + "_"):
            out.append((path.parent.name, tag, path))
    return out


def _dist(values: list) -> dict:
    d = {}
    for v in values:
        d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def aggregate(batchtag: str) -> dict:
    triples = discover_c3(batchtag)
    if not triples:
        raise SystemExit("no c3_scoreboard outputs found")
    sim_cards, canon_cards, missing_canon = {}, {}, []
    rule_x_coder = {"agree": 0, "disagree": 0, "pairs": []}
    for pack, tag, path in triples:
        res = pp.load_json(path)
        sid = pack.split("__")[0]
        sim_cards[f"{pack}:{tag}"] = res
        cache = (pp.get_root() / "data" / "cache" / "canon_cards"
                 / f"{sid}.json")
        if cache.exists():
            canon_cards[sid] = pp.load_json(cache)
        else:
            missing_canon.append(sid)
        # cross-check: engine ledger tension status vs coder mode
        rule_status = res["rule"].get("tension_status")
        mode = res["card"]["tension_resolution"]["mode"]
        if rule_status in ("open", "resolved"):
            coder_open = mode in ("unresolved", "dissolved")
            ok = (rule_status == "open") == coder_open
            rule_x_coder["agree" if ok else "disagree"] += 1
            if not ok:
                rule_x_coder["pairs"].append(
                    {"run": f"{pack}:{tag}", "rule": rule_status,
                     "coder": mode})

    def side(cards):
        goals = [g for c in cards for g in c["card"]["goals"].values()]
        return {
            "n": len(cards),
            "tension_modes": _dist([c["card"]["tension_resolution"]["mode"]
                                    for c in cards]),
            "conflict_curves": _dist([c["card"]["conflict_curve"]
                                      for c in cards]),
            "goal_outcomes": _dist(goals),
            "goal_achieved_rate": (sum(1 for g in goals
                                       if g == "achieved") / len(goals)
                                   if goals else None),
            "transgressions_total": {
                k: sum(c["card"]["transgressions"].get(k, 0)
                       for c in cards) for k in TRANSGRESSIONS},
            "transgressions_per_story": (
                sum(sum(c["card"]["transgressions"].values())
                    for c in cards) / len(cards)) if cards else None,
            "decision_actions": {
                k: sum(c["card"]["decision_actions"].get(k, 0)
                       for c in cards) for k in CATEGORY_KEYS
                if any(c["card"]["decision_actions"].get(k)
                       for c in cards)},
        }

    # paired comparison only over stories that have BOTH sides
    paired_sids = {p.split("__")[0] for p, t, _ in
                   [(a, b, c) for a, b, c in triples]} & set(canon_cards)
    sim_paired = [v for k, v in sim_cards.items()
                  if k.split("__")[0] in paired_sids]
    canon_paired = [canon_cards[s] for s in sorted(paired_sids)]
    sim_side, canon_side = side(sim_paired), side(canon_paired)
    ratio = None
    if canon_side["transgressions_per_story"]:
        ratio = (sim_side["transgressions_per_story"]
                 / canon_side["transgressions_per_story"])
    out = {"batchtag": batchtag, "generated_at": pp.now_str(),
           "n_runs": len(sim_cards), "n_paired_stories": len(paired_sids),
           "missing_canon": sorted(set(missing_canon)),
           "simulation": sim_side, "canon": canon_side,
           "transgression_ratio_sim_over_canon": ratio,
           "tension_rule_x_coder": rule_x_coder}
    out_p = pp.aggregate_dir() / f"c3_{batchtag}.json"
    pp.save_json_atomic(out_p, out)
    (pp.aggregate_dir() / f"c3_{batchtag}.md").write_text(
        c3_markdown(out), encoding="utf-8")
    print(f"[c3] {len(sim_cards)} runs, {len(paired_sids)} paired with "
          f"canon -> {out_p}")
    return out


def c3_markdown(agg: dict) -> str:
    s, c = agg["simulation"], agg["canon"]

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, float) else str(x)

    lines = [f"# C3 social-outcome scoreboard — {agg['batchtag']}",
             f"generated: {agg['generated_at']}  paired stories: "
             f"{agg['n_paired_stories']}  (canon missing: "
             f"{', '.join(agg['missing_canon']) or 'none'})", "",
             "| row | simulation | canon |", "|---|---|---|"]
    for key in ("tension_modes", "conflict_curves", "goal_outcomes",
                "decision_actions", "transgressions_total"):
        lines.append(f"| {key} | {s[key]} | {c[key]} |")
    lines += [f"| goal_achieved_rate | {fmt(s['goal_achieved_rate'])} | "
              f"{fmt(c['goal_achieved_rate'])} |",
              f"| transgressions_per_story | "
              f"{fmt(s['transgressions_per_story'])} | "
              f"{fmt(c['transgressions_per_story'])} |", "",
              f"transgression ratio sim/canon: "
              f"{fmt(agg['transgression_ratio_sim_over_canon'])}",
              f"tension rule-vs-coder cross-check: "
              f"{agg['tension_rule_x_coder']['agree']} agree, "
              f"{agg['tension_rule_x_coder']['disagree']} disagree "
              f"{agg['tension_rule_x_coder']['pairs'] or ''}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.c3_scoreboard")
    ap.add_argument("pack", nargs="?")
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--judge", default=pp.DEEPSEEK)
    ap.add_argument("--canon", metavar="SID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", metavar="BATCHTAG")
    args = ap.parse_args(argv)

    if args.aggregate:
        return aggregate(args.aggregate)
    if args.canon:
        res = canon_card(args.canon, judge=args.judge, force=args.force)
        print(f"[c3] canon {args.canon}: "
              f"{res['card']['tension_resolution']['mode']} / "
              f"{res['card']['conflict_curve']} "
              f"(freeze via {res['freeze_method']})")
        return res
    if not (args.pack and args.tag):
        ap.error("pack and tag required (or --canon SID / --aggregate)")
    res = run_c3(args.pack, args.tag, judge=args.judge)
    print(f"[c3] {args.pack}/{args.tag}: "
          f"{res['card']['tension_resolution']['mode']} / "
          f"{res['card']['conflict_curve']} / trans="
          f"{sum(res['card']['transgressions'].values())}")
    return res


if __name__ == "__main__":
    main()
