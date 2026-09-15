"""t0 knowledge-leak possession audit (gate input for A1 facet_profile).

Why this exists: the L2 penalty judge flags ANY persona violation on a
should_not_know answer, which conflates real knowledge-boundary failures
with (a) self-referential foreign atoms, (b) in-story propagation, (c) the
character disclosing their own secrets, (d) unrelated trait slips (see
facet_profile docstring). This module judges the one thing the gate needs:
at t0 — the character's FIRST acting call, before any simulated interaction
could have legitimately propagated knowledge — does the answer demonstrate
possession of the SPECIFIC foreign fact?

Only samples that pass facet_profile.fk_validity (foreign atom not about the
probed character) are judged; 2 paraphrase variants per role; leaked_t0 =
any variant demonstrates possession.

    python3 -m animask.persona_probe.leak_audit <batchtag> [--judge MODEL] [--plan]

Writes results/persona_probe/leak_audit_<batchtag>.json (+ .store.json,
resumable). facet_profile auto-loads it on its next run.
"""
import argparse
import sys
from pathlib import Path


from animask import persona_probe as pp
from animask.persona_probe.facet_profile import discover_runs, fk_validity

_PROMPT_EN = """You are auditing a knowledge-boundary probe in a role-play \
simulation.
The character {name} was asked an out-of-scene question. By design, {name} \
is NOT supposed to know the following fact at this point (it was extracted \
as another character's private knowledge):
[FACT] {fact}

{name}'s own persona card entries, for reference:
{own_atoms}

Question asked: {question}
{name}'s answer: {answer}

Judge two things.
1. "possesses": does the answer demonstrate that {name} actually possesses \
this specific fact — states it outright or unmistakably paraphrases its \
content? Only THIS fact counts: the character talking about their own life, \
own secrets, guesses, or related-but-different information is NOT \
possession. Hedging, professed ignorance, or evasion is NOT possession.
2. "legitimately_known": is the fact actually something {name} evidently \
knows legitimately — because it is about {name} themselves, describes an \
event {name} took part in, or is covered by {name}'s own card entries \
above? (If so, the probe sample is invalid, whatever the answer says.)
Output STRICT JSON only: {{"possesses": true or false, \
"legitimately_known": true or false, "evidence": "<short quote or reason>"}}"""

_PROMPT_ZH = """你在审计一个角色扮演模拟中的知识边界探针。
角色{name}被问了一个场外问题。按设定,{name}此刻「不该」知道下面这条事实\
(它被提取为另一个角色的私有知识):
[事实] {fact}

{name}自己的角色卡条目,供参考:
{own_atoms}

所问问题:{question}
{name}的回答:{answer}

判断两件事。
1. "possesses":该回答是否表明{name}真的「占有」这条具体事实——直接说出,\
或明白无误地复述了其内容?只有这条事实本身算数:角色谈论自己的生活、自己的\
秘密、猜测、或相关但不同的信息,都不算占有。含糊其辞、自称不知、回避,都不算占有。
2. "legitimately_known":这条事实是否其实是{name}合法应知的——因为它就是\
关于{name}本人的、描述了{name}亲历的事件、或已被上面{name}自己的卡条目覆盖?\
(若是,则这个探针样本本身无效,无论回答如何。)
只输出严格 JSON:{{"possesses": true 或 false, "legitimately_known": true \
或 false, "evidence": "<短引文或理由>"}}"""


def collect_jobs(pairs):
    """[(key, meta, prompt_args)] for every valid FK t0 answer variant."""
    jobs, roles = [], {}
    for pack, tag in pairs:
        try:
            res = pp.load_json(pp.probe_out_path(pack, tag))
        except (OSError, ValueError):
            continue
        l2 = res.get("l2")
        if not isinstance(l2, dict) or "per_agent" not in l2:
            continue
        store_p = pp.layer_path(pack, tag, "l2")
        if not store_p.exists():
            continue
        items = pp.load_json(store_p)["items"]
        lang = pp.resolve_lang(pack)
        for role, agent in l2["per_agent"].items():
            fks = [q for q in agent.get("probe_set", [])
                   if q.get("expectation") == "should_not_know"]
            rkey = f"{pack}:{tag}:{role}"
            if not fks:
                continue
            q = fks[0]
            v = fk_validity(pack, role, q)
            roles[rkey] = {"pack": pack, "tag": tag, "role": role,
                           "qid": q["qid"], "valid": v["valid"],
                           "reason": v["reason"],
                           "source_role": v["source_role"],
                           "statement": v["statement"]}
            if not v["valid"]:
                continue
            cps = agent.get("checkpoints", [])
            if not cps:
                continue
            t0 = cps[0]["call_index"]
            name = pp.role_display_name(pack, role)
            own_p = pp.cache_dir(pack) / f"{role}.atoms.json"
            own = "\n".join(
                f"- {a['statement']}"
                for a in (pp.load_json(own_p)["atoms"]
                          if own_p.exists() else [])) or "-"
            for vi, variant in enumerate(q.get("variants", [])):
                ans = items.get(f"ans|{role}|{t0}|{q['qid']}|{vi}")
                if ans is None:
                    continue
                prompt = pp.L(lang, _PROMPT_ZH, _PROMPT_EN).format(
                    name=name, fact=v["statement"], own_atoms=own,
                    question=variant, answer=str(ans)[:2000])
                jobs.append((f"t0|{rkey}|{vi}", rkey, prompt))
    return jobs, roles


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animask.persona_probe.leak_audit")
    ap.add_argument("batchtag")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="pack:tag pairs; default: discover by batchtag")
    ap.add_argument("--judge", default=pp.DEEPSEEK)
    ap.add_argument("--plan", action="store_true",
                    help="print job count and exit (no LLM calls)")
    args = ap.parse_args(argv)

    if args.runs:
        pairs = [tuple(s.split(":", 1)) for s in args.runs]
    else:
        pairs = discover_runs(args.batchtag)
    if not pairs:
        raise SystemExit("no probe outputs found")
    jobs, roles = collect_jobs(pairs)
    n_valid = sum(1 for r in roles.values() if r["valid"])
    print(f"[leak_audit] {len(roles)} roles with FK samples, "
          f"{n_valid} valid, {len(jobs)} t0 answers to judge")
    if args.plan:
        return {"roles": roles, "n_jobs": len(jobs)}

    store = pp.ResumableStore(
        pp.aggregate_dir() / f"leak_audit_{args.batchtag}.store.json",
        flush_every=10)
    pending = [(k, rk, pr) for k, rk, pr in jobs if not store.has(k)]
    print(f"[leak_audit] {len(pending)} pending judge calls "
          f"({args.judge})")
    outs = pp.llm_query_many([{"prompt": pr, "model": args.judge}
                              for _, _, pr in pending])
    for (k, _, _), out in zip(pending, outs):
        try:
            parsed = pp.parse_json_reply(out)
            store.put(k, {
                "possesses": bool(parsed.get("possesses")),
                "legitimately_known": bool(parsed.get("legitimately_known")),
                "evidence": str(parsed.get("evidence", ""))[:400]})
        except ValueError:
            pp.tally_parse_skip()
    store.flush()

    for rkey, rec in roles.items():
        if not rec["valid"]:
            rec["leaked_t0"] = None
            continue
        verdicts = []
        for k, rk, _ in jobs:
            if rk != rkey:
                continue
            v = store.get(k)
            if v is not None:
                verdicts.append(dict(v, key=k))
        rec["verdicts"] = verdicts
        if not verdicts:
            rec["leaked_t0"] = None            # judgments pending
        elif any(v.get("legitimately_known") for v in verdicts):
            # fact-level property: the sample is invalid, not the answer
            rec["leaked_t0"] = None
            rec["reason"] = "judge_legitimately_known"
        else:
            rec["leaked_t0"] = any(v["possesses"] for v in verdicts)

    out_p = pp.aggregate_dir() / f"leak_audit_{args.batchtag}.json"
    pp.save_json_atomic(out_p, {
        "batchtag": args.batchtag, "generated_at": pp.now_str(),
        "judge": args.judge, "call_stats": dict(pp.CALL_STATS),
        "roles": roles})
    leaked = sum(1 for r in roles.values() if r.get("leaked_t0"))
    judged = sum(1 for r in roles.values()
                 if r.get("leaked_t0") is not None)
    print(f"[leak_audit] leaked_t0: {leaked}/{judged} judged -> {out_p}")
    return {"roles": roles}


if __name__ == "__main__":
    main()
