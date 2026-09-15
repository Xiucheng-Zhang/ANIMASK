"""L3 — persona-ablation attribution arm (sampled checkpoints).

At the {50,100}% checkpoints of each agent's acting-call sequence, the
checkpoint call's messages are copied and the card material (profile ->
neutral placeholder; composed relation lines and card motivation -> removed,
matching the run-level P0 definition in RPAgent._apply_persona_level);
replaying the acting instruction yields the model-default arm.
The same probe battery (base variants) runs on the ablated context.

Attribution quantities:
- 2AFC (position swap x2): does the agent's actual NEXT in-simulation
  behavior resemble the persona arm (the actual logged response at the
  checkpoint) or the default arm (the ablated replay) more?
- probe-score gap between arms (ablated normative score vs the L2 normative
  score at the nearest checkpoint) + Cohen's d over question-level scores.

Checkpoint indices are capped at len(calls)-2 so a real "next behavior"
always exists (deviation from the literal {50,100}% quantiles; see README).
"""
from animask import persona_probe as pp
from animask.persona_probe import atomize, stats
from animask.persona_probe.l2_checkpoint import (ANS_TRUNC, _atoms_block, _wrap,
                                         base_questions, _SCORE_EN, _SCORE_ZH)

REF_TRUNC = 1600

PLACEHOLDER = {"zh": "你是这个故事中的一名角色。",
               "en": "You are a character in this story."}

_AFC_EN = """You are comparing role-play outputs.
REFERENCE — the character's actual next behavior in the simulation:
{reference}

Candidate A:
{a}

Candidate B:
{b}

Which candidate does the REFERENCE resemble more in voice, values and \
behavioral tendency? Judge style and disposition, not plot overlap. The \
candidates may differ in length; length itself must NOT affect your choice.
Output STRICT JSON only: {{"choice": "A" or "B", "reason": "..."}}"""

_AFC_ZH = """你在比较角色扮演输出。
参照——该角色在模拟中的真实下一步行为:
{reference}

候选 A:
{a}

候选 B:
{b}

参照在口吻、价值观和行为倾向上更像哪个候选?只判风格与性格倾向,不判情节重合。\
两个候选长度可能不同;长度本身绝不能影响你的选择。
只输出严格 JSON:{{"choice": "A" 或 "B", "reason": "..."}}"""


def _l3_checkpoints(n_calls: int, n_checkpoints: int) -> list:
    """Fracs (1/n .. 1.0) of the acting sequence, index-capped at n-2 so a
    next real behavior exists. Requires >=2 acting calls."""
    if n_calls < 2:
        return []
    fracs = [(i + 1) / n_checkpoints for i in range(n_checkpoints)]
    return sorted({min(round(f * (n_calls - 1)), n_calls - 2)
                   for f in fracs})


def _persona_blocks(role_info: dict, lang: str):
    """(profile, other card-material strings) as they appear verbatim in
    prompts. Covers the full run-level P0 definition — profile, the
    composed relation lines (RPAgent.search_relation format), and the card
    motivation — so the single-step and whole-run ablations measure the
    same construct. Keep in sync with RPAgent's persona-level switch."""
    profile = role_info.get("profile", "")
    blocks = []
    for v in (role_info.get("relation") or {}).values():
        if not isinstance(v, dict):
            continue
        rtext = ",".join(v.get("relation") or [])
        detail = v.get("detail", "")
        blocks.append(f"This is your {rtext}. {detail}")
        blocks.append(f"这是你的{rtext}. {detail}")
        if detail:
            blocks.append(detail)
    if role_info.get("motivation"):
        blocks.append(role_info["motivation"])
    return profile, [b for b in blocks if b]


def _ablate(messages: list, profile: str, lang: str, extra_blocks=None):
    """(ablated messages, found_flag) — the verbatim profile block becomes
    the placeholder; relation/motivation blocks are removed. found tracks
    the profile only (an un-found profile invalidates the arm)."""
    found = False
    out = []
    for m in messages:
        content = str(m.get("content", ""))
        if profile and profile in content:
            content = content.replace(profile, PLACEHOLDER.get(lang,
                                                               PLACEHOLDER["en"]))
            found = True
        for b in sorted(extra_blocks or [], key=len, reverse=True):
            if b in content:
                content = content.replace(b, "")
        out.append({"role": m.get("role", "user"), "content": content})
    return out, found


def plan_l3(pack, tag, roles, atoms_by_role, n_checkpoints=2) -> dict:
    records = pp.load_call_log(pack, tag)
    if records is None:
        return {"skipped": "call_log_missing"}
    plan = {"per_agent": {}, "skipped_agents": {}}
    total = 0
    for role in roles:
        calls = pp.acting_calls(records, role)
        cps = _l3_checkpoints(len(calls), n_checkpoints)
        if not cps:
            plan["skipped_agents"][role] = "insufficient_acting_calls"
            continue
        n_q = len(base_questions(role, atoms_by_role, lang="en")) \
            if atoms_by_role.get(role) else 9
        n = len(cps) * (1 + 2 + n_q * 2)  # act + 2AFC x2 + probes + scores
        plan["per_agent"][role] = {"acting_calls": len(calls),
                                   "checkpoints": cps, "n_questions": n_q,
                                   "total_calls": n}
        total += n
    plan["total_calls"] = total
    return plan


def run_l3(pack, tag, roles, atoms_by_role, lang,
           actor_model=pp.GPT, main_judge=pp.SONNET, n_checkpoints=2,
           flush_every=50, chunk=25) -> dict:
    records = pp.load_call_log(pack, tag)
    if records is None:
        return {"skipped": "call_log_missing"}
    store = pp.ResumableStore(pp.layer_path(pack, tag, "l3"),
                              flush_every=flush_every)
    store.set_meta("params", {"actor_model": actor_model,
                              "main_judge": main_judge,
                              "n_checkpoints": n_checkpoints, "lang": lang})
    names = {r: pp.role_display_name(pack, r) for r in roles}
    skipped_agents = {}
    agents = {}
    for role in roles:
        calls = pp.acting_calls(records, role)
        cps = _l3_checkpoints(len(calls), n_checkpoints)
        if not cps:
            skipped_agents[role] = "insufficient_acting_calls"
            continue
        try:
            role_info = pp.load_json(atomize.role_info_path(pack, role))
        except (OSError, ValueError):
            role_info = {}
        profile, blocks = _persona_blocks(role_info, lang)
        agents[role] = {"calls": calls, "cps": cps, "profile": profile,
                        "blocks": blocks,
                        "questions": base_questions(role, atoms_by_role,
                                                    lang)}

    def _chat_phase(jobs):
        pending = [j for j in jobs if not store.has(j[0])]
        for start in range(0, len(pending), chunk):
            group = pending[start:start + chunk]
            outs = pp.llm_chat_many([{"messages": m, "model": actor_model,
                                      "temperature": t}
                                     for _, m, t in group])
            for (k, _, _), out in zip(group, outs):
                if isinstance(out, str) and out.startswith("<<ERROR"):
                    continue
                store.put(k, out)
            store.flush()

    def _judge_phase(jobs, model):
        pending = [j for j in jobs if not store.has(j[0])]
        for start in range(0, len(pending), chunk):
            group = pending[start:start + chunk]
            outs = pp.llm_query_many([{"prompt": p, "model": model}
                                      for _, p, _ in group])
            for (k, _, post), out in zip(group, outs):
                try:
                    store.put(k, post(out))
                except ValueError:
                    # transport-ok but unparseable: not stored, so a re-run
                    # retries it — but only if `complete` reflects the skip
                    # (one lost AFC vote of 8 made persona_inert unreachable
                    # with no error anywhere)
                    pp.tally_parse_skip()
                    continue
            store.flush()

    # --- phase 1: default-arm acting replay on the ablated context ---
    ablation_notes = {}
    act_jobs = []
    for role, a in agents.items():
        for ci in a["cps"]:
            rec = a["calls"][ci]
            ablated, found = _ablate(rec["messages"], a["profile"], lang,
                                      a["blocks"])
            if not found:
                ablation_notes[f"{role}|{ci}"] = "profile_not_found_in_context"
                continue
            temp = rec.get("temperature", 0.7)
            act_jobs.append((f"act|{role}|{ci}", ablated,
                             temp if isinstance(temp, (int, float)) else 0.7))
    _chat_phase(act_jobs)

    # --- phase 2: 2AFC (position swap x2) + probes on the ablated context ---
    def _post_afc(out):
        parsed = pp.parse_json_reply(out)
        choice = str(parsed.get("choice", "")).strip().upper()[:1] \
            if isinstance(parsed, dict) else ""
        if choice not in ("A", "B"):
            raise ValueError("2AFC judge gave no A/B choice")
        return {"choice": choice}

    afc_jobs, probe_jobs = [], []
    for role, a in agents.items():
        for ci in a["cps"]:
            akey = f"act|{role}|{ci}"
            if not store.has(akey):
                continue
            rec = a["calls"][ci]
            default_out = pp.extract_detail(str(store.get(akey)))
            persona_out = pp.extract_detail(rec["response"])
            reference = pp.extract_detail(a["calls"][ci + 1]["response"])
            for order in (0, 1):
                cand_a, cand_b = ((persona_out, default_out) if order == 0
                                  else (default_out, persona_out))
                prompt = pp.L(lang, _AFC_ZH, _AFC_EN).format(
                    reference=reference[:REF_TRUNC],
                    a=cand_a[:REF_TRUNC], b=cand_b[:REF_TRUNC])
                afc_jobs.append((f"afc|{role}|{ci}|{order}", prompt,
                                 _post_afc))
            ablated, _ = _ablate(rec["messages"], a["profile"], lang,
                                 a["blocks"])
            base = ablated + [{"role": "assistant",
                               "content": str(store.get(akey))}]
            for q in a["questions"]:
                msgs = base + [{"role": "user",
                                "content": _wrap(names[role], q["text"],
                                                 lang)}]
                probe_jobs.append((f"probe|{role}|{ci}|{q['qid']}", msgs,
                                   0.0))
    _judge_phase(afc_jobs, main_judge)
    _chat_phase(probe_jobs)

    # --- phase 3: penalty-score the ablated-arm probe answers ---
    def _post_score(out):
        parsed = pp.parse_json_reply(out)
        viols = parsed.get("violations", []) if isinstance(parsed, dict) \
            else []
        viols = [v for v in viols if isinstance(v, dict)]
        total = sum(max(1, min(5, int(v.get("severity", 1)))) for v in viols)
        return {"violations": viols, "penalty": total,
                "score": max(0, 100 - 5 * total)}

    pscore_jobs = []
    for role, a in agents.items():
        for ci in a["cps"]:
            for q in a["questions"]:
                pkey = f"probe|{role}|{ci}|{q['qid']}"
                if not store.has(pkey):
                    continue
                prompt = pp.L(lang, _SCORE_ZH, _SCORE_EN).format(
                    name=names[role],
                    atoms=_atoms_block(atoms_by_role[role]),
                    expectation=q["expectation"], question=q["text"],
                    answer=str(store.get(pkey))[:ANS_TRUNC])
                pscore_jobs.append((f"pscore|{role}|{ci}|{q['qid']}", prompt,
                                    _post_score))
    _judge_phase(pscore_jobs, main_judge)
    store.flush()

    # --- persona-arm question scores from the L2 layer file, if present ---
    l2_items = {}
    l2_file = pp.layer_path(pack, tag, "l2")
    if l2_file.exists():
        try:
            l2_items = pp.load_json(l2_file).get("items", {})
        except (OSError, ValueError):
            l2_items = {}

    def _l2_scores_near(role, ci):
        """(nearest l2 call index, question-level scores) or (None, [])."""
        indices = sorted({int(k.split("|")[2]) for k in l2_items
                          if k.startswith(f"score|{role}|")})
        if not indices:
            return None, []
        near = min(indices, key=lambda x: abs(x - ci))
        scores = [v["score"] for k, v in l2_items.items()
                  if k.startswith(f"score|{role}|{near}|")
                  and isinstance(v, dict) and "score" in v]
        return near, scores

    # --- assemble summary ---
    per_agent = {}
    for role, a in agents.items():
        rows = []
        votes = persona_votes = 0
        denom = max(1, len(a["calls"]) - 1)
        for ci in a["cps"]:
            note = ablation_notes.get(f"{role}|{ci}")
            row = {"call_index": ci, "frac": round(ci / denom, 4),
                   "next_call_index": ci + 1, "note": note}
            if note:
                rows.append(row)
                continue
            picks = []
            for order in (0, 1):
                afc = store.get(f"afc|{role}|{ci}|{order}")
                if afc is None:
                    continue
                persona_is = "A" if order == 0 else "B"
                picks.append(afc["choice"] == persona_is)
            row["persona_votes"] = sum(picks)
            row["votes"] = len(picks)
            row["order_consistent"] = (len(picks) == 2
                                       and picks[0] == picks[1])
            votes += len(picks)
            persona_votes += sum(picks)
            abl_scores = [store.get(f"pscore|{role}|{ci}|{q['qid']}")["score"]
                          for q in a["questions"]
                          if store.get(f"pscore|{role}|{ci}|{q['qid']}")
                          is not None]
            row["ablated_probe_score"] = stats.mean(abl_scores)
            near, l2_scores = _l2_scores_near(role, ci)
            row["l2_call_index"] = near
            row["persona_probe_score"] = stats.mean(l2_scores)
            row["drift_gap"] = (row["persona_probe_score"]
                                - row["ablated_probe_score"]
                                if row["persona_probe_score"] is not None
                                and row["ablated_probe_score"] is not None
                                else None)
            row["cohens_d"] = stats.cohens_d(l2_scores, abl_scores)
            akey = f"act|{role}|{ci}"
            row["len_persona"] = len(pp.extract_detail(
                a["calls"][ci]["response"]))
            row["len_default"] = (len(pp.extract_detail(
                str(store.get(akey)))) if store.has(akey) else None)
            rows.append(row)
        per_agent[role] = {
            "checkpoints": rows,
            "persona_rate": (persona_votes / votes) if votes else None,
            "persona_votes": persona_votes, "votes": votes,
            "mean_drift_gap": stats.mean([r["drift_gap"] for r in rows
                                          if r.get("drift_gap")
                                          is not None])}

    store.set_meta("summary_generated_at", pp.now_str())
    store.flush()
    return {"per_agent": per_agent, "skipped_agents": skipped_agents,
            "ablation_notes": ablation_notes, "lang": lang,
            "actor_model": actor_model, "main_judge": main_judge,
            "l2_available_for_gap": bool(l2_items)}
