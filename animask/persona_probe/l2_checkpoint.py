"""L2 — snapshot-then-probe on the forked live context (ContextEcho-style).

Checkpoints sit at evenly spaced quantiles of each agent's acting-call
sequence in llm_calls_<tag>.jsonl (t0 = the agent's first acting call, the
self-baseline). A fork replays the checkpoint call's full messages + actual
response, appends one out-of-scene user probe, and asks the actor-grade model
to answer in character — a counterfactual bypass; the finished main run is
untouched.

Fixed probe battery per agent (reused verbatim at every checkpoint):
identity x2, relation <=2, private-knowledge boundary x2 (1 should-know,
1 should-NOT-know drawn from another role's knowledge atoms, TimeChara-style),
conditional <=3; each question gets paraphrase variants for variance.
Scoring: penalty rubric against the persona card (CoSER-style),
score = max(0, 100 - 5*sum(severity)); a 25% subsample is re-scored by the
cheap judge (agreement reported); self-baseline column compares each answer
with the t0 answer (0/2 consistent/contradict). Auxiliary mini-BFI
(InCharacter-style, 10 items) at the {0,50,100}% checkpoints only.

Old runs without llm_calls_<tag>.jsonl degrade to {"skipped":
"call_log_missing"}.
"""
from animask import persona_probe as pp
from animask.persona_probe import stats

ANS_TRUNC = 2000

_WRAP_EN = ("(Out-of-scene consistency check — this is not part of the "
            "scene. Answer as {name}, in first person, exactly as you are at "
            "this moment of the story. Do not mention this question or your "
            "answer in the ongoing scene. Answer briefly, 1-3 sentences.)\n"
            "Question: {q}")
_WRAP_ZH = ("(场外一致性检查——这不属于剧情。请以{name}的身份、第一人称、"
            "按你在故事此刻的状态回答。不要把这个问题或回答带进正在进行的剧情。"
            "用1-3句话简要回答。)\n问题:{q}")

_IDENTITY = [
    ("I1", "你是谁?用你自己的话,用几句话描述你自己。",
     "Who are you? Describe yourself in your own words, in a few sentences."),
    ("I2", "对你来说什么最重要?有什么事是你绝不会做的?",
     "What matters most to you, and what is something you would never do?"),
]

_PARA_EN = """Rewrite each numbered question below in exactly {k} different \
wording(s). Keep the meaning identical and stay in the same language as the \
question.
Output STRICT JSON only: {{"<qid>": ["rewrite 1", ...]}} with exactly {k} \
rewrite(s) per question.

{questions}"""

_PARA_ZH = """把下面每个带编号的问题各改写 {k} 个不同措辞的版本,含义必须完全\
一致,语言与原问题一致。
只输出严格 JSON:{{"<qid>": ["改写1", ...]}},每题恰好 {k} 个改写。

{questions}"""

_SCORE_EN = """You are grading persona fidelity with a penalty rubric.
Character: {name}
Persona atoms (id [type] statement):
{atoms}

The character was asked this out-of-scene question (expectation: \
{expectation}):
{question}

The character answered in first person:
{answer}

List every VIOLATION of the persona atoms in the answer: contradiction of an \
atom, breaking the fourth wall or speaking as an AI, or — if expectation is \
"should_not_know" — revealing knowledge this character cannot have; if \
expectation is "should_know", failing to show that knowledge is a violation. \
Legitimate in-story development of emotions, plans or relationships is NOT a \
violation.
severity: 1 = minor slip ... 5 = complete persona break.
Output STRICT JSON only: {{"violations": [{{"atom_id": "...", "severity": \
1-5, "evidence": "..."}}], "comment": "..."}} — empty list if none."""

_SCORE_ZH = """你在用扣分制评估 persona 保真度。
角色:{name}
persona 原子条目(id [类型] 陈述):
{atoms}

角色被问了这个场外问题(预期:{expectation}):
{question}

角色以第一人称回答:
{answer}

列出回答中对 persona 条目的所有「违背」:与条目矛盾、出戏(以 AI 口吻说话)、\
或当预期为 "should_not_know" 时暴露了该角色不可能拥有的知识;当预期为 \
"should_know" 时,答不出该知识也算违背。剧情内合法的情绪/计划/关系发展不算违背。
severity:1=轻微失足 … 5=完全人设崩坏。
只输出严格 JSON:{{"violations": [{{"atom_id": "...", "severity": 1-5, \
"evidence": "..."}}], "comment": "..."}},没有则给空列表。"""

_SELF_EN = """The same out-of-scene question was asked to the same role-played \
character at an earlier baseline point and again later.
Question: {question}
Baseline answer (t0): {a0}
Later answer: {a1}
Judge whether the later answer stays persona-consistent with the baseline \
(same values, self-concept, stance). Legitimate plot-state differences (new \
events, changed plans or emotions) do NOT count as contradiction.
Output STRICT JSON only: {{"consistent": true or false, "evidence": "..."}}"""

_SELF_ZH = """同一个场外问题在基线时点(t0)和之后又问了同一个被扮演的角色。
问题:{question}
基线回答(t0):{a0}
之后的回答:{a1}
判断之后的回答与基线是否在 persona 上保持一致(同样的价值观、自我认知、立场)。\
剧情状态的合法差异(新事件、计划或情绪变化)不算矛盾。
只输出严格 JSON:{{"consistent": true 或 false, "evidence": "..."}}"""

# paraphrased mini-BFI (2 items per Big-Five dimension; reverse-keyed flagged)
BFI_ITEMS = [
    ("E", False, "I am outgoing and talkative around other people.",
     "我在他人面前外向健谈。"),
    ("A", True, "I tend to find fault with other people.",
     "我常挑剔别人的毛病。"),
    ("C", False, "I do my tasks thoroughly and carefully.",
     "我做事全面细致。"),
    ("N", False, "I get nervous or anxious easily.",
     "我容易紧张焦虑。"),
    ("O", False, "I have a vivid imagination and enjoy new ideas.",
     "我想象力丰富,喜欢新想法。"),
    ("E", True, "I am usually quiet and reserved.",
     "我通常安静而内敛。"),
    ("A", False, "I am generally trusting and forgiving of others.",
     "我总体上信任并宽容他人。"),
    ("C", True, "I tend to be lazy or careless with my duties.",
     "我对分内之事常懒散马虎。"),
    ("N", True, "I stay calm and relaxed under pressure.",
     "我在压力下保持冷静放松。"),
    ("O", True, "I have little interest in art or abstract ideas.",
     "我对艺术或抽象观念兴趣不大。"),
]

_BFI_EN = ("(Out-of-scene questionnaire — not part of the scene. As {name}, "
           "rate how well each statement describes you RIGHT NOW: 1 = "
           "strongly disagree ... 5 = strongly agree. Output STRICT JSON "
           "only: {{\"1\": n, ..., \"10\": n}}.)\n{items}")
_BFI_ZH = ("(场外问卷——不属于剧情。请以{name}的身份,按你此刻的状态为每条陈述"
           "打分:1=非常不同意 … 5=非常同意。只输出严格 JSON:"
           "{{\"1\": n, ..., \"10\": n}}。)\n{items}")


def _wrap(name, q, lang):
    return pp.L(lang, _WRAP_ZH, _WRAP_EN).format(name=name, q=q)


def _atoms_block(atoms):
    return "\n".join(f"{a['id']} [{a['type']}] {a['statement']}"
                     for a in atoms)


# --------------------------------------------------------------------------
# probe battery (fixed across checkpoints)
# --------------------------------------------------------------------------
def base_questions(role, atoms_by_role, lang):
    """[{qid, text, expectation, atom_id, source_role}] — before paraphrase.
    identity 2 + relation <=2 + knowledge boundary 2 + conditional <=3."""
    atoms = atoms_by_role[role]
    qs = [{"qid": qid, "text": pp.L(lang, zh, en), "expectation": "none",
           "atom_id": None, "source_role": None}
          for qid, zh, en in _IDENTITY]
    rel = [a for a in atoms if a["type"] == "relation" and a["probe_question"]]
    for a in rel[:2]:
        qs.append({"qid": f"Q{a['id']}", "text": a["probe_question"],
                   "expectation": "none", "atom_id": a["id"],
                   "source_role": None})
    own_k = [a for a in atoms if a["type"] == "knowledge"
             and a["probe_question"]]
    if own_k:
        a = own_k[0]
        qs.append({"qid": f"Q{a['id']}", "text": a["probe_question"],
                   "expectation": "should_know", "atom_id": a["id"],
                   "source_role": None})
    foreign = None
    for other in sorted(atoms_by_role):
        if other == role:
            continue
        cand = [a for a in atoms_by_role[other] if a["type"] == "knowledge"
                and a["probe_question"]]
        if cand:
            foreign = (other, cand[0])
            break
    if foreign:
        other, a = foreign
        qs.append({"qid": f"F{a['id']}", "text": a["probe_question"],
                   "expectation": "should_not_know", "atom_id": None,
                   "source_role": other})
    cond = [a for a in atoms if a["type"] == "conditional"
            and a["probe_question"]]
    for a in cond[:3]:
        qs.append({"qid": f"Q{a['id']}", "text": a["probe_question"],
                   "expectation": "none", "atom_id": a["id"],
                   "source_role": None})
    return qs


def build_probe_set(pack, role, atoms_by_role, lang, n_variants,
                    judge_model, store):
    """Question battery with paraphrase variants; cached in the L2 store
    (1 LLM call per agent when n_variants > 1)."""
    key = f"pset|{role}"
    if store.has(key):
        return store.get(key)
    qs = base_questions(role, atoms_by_role, lang)
    for q in qs:
        q["variants"] = [q.pop("text")]
    if n_variants > 1 and qs:
        block = "\n".join(f"{q['qid']}: {q['variants'][0]}" for q in qs)
        prompt = pp.L(lang, _PARA_ZH, _PARA_EN).format(k=n_variants - 1,
                                                       questions=block)
        try:
            rew = pp.parse_json_reply(pp.llm_query(prompt, model=judge_model))
        except ValueError:
            rew = {}
        for q in qs:
            extra = rew.get(q["qid"], [])
            if isinstance(extra, str):
                extra = [extra]
            extra = [str(x) for x in extra][:n_variants - 1]
            while len(extra) < n_variants - 1:  # fallback: reuse original
                extra.append(q["variants"][0])
            q["variants"] += extra
    store.put(key, qs)
    store.flush()
    return qs


# --------------------------------------------------------------------------
# planning / execution
# --------------------------------------------------------------------------
def _agent_checkpoints(records, roles, n_checkpoints):
    """{role: {"calls": [...], "cps": [call idx], "fracs": {idx: frac}}}"""
    out = {}
    for role in roles:
        calls = pp.acting_calls(records, role)
        cps = pp.quantile_indices(len(calls), n_checkpoints)
        denom = max(1, len(calls) - 1)
        out[role] = {"calls": calls, "cps": cps,
                     "fracs": {ci: round(ci / denom, 4) for ci in cps}}
    return out


def _bfi_cps(cps):
    """Nearest checkpoints to the {0,50,100}% quantiles."""
    if not cps:
        return []
    lo, hi = cps[0], cps[-1]
    mid = min(cps, key=lambda c: abs(c - (lo + hi) / 2))
    return sorted({lo, mid, hi})


def plan_l2(pack, tag, roles, atoms_by_role, n_checkpoints=5, n_variants=2,
            second_frac=0.25, bfi=True) -> dict:
    records = pp.load_call_log(pack, tag)
    if records is None:
        return {"skipped": "call_log_missing"}
    ag = _agent_checkpoints(records, roles, n_checkpoints)
    plan = {"per_agent": {}, "skipped_agents": {}}
    total = 0
    for role in roles:
        calls, cps = ag[role]["calls"], ag[role]["cps"]
        if not calls:
            plan["skipped_agents"][role] = "no_acting_calls"
            continue
        # dry-run before atomization: assume the full 9-question battery
        n_q = len(base_questions(role, atoms_by_role, lang="en")) \
            if atoms_by_role.get(role) else 9
        n_ans = len(cps) * n_q * n_variants
        n_self = max(0, (len(cps) - 1)) * n_q * n_variants
        n_second = round(second_frac * n_ans)
        n_para = 1 if n_variants > 1 else 0
        n_bfi = len(_bfi_cps(cps)) if bfi else 0
        n = n_para + n_ans * 2 + n_self + n_second + n_bfi
        plan["per_agent"][role] = {
            "acting_calls": len(calls), "checkpoints": cps,
            "n_questions": n_q, "probe_answers": n_ans,
            "normative_scores": n_ans, "self_baseline": n_self,
            "second_ratings": n_second, "bfi_calls": n_bfi,
            "paraphrase_calls": n_para, "total_calls": n}
        total += n
    plan["total_calls"] = total
    return plan


def run_l2(pack, tag, roles, atoms_by_role, lang,
           actor_model=pp.GPT, main_judge=pp.SONNET, second_judge=pp.MINI,
           n_checkpoints=5, n_variants=2, seed=0, second_frac=0.25,
           bfi=True, flush_every=50, chunk=25) -> dict:
    records = pp.load_call_log(pack, tag)
    if records is None:
        return {"skipped": "call_log_missing"}
    store = pp.ResumableStore(pp.layer_path(pack, tag, "l2"),
                              flush_every=flush_every)
    store.set_meta("params", {"actor_model": actor_model,
                              "main_judge": main_judge,
                              "second_judge": second_judge,
                              "n_checkpoints": n_checkpoints,
                              "n_variants": n_variants, "seed": seed,
                              "lang": lang})
    ag = _agent_checkpoints(records, roles, n_checkpoints)
    names = {r: pp.role_display_name(pack, r) for r in roles}
    skipped_agents = {r: "no_acting_calls" for r in roles
                      if not ag[r]["calls"]}
    active = [r for r in roles if ag[r]["calls"]]
    psets = {r: build_probe_set(pack, r, atoms_by_role, lang, n_variants,
                                main_judge, store) for r in active}

    def _chat_phase(jobs):
        """jobs: [(key, messages)] -> store answers; chunked + resumable."""
        pending = [(k, m) for k, m in jobs if not store.has(k)]
        for start in range(0, len(pending), chunk):
            group = pending[start:start + chunk]
            outs = pp.llm_chat_many([{"messages": m, "model": actor_model,
                                      "temperature": 0.0} for _, m in group])
            for (k, _), out in zip(group, outs):
                if isinstance(out, str) and out.startswith("<<ERROR"):
                    continue  # retried on next invocation
                store.put(k, out)
            store.flush()

    def _judge_phase(jobs, model):
        """jobs: [(key, prompt, postprocess)] -> store parsed values."""
        pending = [(k, p, f) for k, p, f in jobs if not store.has(k)]
        for start in range(0, len(pending), chunk):
            group = pending[start:start + chunk]
            outs = pp.llm_query_many([{"prompt": p, "model": model}
                                      for _, p, _ in group])
            for (k, _, post), out in zip(group, outs):
                try:
                    store.put(k, post(out))
                except ValueError:
                    # unparseable judge reply: item stays missing from the
                    # store; counted so `complete` goes false and a re-run
                    # retries it instead of silently losing the score
                    pp.tally_parse_skip()
                    continue
            store.flush()

    # --- phase 1: probe answers on the forked context ---
    ans_jobs = []
    for role in active:
        calls, cps = ag[role]["calls"], ag[role]["cps"]
        for ci in cps:
            rec = calls[ci]
            base = list(rec["messages"]) + [
                {"role": "assistant", "content": rec["response"]}]
            for q in psets[role]:
                for vi, variant in enumerate(q["variants"]):
                    key = f"ans|{role}|{ci}|{q['qid']}|{vi}"
                    msgs = base + [{"role": "user",
                                    "content": _wrap(names[role], variant,
                                                     lang)}]
                    ans_jobs.append((key, msgs))
    _chat_phase(ans_jobs)

    # --- phase 2: normative penalty scoring ---
    def _post_score(out):
        parsed = pp.parse_json_reply(out)
        viols = parsed.get("violations", []) if isinstance(parsed, dict) \
            else []
        viols = [v for v in viols if isinstance(v, dict)]
        total = sum(max(1, min(5, int(v.get("severity", 1)))) for v in viols)
        return {"violations": viols, "penalty": total,
                "score": max(0, 100 - 5 * total)}

    score_jobs = []
    for role in active:
        for ci in ag[role]["cps"]:
            for q in psets[role]:
                for vi in range(len(q["variants"])):
                    akey = f"ans|{role}|{ci}|{q['qid']}|{vi}"
                    if not store.has(akey):
                        continue
                    prompt = pp.L(lang, _SCORE_ZH, _SCORE_EN).format(
                        name=names[role],
                        atoms=_atoms_block(atoms_by_role[role]),
                        expectation=q["expectation"],
                        question=q["variants"][vi],
                        answer=str(store.get(akey))[:ANS_TRUNC])
                    score_jobs.append((f"score|{role}|{ci}|{q['qid']}|{vi}",
                                       prompt, _post_score))
    _judge_phase(score_jobs, main_judge)

    # --- phase 3: self-baseline vs t0 ---
    def _post_self(out):
        parsed = pp.parse_json_reply(out)
        cons = bool(parsed.get("consistent", True)) \
            if isinstance(parsed, dict) else True
        return {"consistent": cons, "penalty": 0 if cons else 2}

    self_jobs = []
    for role in active:
        cps = ag[role]["cps"]
        t0 = cps[0]
        for ci in cps[1:]:
            for q in psets[role]:
                for vi in range(len(q["variants"])):
                    a0k = f"ans|{role}|{t0}|{q['qid']}|{vi}"
                    a1k = f"ans|{role}|{ci}|{q['qid']}|{vi}"
                    if not (store.has(a0k) and store.has(a1k)):
                        continue
                    prompt = pp.L(lang, _SELF_ZH, _SELF_EN).format(
                        question=q["variants"][vi],
                        a0=str(store.get(a0k))[:ANS_TRUNC],
                        a1=str(store.get(a1k))[:ANS_TRUNC])
                    self_jobs.append((f"self|{role}|{ci}|{q['qid']}|{vi}",
                                      prompt, _post_self))
    _judge_phase(self_jobs, main_judge)

    # --- phase 4: 25% second rating by the cheap judge ---
    scored_keys = sorted(k for k in store.data["items"]
                         if k.startswith("score|"))
    sample = pp.seeded_sample(scored_keys, second_frac, seed)
    second_jobs = []
    for skey in sample:
        _, role, ci, qid, vi = skey.split("|")
        akey = f"ans|{role}|{ci}|{qid}|{vi}"
        q = next((x for x in psets.get(role, []) if x["qid"] == qid), None)
        if q is None or not store.has(akey):
            continue
        prompt = pp.L(lang, _SCORE_ZH, _SCORE_EN).format(
            name=names[role], atoms=_atoms_block(atoms_by_role[role]),
            expectation=q["expectation"], question=q["variants"][int(vi)],
            answer=str(store.get(akey))[:ANS_TRUNC])
        second_jobs.append((f"second|{role}|{ci}|{qid}|{vi}", prompt,
                            _post_score))
    _judge_phase(second_jobs, second_judge)

    # --- phase 5: mini-BFI at {0,50,100}% ---
    if bfi:
        bfi_jobs = []
        for role in active:
            calls = ag[role]["calls"]
            items = "\n".join(
                f"{i + 1}. {pp.L(lang, zh, en)}"
                for i, (_, _, en, zh) in enumerate(BFI_ITEMS))
            for ci in _bfi_cps(ag[role]["cps"]):
                rec = calls[ci]
                msgs = list(rec["messages"]) + [
                    {"role": "assistant", "content": rec["response"]},
                    {"role": "user",
                     "content": pp.L(lang, _BFI_ZH, _BFI_EN).format(
                         name=names[role], items=items)}]
                bfi_jobs.append((f"bfi|{role}|{ci}", msgs))
        _chat_phase(bfi_jobs)
    store.flush()

    # --- assemble summary ---
    per_agent = {}
    for role in active:
        cps, fracs = ag[role]["cps"], ag[role]["fracs"]
        t0 = cps[0]
        checkpoints = []
        for ci in cps:
            n_scores, self_flags, viols = [], [], 0
            for q in psets[role]:
                for vi in range(len(q["variants"])):
                    sc = store.get(f"score|{role}|{ci}|{q['qid']}|{vi}")
                    if sc is not None:
                        n_scores.append(sc["score"])
                        viols += len(sc["violations"])
                    sf = store.get(f"self|{role}|{ci}|{q['qid']}|{vi}")
                    if sf is not None:
                        self_flags.append(sf["consistent"])
            checkpoints.append({
                "call_index": ci, "frac": fracs[ci],
                "normative_score": stats.mean(n_scores),
                "self_score": (100 * sum(self_flags) / len(self_flags))
                if self_flags else (100.0 if ci == t0 else None),
                "is_t0": ci == t0,
                "n_scored": len(n_scores), "violations_total": viols})
        bfi_rows = []
        for ci in _bfi_cps(cps) if bfi else []:
            raw = store.get(f"bfi|{role}|{ci}")
            if raw is None:
                continue
            try:
                parsed = pp.parse_json_reply(raw)
                vals = {int(k): max(1, min(5, int(v)))
                        for k, v in parsed.items()}
                dims = {}
                for d in ("O", "C", "E", "A", "N"):
                    xs = [(6 - vals[i + 1]) if rev else vals[i + 1]
                          for i, (dim, rev, _, _) in enumerate(BFI_ITEMS)
                          if dim == d and (i + 1) in vals]
                    dims[d] = stats.mean(xs)
                bfi_rows.append({"call_index": ci, "frac": fracs[ci],
                                 "raw": vals, "dims": dims})
            except (ValueError, KeyError, TypeError):
                continue
        xs = [c["frac"] for c in checkpoints]
        per_agent[role] = {
            "checkpoints": checkpoints,
            "stats": {
                "normative": stats.series_stats(
                    xs, [c["normative_score"] for c in checkpoints]),
                "self": stats.series_stats(
                    [c["frac"] for c in checkpoints if not c["is_t0"]],
                    [c["self_score"] for c in checkpoints
                     if not c["is_t0"]])},
            "bfi": bfi_rows,
            "probe_set": psets[role]}

    # --- judge agreement (main vs second) ---
    m_bin, s_bin, mads = [], [], []
    for k in store.data["items"]:
        if not k.startswith("second|"):
            continue
        skey = "score|" + k.split("|", 1)[1]
        main, second = store.get(skey), store.get(k)
        if main is None or second is None:
            continue
        m_bin.append(bool(main["violations"]))
        s_bin.append(bool(second["violations"]))
        mads.append(abs(main["score"] - second["score"]))
    judge_agreement = {
        "n_pairs": len(m_bin),
        "binary_agreement": stats.percent_agreement(m_bin, s_bin),
        "cohens_kappa": stats.cohens_kappa(m_bin, s_bin),
        "score_mad": stats.mean(mads),
        "main_judge": main_judge, "second_judge": second_judge}

    store.set_meta("summary_generated_at", pp.now_str())
    store.flush()
    return {"per_agent": per_agent, "skipped_agents": skipped_agents,
            "judge_agreement": judge_agreement, "lang": lang,
            "actor_model": actor_model}
