"""L1 — per-utterance persona ECG (cheap, full-run coverage).

Timeline = the agent's n-th utterance in the transcript (finer than rounds,
no round-boundary parsing). An LLM-NLI judge (the cheap-tier MINI model,
8 utterances per call) labels each utterance x applicable atom as
entail/neutral/contradict;
utterance score = 1 - contradicted/applicable (applicable = non-neutral).
Pure-Python behavioral features ride along (length, punctuation density,
quote ratio, lexical diversity — a homogenization proxy). A seeded 10%
subsample is re-judged by the strong judge; percent agreement + Cohen's kappa
are reported.
"""
import math
import re

from animask import persona_probe as pp
from animask.persona_probe import stats

UTT_TRUNC = 1200
VERDICTS = ("entail", "neutral", "contradict")

_PROMPT_EN = """You are a strict NLI judge for persona consistency in a \
role-play simulation.
Character: {name}
Persona atoms (id [type] statement):
{atoms}

Below are numbered utterances by this character (they may mix [inner \
thoughts], (stage directions) and dialogue).
For EACH utterance and EACH atom, judge:
- "entail": the utterance actively expresses or conforms to the atom
- "neutral": the atom is not engaged by this utterance
- "contradict": the utterance violates the atom
Judge persona traits only: legitimate in-story changes of emotion, plans or \
knowledge are NOT contradictions unless the stated disposition itself is \
violated.
Output STRICT JSON only: {{"<utterance_number>": {{"<atom_id>": \
"entail|neutral|contradict", ...}}, ...}} covering every utterance and every \
atom.

UTTERANCES:
{utterances}"""

_PROMPT_ZH = """你是一名严格的 NLI 裁判,评估角色扮演模拟中的 persona 一致性。
角色:{name}
persona 原子条目(id [类型] 陈述):
{atoms}

下面是该角色的若干条带编号的发言(可能混合[内心独白]、(动作描写)和台词)。
对每条发言 × 每个原子条目,判定:
- "entail":发言积极体现/符合该条目
- "neutral":该条目与这条发言无关
- "contradict":发言违背该条目
只判 persona 特质:剧情内合法的情绪/计划/知识变化不算 contradict,除非违背了\
条目所述的性格倾向本身。
只输出严格 JSON:{{"<发言编号>": {{"<原子id>": "entail|neutral|contradict", \
...}}, ...}},覆盖每条发言和每个条目。

发言:
{utterances}"""


# --------------------------------------------------------------------------
# transcript parsing + behavioral features (no LLM)
# --------------------------------------------------------------------------
def agent_utterances(transcript: list, role_code: str) -> list:
    """[{n, t_index, text}] — the agent's utterance sequence."""
    out = []
    for i, entry in enumerate(transcript):
        if entry.get("actor_type") == "role" and entry.get("code") == role_code:
            out.append({"n": len(out), "t_index": i,
                        "text": str(entry.get("text", ""))})
    return out


def subsample(utts: list, max_n) -> list:
    """Evenly subsample to max_n items (keeping first and last)."""
    if not max_n or max_n <= 0 or len(utts) <= max_n:
        return utts
    if max_n == 1:
        return [utts[-1]]
    idx = sorted({round(i * (len(utts) - 1) / (max_n - 1))
                  for i in range(max_n)})
    return [utts[i] for i in idx]


_QUOTE_PAIRS = [('"', '"'), ("“", "”"), ("「", "」"),
                ("『", "』")]
_PUNCT = set(".,;:!?…—-()[]{}\"'“”‘’、。,;:!?《》「」『』()·")


def behavior_features(text: str, lang: str) -> dict:
    n_chars = len(text)
    if lang == "zh":
        tokens = [c for c in text if c.isalnum()]
    else:
        tokens = re.findall(r"\w+", text.lower())
    n_tok = len(tokens)
    ttr = len(set(tokens)) / n_tok if n_tok else 0.0
    punct = sum(1 for c in text if c in _PUNCT)
    quoted = 0
    for op, cl in _QUOTE_PAIRS:
        if op == cl:
            spans = re.findall(re.escape(op) + r"(.*?)" + re.escape(cl),
                               text, re.S)
        else:
            spans = re.findall(re.escape(op) + r"(.*?)" + re.escape(cl),
                               text, re.S)
        quoted += sum(len(s) for s in spans)
    return {"n_chars": n_chars, "n_tokens": n_tok,
            "ttr": round(ttr, 4),
            "punct_density": round(punct / n_chars, 4) if n_chars else 0.0,
            "quote_ratio": round(min(1.0, quoted / n_chars), 4)
            if n_chars else 0.0}


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------
def _atoms_block(atoms: list) -> str:
    return "\n".join(f"{a['id']} [{a['type']}] {a['statement']}"
                     for a in atoms)


def _utts_block(batch: list) -> str:
    return "\n\n".join(f"#{u['n']}: {u['text'][:UTT_TRUNC]}" for u in batch)


def _build_prompt(name: str, atoms: list, batch: list, lang: str) -> str:
    tmpl = _PROMPT_ZH if lang == "zh" else _PROMPT_EN
    return tmpl.format(name=name, atoms=_atoms_block(atoms),
                       utterances=_utts_block(batch))


def _norm_verdict(v) -> str:
    v = str(v).strip().lower()
    for cand in VERDICTS:
        if v.startswith(cand[0]):
            return cand
    return "neutral"


def _parse_batch_reply(reply: str, batch: list, atoms: list) -> dict:
    """{str(n): {atom_id: verdict}} for the utterances of this batch."""
    parsed = pp.parse_json_reply(reply)
    if not isinstance(parsed, dict):
        raise ValueError("L1 judge reply is not an object")
    # judges echo the prompt's "#0:" labels as keys ("#0", "utt 0", ...) —
    # normalize every reply key to its digit core before lookup
    by_n = {}
    for k, v in parsed.items():
        nk = re.sub(r"\D", "", str(k))
        if nk and isinstance(v, dict):
            by_n.setdefault(nk, v)
    atom_ids = [a["id"] for a in atoms]
    out = {}
    matched = 0
    for u in batch:
        row = by_n.get(str(u["n"]), {})
        matched += bool(row)
        out[str(u["n"])] = {aid: _norm_verdict(row.get(aid, "neutral"))
                            for aid in atom_ids}
    if batch and not matched:
        # a silent all-neutral fill would poison the series — fail loudly so
        # the batch lands in `failures` and is retried on the next run
        raise ValueError(f"no utterance keys matched reply keys "
                         f"{list(parsed)[:6]}")
    return out


def _score(verdict_row: dict):
    n_e = sum(1 for v in verdict_row.values() if v == "entail")
    n_c = sum(1 for v in verdict_row.values() if v == "contradict")
    applicable = n_e + n_c
    score = (1 - n_c / applicable) if applicable else None
    return score, n_e, n_c, applicable


def _batches(pack, tag, roles, max_utt, batch_size):
    """[(role, batch_index, [utt, ...])] for all roles with utterances."""
    transcript = pp.load_json(pp.transcript_path(pack, tag))
    plan = []
    per_agent = {}
    for role in roles:
        utts = subsample(agent_utterances(transcript, role), max_utt)
        per_agent[role] = utts
        for bi in range(0, len(utts), batch_size):
            plan.append((role, bi // batch_size, utts[bi:bi + batch_size]))
    return plan, per_agent


def plan_l1(pack, tag, roles, max_utt=None, batch_size=8,
            calib_frac=0.10) -> dict:
    """Dry-run: parse + group only, report planned LLM calls."""
    batches, per_agent = _batches(pack, tag, roles, max_utt, batch_size)
    n_utt = sum(len(v) for v in per_agent.values())
    calib_utts = max(1, round(calib_frac * n_utt)) if n_utt else 0
    calib_batches = sum(
        math.ceil(min(len(per_agent[r]), max(1, round(
            calib_frac * len(per_agent[r])))) / batch_size)
        for r in roles if per_agent.get(r))
    return {"n_utterances": n_utt,
            "per_agent_utterances": {r: len(v) for r, v in per_agent.items()},
            "judge_batches": len(batches),
            "calib_utterances": calib_utts,
            "calib_batches": calib_batches,
            "total_calls": len(batches) + calib_batches}


def run_l1(pack, tag, roles, atoms_by_role, lang,
           l1_judge=pp.MINI, calib_judge=pp.SONNET, batch_size=8,
           max_utt=None, seed=0, calib_frac=0.10, flush_every=50,
           chunk=25) -> dict:
    store = pp.ResumableStore(pp.layer_path(pack, tag, "l1"),
                              flush_every=flush_every)
    store.set_meta("params", {"l1_judge": l1_judge, "calib_judge": calib_judge,
                              "batch_size": batch_size, "max_utt": max_utt,
                              "seed": seed, "lang": lang})
    batches, per_agent = _batches(pack, tag, roles, max_utt, batch_size)
    names = {r: pp.role_display_name(pack, r) for r in roles}
    failures = []

    # --- primary judging (resumable per batch) ---
    def _run_batches(pending, judge, key_prefix):
        for start in range(0, len(pending), chunk):
            group = pending[start:start + chunk]
            jobs = [{"prompt": _build_prompt(names[r],
                                            atoms_by_role[r], b, lang),
                     "model": judge} for (r, bi, b) in group]
            replies = pp.llm_query_many(jobs)
            for (r, bi, b), reply in zip(group, replies):
                key = f"{key_prefix}|{r}|{bi}"
                try:
                    store.put(key, {"utts": [u["n"] for u in b],
                                    "verdicts": _parse_batch_reply(
                                        reply, b, atoms_by_role[r])})
                except ValueError as e:
                    failures.append({"key": key, "error": str(e)[:200]})
                    # not stored -> resumable, but only if `complete` knows
                    pp.tally_parse_skip()
            store.flush()

    pending = [(r, bi, b) for (r, bi, b) in batches
               if not store.has(f"batch|{r}|{bi}")]
    _run_batches(pending, l1_judge, "batch")

    # --- calibration subsample re-judged by the strong judge ---
    calib_batches = []
    for role in roles:
        judged = [u for u in per_agent[role]
                  if any(store.has(f"batch|{role}|{bi}")
                         and str(u["n"]) in store.get(
                             f"batch|{role}|{bi}")["verdicts"]
                         for bi in range(
                             math.ceil(len(per_agent[role]) / batch_size)))]
        # NB: deterministic per-role seed (built-in hash() is salted per
        # process and would break resume reproducibility)
        sample = pp.seeded_sample(judged, calib_frac,
                                  seed * 1000 + roles.index(role))
        for bi in range(0, len(sample), batch_size):
            calib_batches.append((role, bi // batch_size,
                                  sample[bi:bi + batch_size]))
    pending = [(r, bi, b) for (r, bi, b) in calib_batches
               if not store.has(f"calib|{r}|{bi}")]
    _run_batches(pending, calib_judge, "calib")
    store.flush()

    # --- assemble summary ---
    per_agent_summary = {}
    for role in roles:
        utts = per_agent[role]
        series = []
        for u in utts:
            bi = next((k for k in range(
                math.ceil(len(utts) / batch_size))
                if store.has(f"batch|{role}|{k}") and str(u["n"]) in
                store.get(f"batch|{role}|{k}")["verdicts"]), None)
            row = (store.get(f"batch|{role}|{bi}")["verdicts"][str(u["n"])]
                   if bi is not None else None)
            if row is None:
                score = n_e = n_c = appl = None
            else:
                score, n_e, n_c, appl = _score(row)
            series.append({"n": u["n"], "t_index": u["t_index"],
                           "score": score, "n_entail": n_e,
                           "n_contradict": n_c, "n_applicable": appl,
                           "features": behavior_features(u["text"], lang)})
        st = stats.series_stats([s["n"] for s in series],
                                [s["score"] for s in series])
        feats = [s["features"] for s in series]
        per_agent_summary[role] = {
            "n_utterances": len(utts),
            "series": series,
            "stats": st,
            "feature_means": {k: stats.mean([f[k] for f in feats])
                              for k in ("n_chars", "ttr", "punct_density",
                                        "quote_ratio")} if feats else {}}

    # --- judge agreement on the calibration overlap ---
    prim, calib = [], []
    for (role, bi, b) in calib_batches:
        c = store.get(f"calib|{role}|{bi}")
        if not c:
            continue
        for u in b:
            n = str(u["n"])
            pbi = next((k for k in range(
                math.ceil(len(per_agent[role]) / batch_size))
                if store.has(f"batch|{role}|{k}") and n in
                store.get(f"batch|{role}|{k}")["verdicts"]), None)
            if pbi is None or n not in c["verdicts"]:
                continue
            prow = store.get(f"batch|{role}|{pbi}")["verdicts"][n]
            crow = c["verdicts"][n]
            for aid in prow:
                if aid in crow:
                    prim.append(prow[aid])
                    calib.append(crow[aid])
    calibration = {"n_pairs": len(prim),
                   "percent_agreement": stats.percent_agreement(prim, calib),
                   "cohens_kappa": stats.cohens_kappa(prim, calib),
                   "judge": l1_judge, "calib_judge": calib_judge}

    summary = {"per_agent": per_agent_summary, "calibration": calibration,
               "n_batches": len(batches), "failures": failures,
               "judge_model": l1_judge, "lang": lang}
    store.set_meta("summary_generated_at", pp.now_str())
    store.flush()
    return summary
