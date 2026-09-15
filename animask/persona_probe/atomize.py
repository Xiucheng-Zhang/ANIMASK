"""Persona-card atomization (once per pack per role, cached).

Decomposes role_info.json (profile + relations + motivation) into atomic,
independently checkable entries. Atoms of type conditional/relation/knowledge
carry a fixed scenario probe_question, reused verbatim at every checkpoint so
scores are comparable across time.

Cache: data/cache/persona_probe/<pack>/<role>.atoms.json
"""
import json

from animask import persona_probe as pp

ATOM_TYPES = ("trait", "conditional", "relation", "knowledge",
              "capability_limit")
_TYPE_PREFIX = {"trait": "T", "conditional": "C", "relation": "R",
                "knowledge": "K", "capability_limit": "L"}

_PROMPT_EN = """You are analyzing a fictional character card for a \
role-playing simulation study.
Decompose the character card below into ATOMIC persona entries.

Rules:
- Each atom is ONE independently checkable statement about the character, \
written in third person.
- "type" is one of: trait (stable disposition, speech style, or value), \
conditional (if-situation-then-behavior tendency), relation (stance toward a \
named other character), knowledge (a specific piece of private knowledge the \
character holds), capability_limit (something the character cannot do or is \
forbidden to do).
- For types conditional, relation and knowledge ONLY, also write \
"probe_question": a short self-contained scenario question addressed to the \
character in second person, whose in-character answer would reveal whether \
the atom still holds. It must make sense at ANY point of the story (it will \
be reused at several checkpoints). For other types set it to null.
- Produce 8 to 20 atoms covering: speech style, core values, \
condition-dependent tendencies, each listed relation, private knowledge, \
and stated limits.
- Output STRICT JSON: a list of objects \
{{"type": ..., "statement": ..., "probe_question": ... or null}}. \
No other text.

CHARACTER CARD (name: {name}):
{card}"""

_PROMPT_ZH = """你在为一项角色扮演模拟研究分析一张虚构角色卡。
请把下面的角色卡拆解为「原子」persona 条目。

规则:
- 每个原子是一条可独立检验的关于该角色的陈述,用第三人称陈述句。
- "type" 取值:trait(稳定性格/说话风格/价值观)、conditional(条件式行为倾向:\
在某类情境下会如何)、relation(对某个具名角色的态度)、knowledge(该角色掌握的\
一条具体私有知识)、capability_limit(该角色做不到或被禁止做的事)。
- 仅对 conditional、relation、knowledge 三类,额外写 "probe_question":\
一个简短、自足的情景问题,用第二人称向角色提问,其入戏回答能暴露该原子是否仍然\
成立。问题必须在故事任何时点都说得通(它会在多个检查点原样复用)。其余类型填 null。
- 产出 8 到 20 个原子,覆盖:说话风格、核心价值观、条件式倾向、每一条列出的关系、\
私有知识、能力限制。
- 输出严格 JSON:对象列表 \
{{"type": ..., "statement": ..., "probe_question": ... 或 null}}。不要输出其他文字。

角色卡(姓名:{name}):
{card}"""


def build_card(info: dict) -> str:
    """Flatten role_info.json into the text handed to the atomizer."""
    parts = [info.get("profile", "")]
    if info.get("motivation"):
        parts.append("[MOTIVATION]\n" + str(info["motivation"]))
    if info.get("relation"):
        parts.append("[RELATIONS]\n"
                     + json.dumps(info["relation"], ensure_ascii=False,
                                  indent=1))
    return "\n\n".join(p for p in parts if p)


def _normalize(raw_atoms: list, lang: str) -> list:
    """Validate types, assign stable ids (T01/C01/R01/K01/L01...)."""
    counters = {p: 0 for p in _TYPE_PREFIX.values()}
    atoms = []
    for a in raw_atoms:
        if not isinstance(a, dict):
            continue
        typ = str(a.get("type", "")).strip().lower()
        if typ not in ATOM_TYPES:
            typ = "trait"
        stmt = str(a.get("statement", "")).strip()
        if not stmt:
            continue
        prefix = _TYPE_PREFIX[typ]
        counters[prefix] += 1
        q = a.get("probe_question")
        q = str(q).strip() if q else None
        if typ not in ("conditional", "relation", "knowledge"):
            q = None
        atoms.append({"id": f"{prefix}{counters[prefix]:02d}", "type": typ,
                      "statement": stmt, "probe_question": q, "lang": lang})
    return atoms


def role_info_path(pack: str, role: str):
    return pp.roles_dir(pack) / role / "role_info.json"


def atom_cache_path(pack: str, role: str):
    return pp.cache_dir(pack) / f"{role}.atoms.json"


def atomize_role(pack: str, role: str, lang: str,
                 model: str = pp.SONNET, force: bool = False) -> list:
    """Return the atom list for one role, atomizing (1 LLM call) on cache
    miss."""
    cache = atom_cache_path(pack, role)
    if cache.exists() and not force:
        return pp.load_json(cache)["atoms"]
    info = pp.load_json(role_info_path(pack, role))
    name = info.get("role_name") or role
    tmpl = _PROMPT_ZH if lang == "zh" else _PROMPT_EN
    prompt = tmpl.format(name=name, card=build_card(info))
    reply = pp.llm_query(prompt, model=model)
    atoms = _normalize(pp.parse_json_reply(reply), lang)
    if not atoms:
        raise ValueError(f"atomize produced no atoms for {pack}/{role}")
    pp.save_json_atomic(cache, {"pack": pack, "role": role, "lang": lang,
                                "model": model, "generated_at": pp.now_str(),
                                "atoms": atoms})
    return atoms


def plan_atomize(pack: str, roles: list) -> dict:
    """Dry-run: how many atomize calls a run would make (cache misses)."""
    missing = [r for r in roles if not atom_cache_path(pack, r).exists()]
    return {"n_calls": len(missing), "cache_misses": missing}


def atomize_pack(pack: str, roles: list, lang: str,
                 model: str = pp.SONNET, force: bool = False) -> dict:
    return {r: atomize_role(pack, r, lang, model=model, force=force)
            for r in roles}
