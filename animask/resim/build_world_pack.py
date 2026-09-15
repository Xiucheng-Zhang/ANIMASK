"""Hybrid extraction v2: persona/memory split, emitting a world package.

Design: the persona layer (who the character fundamentally IS) and
the memory layer (what they have LIVED through) are extracted separately.

  persona  — speech style, values, constant traits, self-concept, sensitive
             topics, behavioral axioms. Scope selectable:
               --persona-scope full      : whole story ("oracle" condition —
                                           the author-complete personality)
               --persona-scope prefreeze : pre-freeze text only (matched
                                           baseline for the oracle contrast)
  memory   — turning points, first-person experiences, relations, agentic
             state (goals/location/private knowledge), world, locations.
             ALWAYS scoped to text[:freeze_word]. The agent may BE the person
             the whole book reveals, but only KNOWS pre-freeze events.

Anti-leak machinery (applies to the persona layer in both scopes):
  - prompts demand abstract conditional dispositions; no events, objects,
    places, or other characters' names may appear in persona text
  - stability labels: constant | latent (revealed late, but always true) |
    developmental (genuine change, with [start,end] position span)
  - freeze projection: developmental stages that begin only after the freeze
    are dropped from the card; latent/constant items always survive
  - a leakage-audit pass (sees the items ONLY, not the story) rewrites or
    drops anything that still narrates plot or predicts concrete actions
  Position spans stay in the structured x_ layers for analysis; the card
  text injected into the agent carries no story-position metadata.

Stages (every memory-layer prompt sees ONLY text[:freeze_word]):
  A. scene extraction (persona scope, position-stamped)
  B. persona synthesis + axiom induction (per character, persona scope)
  C. memory layer: turning points + identity memory (pre-freeze scenes),
     agentic fields (pre-freeze text)
  D. relations adjacency, locations, world description, neutral intervention
  E. leakage audit + freeze projection -> fold into the engine's `profile`

Output -> engine/data/{roles,worlds,locations}/<pack_id>/ + preset,
where pack_id = <sid>__<variant> (plain <sid> when --variant is empty).

Usage (from the repository root):
  python3 -m animask.resim.build_world_pack story01 --persona-scope full
      -> pack story01__oracle
  python3 -m animask.resim.build_world_pack story01 --persona-scope prefreeze --variant prefreeze
      -> pack story01__prefreeze (baseline matched to the oracle pipeline)
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

from animask.llm import GPT, query, query_many  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = ROOT / "engine"
CACHE_DIR = ROOT / "data" / "cache" / "world_pack"
# provenance: bump when any extraction prompt changes meaningfully, so packs
# record which prompt generation produced them
PROMPT_VERSION = "2026-08-22"


# set by --language zh: appended to every builder prompt so extraction output
# matches the simulation language; empty for en (default) — no behavior change
LANG_DIRECTIVE = ""
ZH_DIRECTIVE = ("\n\n重要:所有输出字段的自然语言内容一律用中文书写"
                "(JSON 键名保持英文,人名/地名等专有名词保留原文写法)。")


class QueryCache:
    """Disk cache for raw LLM outputs, keyed by (prompt, stdin, model), so a
    failed pack build can be re-run without repeating completed calls."""

    def __init__(self, pack_id: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{pack_id}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(job: dict) -> str:
        raw = "\x00".join([job.get("model", GPT), job["prompt"],
                           job.get("stdin") or ""])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def query_many(self, jobs: list[dict]) -> list[str]:
        if LANG_DIRECTIVE:
            # cache key includes the directive, so zh/en variants never collide
            jobs = [{**j, "prompt": j["prompt"] + LANG_DIRECTIVE} for j in jobs]
        keys = [self.key(j) for j in jobs]
        todo = [i for i, k in enumerate(keys)
                if not isinstance(self.data.get(k), str)
                or self.data[k].startswith("<<ERROR")]
        if todo:
            fresh = query_many([jobs[i] for i in todo])
            for i, out in zip(todo, fresh):
                self.data[keys[i]] = out
            self.path.write_text(json.dumps(self.data))
        print(f"  ({len(jobs) - len(todo)}/{len(jobs)} from cache)")
        return [self.data[k] for k in keys]

    def query(self, job: dict) -> str:
        return self.query_many([job])[0]

    def invalidate(self, job: dict) -> None:
        """Drop a cached output that parsed but failed shape validation.
        Must apply the same LANG_DIRECTIVE augmentation as query_many — the
        stored key was computed on the augmented prompt, so hashing the bare
        prompt here deleted nothing and zh retries re-read the bad reply."""
        if LANG_DIRECTIVE:
            job = {**job, "prompt": job["prompt"] + LANG_DIRECTIVE}
        self.data.pop(self.key(job), None)
        self.path.write_text(json.dumps(self.data))

SCENES_PROMPT = (
    "The piped text is {scope_desc}. Extract {n_scenes} SCENES covering the "
    "whole text in order. A scene = a continuous moment where characters act "
    "or interact. position_frac = the scene's position as a fraction of the "
    "piped text.\nReply ONLY a JSON array:\n"
    '[{{"position_frac": 0.05, "location": "<where>", '
    '"characters": ["<name>", ...], "situation": "<1-2 sentences>", '
    '"actions": [{{"character": "<name>", "behavior": "<what they do/say and '
    'how>"}}], "emotions": {{"<name>": "<emotional state>"}}, '
    '"emotional_weight": "high|medium|low", '
    '"quote": "<one short verbatim quote from the scene>"}}]'
)

PERSONA_PROMPT = (
    "You are a professional literary analyst. Based ONLY on these scenes "
    "featuring {name}, build a TIMELESS portrait of who {name} fundamentally "
    "IS — a portrait that would hold in any story about this person.\n\n"
    "STRICT SEPARATION RULES — personality, not plot:\n"
    "- Nowhere in the output may you mention specific events, actions taken, "
    "objects, artifacts, places, institutions, technologies, or the names of "
    "other characters.\n"
    "- Phrase every disposition abstractly and conditionally: 'When <type of "
    "situation>, {name} <type of response>.'\n"
    "- Late scenes often REVEAL what was always true. If late behavior "
    "recontextualizes earlier behavior, state the underlying disposition and "
    "label it stability='latent'.\n"
    "- If the character genuinely CHANGES across the scenes, do not average "
    "the change away: emit one entry per stage, stability='developmental', "
    "with span=[start,end] position fractions taken from the scene stamps.\n"
    "- Traits evidenced uniformly throughout get stability='constant' (span "
    "may be null).\n\nSCENES:\n{scenes}\n\n"
    "Reply ONLY a JSON object:\n"
    "{{\n"
    '  "speech_style": "<how {name} actually expresses themselves IN THE '
    "SCENES: vocabulary, sentence shape, register, tone. Evidence rule: "
    "describe only expression channels the scenes show. If {name} never "
    "speaks, say so and characterize the channels they do use (action, "
    'movement, silence) — never invent a spoken register>",\n'
    '  "verbal_tics": ["<habitual phrase or pattern actually evidenced; '
    'empty list if the character does not speak>", ...],\n'
    '  "core_values": [{{"value": "<value>", "stability": "constant|latent|'
    'developmental", "span": [0.0, 1.0]}}],\n'
    '  "constants": [{{"trait": "<disposition>", "stability": "...", '
    '"span": [0.0, 1.0]}}],\n'
    '  "self_concept": [{{"view": "<how they see themselves>", '
    '"stability": "...", "span": [0.0, 1.0]}}],\n'
    '  "sensitive_topics": [{{"topic": "<abstract theme triggering strong '
    'reactions>", "stability": "...", "span": [0.0, 1.0]}}]\n'
    "}}"
)

AXIOM_PROMPT = (
    "You are a literary analysis expert. From these scenes featuring {name}, "
    "extract personality axioms — generalized behavioral rules that would "
    "cover situations NOT in the text.\n"
    "Example: \"When faced with injustice, {name} chooses direct "
    "confrontation over avoidance, even at personal cost.\"\n\n"
    "STRICT SEPARATION RULES — the axiom text must contain personality, not "
    "plot: no events, objects, places, institutions, or other characters' "
    "names; abstract conditional phrasing only. Late scenes often REVEAL "
    "what was always true (stability='latent'); genuine change over time is "
    "stability='developmental' with span=[start,end] from the scene stamps; "
    "uniform behavior is stability='constant'.\n\nSCENES:\n{scenes}\n\n"
    "Extract 1-3 axioms for EACH domain: relationships / conflict / values / "
    "self_concept / worldview. Include, where the scenes support them, the "
    "character's darker or costlier dispositions — what they are capable of "
    "under extreme pressure, not only their everyday manner.\n"
    "Reply ONLY a JSON object: {{\"axioms\": [{{\"domain\": \"...\", "
    "\"axiom\": \"...\", \"stability\": \"constant|latent|developmental\", "
    "\"span\": [0.0, 1.0], \"evidence\": \"<which scene pattern supports "
    "it — for analysis only, may reference the plot>\"}}]}}"
)

MEMORY_PROMPT = (
    "Based ONLY on these scenes featuring {name} — they cover the story up "
    "to the present moment — extract {name}'s lived memory as of NOW.\n\n"
    "SCENES:\n{scenes}\n\n"
    "Reply ONLY a JSON object:\n"
    "{{\n"
    '  "turning_points": [{{"trigger": "...", "before_state": "...", '
    '"after_state": "...", "significance": "high|medium"}}],\n'
    '  "identity_memory": {{"core_experiences": ["<first-person memory>", '
    "...]}}\n"
    "}}"
)

AUDIT_PROMPT = (
    "Below are numbered items from a dispositional portrait of a fictional "
    "character, to be injected into a role-playing agent. Each item must "
    "describe PERSONALITY ONLY. An item violates the contract if it narrates "
    "specific plot events, names other characters / places / objects / "
    "institutions / technologies, or predicts a specific concrete action.\n"
    "For each item: keep (already abstract), rewrite (plot residue — rewrite "
    "into an abstract conditional disposition preserving the full "
    "psychological content, including dark or destructive potentials), or "
    "drop (irreducibly plot).\n"
    "Reply ONLY a JSON object: {{\"items\": [{{\"id\": 0, \"action\": "
    "\"keep|rewrite|drop\", \"text\": \"<final text; empty if drop>\"}}]}}\n\n"
    "ITEMS:\n{items}"
)

AGENTIC_PROMPT = (
    "The piped text is the beginning of a story, cut at a freeze point. For "
    "the character {name}, extract their agentic state AT THE FREEZE MOMENT.\n"
    "The embodiment fields define the character's hard physical envelope. "
    "An actor holding only this card must be able to answer, for any "
    "contemplated action, whether it lies within this being's body, senses, "
    "reach, and skills. Derive the envelope from what the character IS and "
    "what the text shows them doing — absence of an explicit statement is "
    "not license to exceed the envelope.\n"
    "Reply ONLY a JSON object:\n"
    "{{\n"
    '  "motivation": "<their long-term drive, in second person (\'You want...\')>",\n'
    '  "goal_now": "<their immediate objective at the freeze moment>",\n'
    '  "activity": <0.1-1.0 — how proactively they act (1 = drives events)>,\n'
    '  "location_now": "<where they are at the freeze moment>",\n'
    '  "state_now": "<objective physical/mental state at the freeze moment, '
    'one sentence>",\n'
    '  "conscious": true|false — false if asleep, unconscious, or otherwise '
    "unable to act at the freeze moment,\n"
    '  "private_knowledge": ["<things ONLY they know: secrets, plans, '
    "feelings. State each fact ITSELF, concretely — never a reference to or "
    "description of knowledge. 'The plan he intends to propose' is INVALID; "
    "write what the plan IS, with its actual names and content>\"],\n"
    '  "embodiment": {{"kind": "<what kind of being this is: human / robot / '
    'fixed device / animal / disembodied system / ...>", "form": "<its body '
    "or substrate: what it physically is, and whether it is mobile or fixed "
    'in place>", "senses": ["<each way it perceives, with that channel\'s '
    'boundary>"], "actuation": ["<each way it can act on the physical '
    'world, with its reach>"]}},\n'
    '  "competences_lacking": ["<general competences this character does '
    "not have although their situation would otherwise assume them — state "
    "the general lack (e.g. 'cannot read or write'), never one instance of "
    'it (\'cannot read the poems\'); empty list if none>"],\n'
    '  "does_not_know": ["<facts others know but they do not>"]\n'
    "}}"
)

# oracle variant: agentic state is still taken AT the freeze moment, but the
# knowledge scope follows the full text — standing facts the text only
# reveals later (people the character already knew, places that already
# existed, their own secrets) belong in private_knowledge; trajectory facts
# (anything true only because of post-freeze events) never do.
AGENTIC_PROMPT_FULL = AGENTIC_PROMPT.replace(
    "The piped text is the beginning of a story, cut at a freeze point.",
    "The piped text is a COMPLETE story with a <<<FREEZE POINT>>> marker "
    "where a simulation takes over.",
) + (
    "\nKnowledge scope: private_knowledge may include standing facts that "
    "the text only reveals after the marker IF the character already "
    "possessed them at the freeze moment (people they already knew and how "
    "they know them, places and institutions that already existed, their own "
    "long-held secrets, plans they have already formed — stated with their "
    "actual names and content). NEVER include anything the character only "
    "learns, decides, or does through events after the marker, and never "
    "include other characters' post-marker actions or outcomes. Apply this "
    "to names too: if {name} only learns another character's name after the "
    "marker, refer to that character by the appellation {name} uses at the "
    "freeze moment, not the later-revealed name.\n"
    "What matters is whether {name} already knew a fact — NOT whether the "
    "reader has seen it yet. Facts {name} already possesses (a person they "
    "know and that person's actual name, an establishment and what it "
    "offers, a plan they have already formed) must be stated CONCRETELY "
    "with their proper nouns and content. Never compress an already-formed "
    "plan into a vague category ('a kind of work in another district'): an "
    "actor holding only that string could not play the scene — name the "
    "place, the person, and the arrangement."
)

# private_knowledge items must be usable by an actor who holds ONLY that
# string: a self-referential placeholder ("the exact route you are about to
# offer") silently deletes the very fact the oracle condition exists to
# provide. Flag such items at build time.
_REFERENTIAL_PK = re.compile(
    r"\b(the exact|the specific|the precise|the details? of)\b"
    r"|\b(that|which) (you|he|she|they) (are about to|intend|plan|will) ",
    re.I)


def lint_private_knowledge(name: str, agentic: dict) -> list:
    flagged = []
    for item in agentic.get("private_knowledge") or []:
        if _REFERENTIAL_PK.search(str(item)) or len(str(item).split()) < 4:
            flagged.append(item)
            print(f"  WARNING [{name}] private_knowledge looks referential, "
                  f"not a stated fact: {str(item)[:120]!r}")
    return flagged

RELATIONS_PROMPT = (
    "The piped text is the beginning of a story. Describe how {name} views "
    "and relates to EACH of these characters: {others}. Base it only on the "
    "text; the view may be asymmetric.\nReply ONLY a JSON object keyed by "
    "those exact names:\n"
    '{{"<Name>": {{"relation": ["<1-3 short labels>"], '
    '"detail": "<{name}\'s attitude toward them, tensions, history>"}}}}'
)

LOCATIONS_PROMPT = (
    "The piped text is a story (possibly cut at a freeze point). List every "
    "distinct physical location that exists in it and could host scenes "
    "(include at least one where a character could act unobserved, if the "
    "text supports it).\n"
    "Reply ONLY a JSON array: [{{\"location_name\": \"<name>\", "
    "\"description\": \"<one line>\", \"detail\": \"<who/what is there, "
    "visibility, access>\"}}]"
)

IDENTITY_PROMPT = (
    "The piped text is a story. Build its IDENTITY REGISTRY: every sentient "
    "entity that acts or is referred to, with ALL names, titles, and "
    "appellations the text uses for it. One entry per underlying entity — "
    "when one being carries several appellations (a formal name, a role "
    "title, a nickname, a secret or later-revealed identity), they belong "
    "to the SAME entry, never to separate entries. Resolve narrator "
    "identities the same way: if the narrator is or becomes a named entity, "
    "that is one entity.\n"
    "An appellation is a name, title, or epithet actually used in the text "
    "to refer to or address the entity. Descriptions are not appellations "
    "(phrasings like 'his sister' or 'the taller one' describe; they do "
    "not name).\n"
    "For each appellation, state its knowledge scope: who in the story "
    "knows that this appellation refers to this entity ('all' if common "
    "knowledge).\n"
    "These cast characters exist as separate simulation agents: {cast}. "
    "Mark the entry that corresponds to each.\n"
    "Reply ONLY a JSON array:\n"
    '[{{"entity": "<canonical name>", '
    '"appellations": [{{"name": "<name/title as used in the text>", '
    '"known_by": "all|<who knows this appellation refers to this '
    'entity>"}}], '
    '"cast_name": "<the exact matching cast name from the list above, or '
    'null if not cast>"}}]'
)

NPC_CARDS_PROMPT = (
    "The piped text is a story. List every named or title-identified "
    "character in it OTHER THAN: {cast}. These become canonical NPC cards — "
    "when the simulation needs one of them to act or speak, the world model "
    "plays them from this card. Include characters who are only referred to "
    "when the text establishes concrete facts about them; skip anonymous "
    "walk-ons about whom nothing is established.\n"
    "Card contents must be STANDING facts: identity, position, ownership, "
    "authority, manner, general behavior patterns, long-standing "
    "relationships. Do NOT include plot events, intentions, or interactions "
    "involving {cast} — the simulation may diverge from the story, and the "
    "card must stay valid in any timeline. (A long-standing relationship "
    "that predates the story's events is fine; 'she intends to…' or 'she "
    "tells X…' is not.)\n"
    "Reply ONLY a JSON array:\n"
    '[{{"name": "<canonical name>", '
    '"aliases": ["<other names/titles they are called by>", ...], '
    '"identity": "<who they are: position, what they own or control, '
    'authority>", '
    '"manner": "<speech style / demeanor, one line>", '
    '"facts": ["<objective fact the text establishes about them>", ...], '
    '"relationships": ["<relation to another character, one line>", ...]}}]'
)

WORLD_PROMPT = (
    "The piped text is the beginning of a story, cut at a freeze point.\n"
    "Reply ONLY a JSON object:\n"
    "{{\n"
    '  "world_name": "<short name for this setting>",\n'
    '  "description": "<the world: place, era, technology/rules, social '
    'structure — everything an actor needs, 150-250 words>",\n'
    '  "intervention": "<the situation at the exact freeze moment, 60-120 '
    "words, strictly NEUTRAL and factual: who is where doing what, what is "
    "unresolved. No foreshadowing, no drama language, no hints about what "
    'should happen next>"\n'
    "}}"
)


def raw_prefix(text: str, n_words: int) -> str:
    """The first n_words whitespace tokens of text as a raw slice (line
    breaks intact) — same span as " ".join(text.split()[:n_words])."""
    end = 0
    for i, m in enumerate(re.finditer(r"\S+", text)):
        if i >= n_words:
            break
        end = m.end()
    return text[:end]


def jparse(raw: str):
    m = re.search(r"[\[{]", raw or "")
    if not m:
        return None
    s = raw[m.start():]
    for candidate in (s, s.replace("“", '"').replace("”", '"')):
        try:
            return json.JSONDecoder().raw_decode(candidate)[0]
        except json.JSONDecodeError:
            continue
    return None


def parse_or_retry(cache: "QueryCache", job: dict, out: str, tries: int = 2):
    v = jparse(out)
    while v is None and tries > 0:
        # retry through the cache, not a bare query(): the bare path skipped
        # the zh LANG_DIRECTIVE (English content could enter zh packs) and
        # never wrote the good reply back, so every resumed build re-paid it
        cache.invalidate(job)
        v = jparse(cache.query(job))
        tries -= 1
    if v is None:
        raise ValueError(f"unparseable after retries: {job['prompt'][:80]}...")
    return v


PACK_LANG = "en"  # set from --language in main(); suffixes role codes


def code_of(name: str) -> str:
    ascii_part = re.sub(r"[^A-Za-z0-9]", "", name)
    if not ascii_part and re.search(r"[一-鿿]", name):
        # CJK name -> pinyin code, following the engine's convention
        # (e.g. JiaBaoyu-zh); role codes must be ASCII (dir names, db names)
        try:
            from pypinyin import lazy_pinyin
            ascii_part = "".join(w.capitalize() for w in lazy_pinyin(
                re.sub(r"[^一-鿿A-Za-z0-9]", "", name)))
        except ImportError:
            ascii_part = "R" + hashlib.sha256(
                name.encode()).hexdigest()[:8]
    return ascii_part + "-" + PACK_LANG


def ascii_slug(name: str) -> str:
    """ASCII identifier for an arbitrary name. Latin letters/digits are kept;
    CJK runs are transliterated to capitalised pinyin (hash fallback without
    pypinyin). Unlike the old `re.sub(r"[^A-Za-z0-9]", "", name)`, a pure-CJK
    name never collapses to "" and a mixed name ("秦氏赌场VIP包厢") keeps its
    CJK part instead of degrading to the ASCII fragment ("VIP")."""
    if not re.search(r"[一-鿿]", name):
        return re.sub(r"[^A-Za-z0-9]", "", name)
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return "H" + hashlib.sha256(name.encode()).hexdigest()[:8]
    parts = []
    for seg in re.findall(r"[一-鿿]+|[A-Za-z0-9]+", name):
        if re.match(r"[一-鿿]", seg):
            parts.append("".join(w.capitalize() for w in lazy_pinyin(seg)))
        else:
            parts.append(seg)
    return "".join(parts)


def build_location_table(locations: list) -> dict:
    """{location_code: {...}} for the engine's data/locations/<pack>.json.
    Codes come from ascii_slug; distinct names that slug to the same code get
    a numeric suffix (the old code silently overwrote — every CJK-only name
    slugged to "" and all zh packs ended up with a single location); exact
    duplicate names and empty names are dropped."""
    locs, seen_names = {}, set()
    for L in locations or []:
        if not isinstance(L, dict):
            continue
        name = str(L.get("location_name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        base = ascii_slug(name) or "Loc"
        lcode, k = base, 2
        while lcode in locs:
            lcode, k = f"{base}{k}", k + 1
        locs[lcode] = {"location_code": lcode, "location_name": name,
                       "description": L.get("description", ""),
                       "detail": L.get("detail", "")}
    return locs


# ---------------------------------------------------------------- persona ops

def span_start(item: dict):
    s = item.get("span")
    return s[0] if isinstance(s, list) and s and isinstance(s[0], (int, float)) else None


def project_to_freeze(items: list, freeze_frac: float, eps: float = 0.05) -> tuple:
    """Constant/latent dispositions always survive; developmental stages
    survive only if they have begun by the freeze point."""
    kept, dropped = [], []
    for it in items:
        if it.get("stability") == "developmental":
            s = span_start(it)
            if s is not None and s > freeze_frac + eps:
                dropped.append(it)
                continue
        kept.append(it)
    return kept, dropped


def audit_persona(name: str, persona: dict, axioms: list,
                  cache: "QueryCache") -> dict:
    """Leakage audit: the auditor sees the items only, never the story.
    Mutates persona/axioms in place; returns the audit log."""
    fields = [("core_values", "value"), ("constants", "trait"),
              ("self_concept", "view"), ("sensitive_topics", "topic")]
    entries = []  # (container, index, key)
    for fname, key in fields:
        for i, it in enumerate(persona.get(fname) or []):
            entries.append((persona[fname], i, key))
    for i, ax in enumerate(axioms):
        entries.append((axioms, i, "axiom"))
    numbered = "\n".join(
        f"{n}. {cont[i][key]}" for n, (cont, i, key) in enumerate(entries))
    out = cache.query({"prompt": AUDIT_PROMPT.format(items=numbered),
                       "model": GPT})
    verdicts = (jparse(out) or {}).get("items", [])
    log, to_drop = [], set()
    for v in verdicts:
        n = v.get("id")
        if not isinstance(n, int) or not 0 <= n < len(entries):
            continue
        cont, i, key = entries[n]
        if v.get("action") == "rewrite" and v.get("text"):
            log.append({"action": "rewrite", "from": cont[i][key], "to": v["text"]})
            cont[i][key] = v["text"]
        elif v.get("action") == "drop":
            log.append({"action": "drop", "from": cont[i][key]})
            to_drop.add((id(cont), i))
    for fname, key in fields:
        persona[fname] = [it for i, it in enumerate(persona.get(fname) or [])
                          if (id(persona[fname]), i) not in to_drop]
    axioms[:] = [ax for i, ax in enumerate(axioms)
                 if (id(axioms), i) not in to_drop]
    return {"n_items": len(entries), "changes": log}


def fmt_profile(name: str, persona: dict, axioms: list, memory: dict,
                agentic: dict) -> str:
    """Fold the layers into the engine's `profile` text field, which the
    engine injects verbatim into every role prompt. No story-position
    metadata may appear here."""
    parts = [f"[SPEECH STYLE]\n{persona.get('speech_style', '')}"]
    if persona.get("verbal_tics"):
        parts.append("[VERBAL TICS — descriptions of tendencies; realize them "
                     "naturally in your own words, NEVER say these "
                     "descriptions' wording aloud]\n"
                     + "; ".join(persona["verbal_tics"]))
    parts.append("[CORE VALUES]\n" + "\n".join(
        f"- {v['value']}" for v in persona.get("core_values", [])))
    parts.append("[CONSTANT TRAITS]\n" + "\n".join(
        f"- {c['trait']}" for c in persona.get("constants", [])))
    if persona.get("self_concept"):
        parts.append("[SELF-CONCEPT]\n" + "\n".join(
            f"- {s['view']}" for s in persona["self_concept"]))
    if memory.get("turning_points"):
        tp = "\n".join(f"- {t['trigger']}: {t['before_state']} -> {t['after_state']}"
                       for t in memory["turning_points"])
        parts.append(f"[TURNING POINTS SO FAR]\n{tp}")
    im = memory.get("identity_memory", {})
    if im.get("core_experiences"):
        parts.append("[MEMORIES (first person)]\n"
                     + "\n".join(f"- {e}" for e in im["core_experiences"]))
    if persona.get("sensitive_topics"):
        parts.append("[EMOTIONALLY SENSITIVE TOPICS]\n"
                     + "; ".join(t["topic"] for t in persona["sensitive_topics"]))
    parts.append("[BEHAVIORAL AXIOMS]\n"
                 + "\n".join(f"- ({a['domain']}) {a['axiom']}" for a in axioms))
    emb = agentic.get("embodiment") or {}
    if emb:
        lines = []
        if emb.get("kind") or emb.get("form"):
            lines.append(f"- You are: {emb.get('kind', '')}"
                         + (f" — {emb['form']}" if emb.get("form") else ""))
        for s in emb.get("senses") or []:
            lines.append(f"- Perception: {s}")
        for a in emb.get("actuation") or []:
            lines.append(f"- Action: {a}")
        parts.append("[EMBODIMENT — your physical envelope. Every action "
                     "must lie within this body, these senses, and this "
                     "reach; nothing outside it is possible for you]\n"
                     + "\n".join(lines))
    lacks = agentic.get("competences_lacking") or []
    if lacks:
        parts.append("[COMPETENCES YOU LACK — never exhibit these skills, "
                     "in any form or degree]\n"
                     + "\n".join(f"- {c}" for c in lacks))
    if agentic.get("capability_limits"):
        parts.append("[CAPABILITIES / LIMITS — hard constraints; never act "
                     "beyond them]\n"
                     + "\n".join(f"- {c}" for c in agentic["capability_limits"]))
    if agentic.get("private_knowledge"):
        parts.append("[PRIVATE KNOWLEDGE — only you know this]\n"
                     + "\n".join(f"- {p}" for p in agentic["private_knowledge"]))
    if agentic.get("does_not_know"):
        parts.append("[YOU DO NOT KNOW]\n"
                     + "\n".join(f"- {d}" for d in agentic["does_not_know"]))
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--persona-scope", default="prefreeze",
                    choices=["prefreeze", "full"])
    ap.add_argument("--variant", default=None,
                    help="pack suffix; defaults to 'oracle' for scope=full, "
                         "none for scope=prefreeze")
    ap.add_argument("--language", default="en", choices=["en", "zh"],
                    help="simulation language written into the preset; zh also "
                         "directs all extraction output to Chinese")
    args = ap.parse_args()
    global LANG_DIRECTIVE, PACK_LANG
    PACK_LANG = args.language
    if args.language == "zh":
        LANG_DIRECTIVE = ZH_DIRECTIVE
    sid = args.sid
    variant = args.variant if args.variant is not None else (
        "oracle" if args.persona_scope == "full" else "")
    pack_id = f"{sid}__{variant}" if variant else sid

    meta = json.loads((ROOT / "data" / "meta" / f"resim_{sid}.json").read_text())
    text = (ROOT / "data" / "clean" / f"{sid}.txt").read_text()
    pre_text = " ".join(text.split()[: meta["freeze_word"]])
    # role-agent RAG corpus: the same freeze-scoped span, but as a raw slice
    # of the original text so its line breaks survive — the engine chunks
    # role data BY LINE (utils.split_text_by_max_words), and a newline-free
    # join collapses the whole span into one chunk that retrieval then injects
    # verbatim into every role prompt
    pre_text_raw = raw_prefix(text, meta["freeze_word"])
    freeze_frac = meta["freeze_word"] / meta["total_words"]
    cast = [c["name"] for c in meta["state"]["characters"] if c.get("simulate")]
    print(f"{pack_id}: persona_scope={args.persona_scope}, "
          f"freeze@{meta['freeze_word']}w ({freeze_frac:.0%}), cast={cast}")
    # role-code collision check BEFORE any paid call: two distinct cast names
    # mapping to one code (homophones like 萧姗/萧珊, or ASCII-fragment ties)
    # would silently overwrite each other's role dir and double-schedule one
    # agent — refuse and let a human decide (often it is one person with a
    # spelling variant in the cast list)
    by_code: dict = {}
    for c in cast:
        by_code.setdefault(code_of(c), []).append(c)
    collisions = {k: v for k, v in by_code.items() if len(v) > 1}
    if collisions:
        raise SystemExit(f"cast role-code collision: {collisions} — fix the "
                         f"cast in data/meta/resim_{sid}.json before "
                         "building")

    # -- stage A: scenes over the persona scope, position-stamped ------------
    if args.persona_scope == "full":
        scene_src, scope_desc, n_scenes = text, "a complete short story", "18-30"
    else:
        scene_src, scope_desc, n_scenes = (
            pre_text, "the beginning of a story, cut at a freeze point", "12-25")
    cache = QueryCache(pack_id)
    print("stage A: scene extraction ...")
    scene_job = {"prompt": SCENES_PROMPT.format(scope_desc=scope_desc,
                                                n_scenes=n_scenes),
                 "model": GPT, "stdin": scene_src}
    scenes = None
    for attempt in range(3):
        parsed = jparse(cache.query(scene_job))
        if isinstance(parsed, list):
            good = [s for s in parsed if isinstance(s, dict)
                    and "position_frac" in s and "situation" in s]
            if len(good) >= 5:
                scenes = good
                break
        cache.invalidate(scene_job)
        print(f"  (scene shape invalid, attempt {attempt + 1}/3)")
    if scenes is None:
        # whole-text extraction can exceed the provider's gateway timeout on
        # long/dense stories (HTTP 524, deterministic) — halve input AND
        # output per call, then merge with rescaled positions
        print("  falling back to split-scene extraction (2 halves) ...")
        words_a = scene_src.split()
        mid = len(words_a) // 2
        halves = [(" ".join(words_a[:mid]), 0.0),
                  (" ".join(words_a[mid:]), 0.5)]
        merged = []
        for part_text, frac_lo in halves:
            part_job = {"prompt": SCENES_PROMPT.format(
                scope_desc=scope_desc + " (one half of a longer text)",
                n_scenes="9-15"), "model": GPT, "stdin": part_text}
            for attempt in range(3):
                parsed = jparse(cache.query(part_job))
                if isinstance(parsed, list):
                    good = [s for s in parsed if isinstance(s, dict)
                            and "position_frac" in s and "situation" in s]
                    if len(good) >= 3:
                        for s in good:
                            try:
                                s["position_frac"] = round(
                                    frac_lo + float(s["position_frac"]) * 0.5,
                                    4)
                            except (TypeError, ValueError):
                                s["position_frac"] = frac_lo + 0.25
                        merged.extend(good)
                        break
                cache.invalidate(part_job)
                print(f"  (half@{frac_lo} shape invalid, "
                      f"attempt {attempt + 1}/3)")
        if len(merged) >= 5:
            scenes = sorted(merged, key=lambda s: s["position_frac"])
            print(f"  split extraction ok: {len(scenes)} scenes")
    if scenes is None:
        raise ValueError("scene extraction failed shape validation after retries")
    # anchor scene positions on their verbatim quotes where findable — the
    # model's own position_frac estimates are too noisy for boundary filtering
    src_norm = re.sub(r"[“”\"']", "", scene_src)
    src_words = len(src_norm.split())
    for s in scenes:
        q = re.sub(r"[“”\"']", "", (s.get("quote") or "")).strip()
        toks = q.split()
        # specificity gate: >=4 whitespace tokens (en) or >=8 CJK chars — a
        # CJK quote is ONE whitespace token, so the token-only gate never
        # anchored any zh scene: every zh position stayed a model guess and
        # memory_cutoff filtered on noise (post-freeze scenes could slip
        # into the oracle memory layer)
        if len(toks) >= 4:
            needle = " ".join(toks[:8])
        elif len(re.findall(r"[一-鿿]", q)) >= 8:
            needle = q[:24]
        else:
            continue
        pos = src_norm.find(needle)
        if pos >= 0:
            s["position_frac_est"] = s.get("position_frac")
            s["position_frac"] = round(
                len(src_norm[:pos].split()) / max(1, src_words), 3)

    # scene positions are fractions of scene_src: for full scope compare
    # against freeze_frac; for prefreeze scope everything is pre-freeze by
    # construction (stamps span 0..1 of the PRE-FREEZE text — never compare
    # those against freeze_frac, which lives on the full-text axis)
    memory_cutoff = freeze_frac if args.persona_scope == "full" else 1.01
    scenes_pre = [s for s in scenes
                  if float(s.get("position_frac", 0)) <= memory_cutoff]
    anchored = sum(1 for s in scenes if "position_frac_est" in s)
    print(f"  {len(scenes)} scenes ({len(scenes_pre)} pre-freeze, "
          f"{anchored} quote-anchored)")

    def render(subset: list, name: str) -> str:
        mine = [s for s in subset if any(name.lower() in c.lower()
                                         for c in s.get("characters", []))]
        return "\n---\n".join(
            f"[{s['position_frac']:.2f} | {s['emotional_weight']}] {s['situation']}\n"
            + "\n".join(f"  {a['character']}: {a['behavior']}" for a in s.get("actions", []))
            + (f"\n  quote: {s.get('quote', '')}" if s.get("quote") else "")
            for s in mine)

    # ambient characters (e.g. a system-entity like Hypnos) may never be
    # listed in scene character lists; fall back to the raw text for them
    FALLBACK = ("(no indexed scenes feature this character by name — infer "
                "their portrait from the piped story text instead)")

    def sc_args(subset: list, name: str, src: str) -> tuple:
        block = render(subset, name)
        return (block, None) if block.strip() else (FALLBACK, src)

    # -- stages B+C+D: per-character persona / axioms / memory / agentic -----
    print("stage B+C+D: persona / axioms / memory / agentic (parallel) ...")
    jobs = []
    for n in cast:
        pb, pstdin = sc_args(scenes, n, scene_src)
        mb, mstdin = sc_args(scenes_pre, n, pre_text)
        jobs.append({"prompt": PERSONA_PROMPT.format(name=n, scenes=pb),
                     "model": GPT, "stdin": pstdin})
        jobs.append({"prompt": AXIOM_PROMPT.format(name=n, scenes=pb),
                     "model": GPT, "stdin": pstdin})
        jobs.append({"prompt": MEMORY_PROMPT.format(name=n, scenes=mb),
                     "model": GPT, "stdin": mstdin})
        if args.persona_scope == "full":
            marker_text = pre_text + "\n<<<FREEZE POINT>>>\n" + \
                " ".join(text.split()[meta["freeze_word"]:])
            jobs.append({"prompt": AGENTIC_PROMPT_FULL.format(name=n),
                         "model": GPT, "stdin": marker_text})
        else:
            jobs.append({"prompt": AGENTIC_PROMPT.format(name=n), "model": GPT,
                         "stdin": pre_text})
    for n in cast:
        jobs.append({"prompt": RELATIONS_PROMPT.format(
            name=n, others=", ".join(c for c in cast if c != n)),
            "model": GPT, "stdin": pre_text})
    # locations and NPC cards follow the persona scope: the oracle condition
    # grounds the world side in the same text the cast's knowledge comes
    # from, so post-freeze places and characters are not unregistered ghosts
    jobs.append({"prompt": LOCATIONS_PROMPT, "model": GPT, "stdin": scene_src})
    jobs.append({"prompt": WORLD_PROMPT, "model": GPT, "stdin": pre_text})
    jobs.append({"prompt": NPC_CARDS_PROMPT.format(cast=", ".join(cast)),
                 "model": GPT, "stdin": scene_src})
    jobs.append({"prompt": IDENTITY_PROMPT.format(cast=", ".join(cast)),
                 "model": GPT, "stdin": scene_src})
    outs = cache.query_many(jobs)

    per_char, i = {}, 0
    for n in cast:
        ax = parse_or_retry(cache, jobs[i + 1], outs[i + 1])
        if isinstance(ax, dict):
            ax = ax.get("axioms")
        if not isinstance(ax, list):
            # parseable but wrong shape: drop the cached reply so a resumed
            # build re-queries instead of failing on it forever
            cache.invalidate(jobs[i + 1])
            raise ValueError(f"axioms reply for {n} has no 'axioms' list")
        per_char[n] = {"persona": parse_or_retry(cache, jobs[i], outs[i]),
                       "axioms": ax,
                       "memory": parse_or_retry(cache, jobs[i + 2], outs[i + 2]),
                       "agentic": parse_or_retry(cache, jobs[i + 3], outs[i + 3])}
        i += 4
    relations = {}
    for n in cast:
        try:
            relations[n] = parse_or_retry(cache, jobs[i], outs[i])
        except ValueError:
            print(f"  WARNING: relations unparseable for {n}, using empty map")
            relations[n] = {}
        i += 1
    locations = parse_or_retry(cache, jobs[i], outs[i])
    world = parse_or_retry(cache, jobs[i + 1], outs[i + 1])
    try:
        npc_cards_list = parse_or_retry(cache, jobs[i + 2], outs[i + 2])
    except ValueError:
        print("  WARNING: NPC cards unparseable, writing empty set")
        npc_cards_list = []
    try:
        identity_list = parse_or_retry(cache, jobs[i + 3], outs[i + 3])
    except ValueError:
        print("  WARNING: identity registry unparseable, writing empty set")
        identity_list = []
    # every appellation of a cast entity — the exclusion set for NPC cards
    # and the runtime routing table for "npc" targets that are really cast.
    # An appellation that ALSO denotes a non-cast entity (a double's assumed
    # name, a shared title) is ambiguous: it identifies no one uniquely, so
    # it must be excluded from both uses.
    def _akey(s: str) -> tuple:
        return tuple(sorted(set(re.findall(r"[a-z]+|[\u4e00-\u9fff]+",
                                           s.lower()))
                            - {"the", "a", "an"}))
    noncast_keys = set()
    for ent in identity_list if isinstance(identity_list, list) else []:
        if isinstance(ent, dict) and not ent.get("cast_name"):
            for a in ent.get("appellations") or []:
                if isinstance(a, dict) and a.get("name"):
                    noncast_keys.add(_akey(a["name"]))
            if ent.get("entity"):
                noncast_keys.add(_akey(ent["entity"]))
    cast_aliases: dict = {}
    for ent in identity_list if isinstance(identity_list, list) else []:
        if not isinstance(ent, dict) or not ent.get("cast_name"):
            continue
        names = [a.get("name", "") for a in ent.get("appellations") or []
                 if isinstance(a, dict)]
        kept, ambiguous = [], []
        for n in names + [ent.get("entity", "")]:
            if not n:
                continue
            (ambiguous if _akey(n) in noncast_keys else kept).append(n)
        if ambiguous:
            print(f"  identity: ambiguous appellation(s) for "
                  f"{ent['cast_name']} dropped: {ambiguous}")
        cast_aliases.setdefault(ent["cast_name"], []).extend(kept)

    # -- stage E: freeze projection + leakage audit ---------------------------
    # projection cutoff lives on the same axis as the persona-scope scene
    # stamps: full scope -> freeze_frac; prefreeze scope -> no projection
    # (its 0..1 axis is entirely pre-freeze already)
    projection_cutoff = freeze_frac if args.persona_scope == "full" else 1.01
    print("stage E: freeze projection + leakage audit ...")
    for n in cast:
        d = per_char[n]
        p, projected = d["persona"], {}
        for f in ("core_values", "constants", "self_concept", "sensitive_topics"):
            p[f], dropped = project_to_freeze(p.get(f) or [], projection_cutoff)
            if dropped:
                projected[f] = dropped
        d["axioms"], dropped = project_to_freeze(d["axioms"], projection_cutoff)
        if dropped:
            projected["axioms"] = dropped
        d["projection_dropped"] = projected
        d["audit"] = audit_persona(n, p, d["axioms"], cache)
        lint_private_knowledge(n, d["agentic"])
        print(f"  {n}: projection dropped "
              f"{sum(len(v) for v in projected.values())}, "
              f"audit changed {len(d['audit']['changes'])}/{d['audit']['n_items']}")

    # -- write the world package ------------------------------------------
    roles_dir = ENGINE / "data" / "roles" / pack_id
    for n in cast:
        d = per_char[n]
        rdir = roles_dir / code_of(n)
        rdir.mkdir(parents=True, exist_ok=True)
        rel = {}
        for other, r in (relations.get(n) or {}).items():
            if other == n:
                continue
            low = other.lower()
            matches = [c for c in cast if c != n
                       and (low in c.lower() or c.lower() in low)]
            if not matches:
                continue
            # key by the MATCHED cast member's code — code_of(other) on a
            # model-written variant name yields a code no role owns and the
            # relation silently vanishes at runtime (search_relation misses)
            exact = [c for c in matches if c.lower() == low]
            contained = [c for c in matches if c.lower() in low]
            target = (exact[0] if exact
                      else max(contained, key=len) if contained
                      else min(matches, key=len))
            rel[code_of(target)] = {"relation": r.get("relation", []),
                                    "detail": r.get("detail", "")}
        role_info = {
            "role_code": code_of(n), "role_name": n, "nickname": n.split()[0],
            "source": pack_id, "activity": d["agentic"].get("activity", 1.0),
            "profile": fmt_profile(n, d["persona"], d["axioms"],
                                   d["memory"], d["agentic"]),
            "relation": rel,
            "motivation": d["agentic"].get("motivation", ""),
            # structured layers kept for engine patches / analysis
            "x_persona_scope": args.persona_scope,
            "x_persona": d["persona"], "x_axioms": d["axioms"],
            "x_memory": d["memory"], "x_agentic": d["agentic"],
            "x_projection_dropped": d["projection_dropped"],
            "x_audit": d["audit"],
        }
        (rdir / "role_info.json").write_text(
            json.dumps(role_info, indent=2, ensure_ascii=False))
        (rdir / "pre_freeze.txt").write_text(pre_text_raw)  # freeze-scoped RAG corpus
        print(f"  role written: {code_of(n)} (activity={role_info['activity']})")

    wdir = ENGINE / "data" / "worlds" / pack_id
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "general.json").write_text(json.dumps({
        "world_name": world["world_name"], "source": pack_id,
        "description": world["description"], "language": args.language},
        indent=2, ensure_ascii=False))

    npcs = {}
    all_cast_names = set(cast) | {a for v in cast_aliases.values() for a in v}
    def _norm(s: str) -> tuple:
        return tuple(sorted(set(re.findall(r"[a-z]+|[\u4e00-\u9fff]+",
                                           s.lower()))
                            - {"the", "a", "an"}))
    cast_keys = {_norm(n) for n in all_cast_names if n}
    def _is_cast_entity(card: dict) -> bool:
        # whole-appellation equality only: a card that merely CONTAINS a
        # cast name ("Tillie's Mother") is a different person, not the cast
        for label in [card.get("name", "")] + list(card.get("aliases") or []):
            if label and _norm(label) in cast_keys:
                return True
        return False
    for c in npc_cards_list if isinstance(npc_cards_list, list) else []:
        if isinstance(c, dict) and c.get("name"):
            if _is_cast_entity(c):
                print(f"  npc card excluded (cast identity): {c['name']}")
                continue
            npcs[c["name"]] = c
    (wdir / "npc_cards.json").write_text(
        json.dumps(npcs, indent=2, ensure_ascii=False))
    print(f"  npc cards written: {len(npcs)} ({', '.join(npcs) or 'none'})")
    (wdir / "identity_map.json").write_text(json.dumps({
        "cast_aliases": cast_aliases,
        "registry": identity_list}, indent=2, ensure_ascii=False))
    print(f"  identity map written: "
          f"{ {k: len(v) for k, v in cast_aliases.items()} }")

    locs = build_location_table(locations)
    ldir = ENGINE / "data" / "locations"
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / f"{pack_id}.json").write_text(json.dumps(locs, indent=2, ensure_ascii=False))

    preset = {
        "experiment_subname": "resim",
        "world_file_path": f"./data/worlds/{pack_id}/general.json",
        "loc_file_path": f"./data/locations/{pack_id}.json",
        "role_file_dir": "./data/roles/",
        "role_agent_codes": [code_of(n) for n in cast],
        "intervention": world["intervention"],
        "script": "", "language": args.language, "source": pack_id,
    }
    pdir = ENGINE / "presets"
    (pdir / f"{pack_id}.json").write_text(
        json.dumps(preset, indent=2, ensure_ascii=False))
    # provenance: which model/prompt generation built this pack
    (wdir / "provenance.json").write_text(json.dumps({
        "extract_model": GPT,
        "prompt_version": PROMPT_VERSION,
        "persona_scope": args.persona_scope,
        "language": args.language,
        "built_at": datetime.date.today().isoformat()},
        indent=2, ensure_ascii=False))
    print(f"package complete: preset={pack_id}.json, "
          f"{len(cast)} roles, {len(locs)} locations")
    print(f"intervention: {world['intervention'][:150]}")


if __name__ == "__main__":
    main()
