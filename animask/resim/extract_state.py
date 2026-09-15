"""Stage 1 of re-simulation: pick a freeze point and extract the scoped world
state + persona cards.

Knowledge scoping is enforced by construction: every extraction prompt sees
ONLY text[:freeze]. Nothing downstream of the freeze point ever enters any
agent's context.

Persona card structure follows CharaChat's L0/L1 design (static profile +
behavioral axioms) with per-character private knowledge for information
asymmetry.

Usage: python3 -m animask.resim.extract_state <story_id> [freeze_fraction]
  (run from the repository root; default freeze picked by LLM near 55-65%)
"""
import json
import re
import sys
from pathlib import Path

from animask.llm import GPT, SONNET, query  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
# the freeze pick is the one full-text call whose result is reusable: cache
# it so a failure later in the stage (state extraction) never re-picks
CACHE_DIR = ROOT / "data" / "cache" / "extract_state"

FREEZE_PROMPT = (
    "The piped text is a complete short story. Identify the best FREEZE POINT "
    "for a character-simulation experiment: a moment 50-70% through the story "
    "where (a) the main characters and their goals are fully established, "
    "(b) a pivotal character decision or confrontation is imminent but has NOT "
    "yet happened, and (c) the remaining outcome depends chiefly on what the "
    "characters choose to do next.\n"
    "Reply ONLY a JSON object:\n"
    '{"freeze_quote": "<exact short quote (5-15 words) from the story marking '
    'the freeze moment>", "approx_fraction": 0.55, "imminent_decision": '
    '"<one sentence: what is about to be decided>", "rationale": "<one sentence>"}'
)

STATE_PROMPT = (
    "The piped text is the beginning of a story, cut off at a freeze point. "
    "You will see NOTHING beyond the cut. Build a simulation state from ONLY "
    "this text.\n"
    "Reply ONLY a JSON object:\n"
    "{\n"
    '  "world": {\n'
    '    "setting": "<where/when, physical and social rules that matter>",\n'
    '    "situation_now": "<what is happening at the exact freeze moment: who '
    'is where, doing what>",\n'
    '    "open_tensions": ["<unresolved tension>", ...],\n'
    '    "constraints": ["<physical/social constraint that limits available actions>", ...]\n'
    "  },\n"
    '  "narration_style": "<1-2 sentences on the prose voice, for later rendering>",\n'
    '  "characters": [\n'
    "    {\n"
    '      "name": "<name>",\n'
    '      "simulate": true|false,  // true for characters with real agency in the current scene\n'
    '      "identity": "<who they are, role, appearance if relevant>",\n'
    '      "traits": ["<stable personality trait>", ...],\n'
    '      "goals": {"long_term": "...", "right_now": "..."},\n'
    '      "relationships": {"<other name>": "<attitude, history, tension>"},\n'
    '      "speech_style": "<how they talk, verbal tics, register>",\n'
    '      "axioms": ["When facing X, they do Y (grounded in a scene from the text)", ...],\n'
    '      "private_knowledge": ["<things ONLY this character knows: secrets, plans, feelings — state each fact ITSELF with its actual names and content, never a reference like \'the plan they intend to propose\'>", ...],\n'
    '      "does_not_know": ["<facts other characters know but this one does not>", ...],\n'
    '      "state_now": "<location, physical/emotional state at freeze>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Include AT MOST 8 character entries: the characters with real agency in "
    "the current situation (simulate=true, at most 6) plus the most relevant "
    "background figures (simulate=false). Fold all remaining minor figures "
    "into a single sentence inside world.situation_now instead of giving them "
    "entries. Ground axioms in actual scenes from the text."
)


def jparse_retry(prompt: str, stdin: str, tries: int = 3,
                 require: list | None = None, max_tokens: int = 4096):
    """Robust parse with re-query and schema check: parse from the first
    brace; a result missing any required key counts as a failure too
    (catches truncated / partially-formed outputs).

    A backend/transport failure (<<ERROR) is NOT a parse failure: llm.query
    has already retried it, and re-sending the full text here only
    multiplies the bill — fail the stage fast instead (batch_runner /
    supervisor resume it later)."""
    for attempt in range(tries):
        out = query(prompt, model=GPT, stdin=stdin, max_tokens=max_tokens)
        if (out or "").startswith("<<ERROR"):
            raise RuntimeError(f"backend error: {out[:300]}")
        m = re.search(r"\{", out or "")
        if m:
            try:
                obj = json.JSONDecoder().raw_decode(out[m.start():])[0]
                if not require or all(k in obj for k in require):
                    return obj
                print(f"  (missing keys {set(require) - set(obj)}, "
                      f"attempt {attempt + 1}/{tries})")
                continue
            except json.JSONDecodeError:
                pass
        print(f"  (parse failure, attempt {attempt + 1}/{tries}; "
              f"reply head: {(out or '')[:200]!r})")
    raise ValueError(f"unparseable after {tries} tries: {prompt[:80]}...")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m animask.resim.extract_state <sid> [freeze_fraction]")
    sid = sys.argv[1]
    story_p = ROOT / "data" / "clean" / f"{sid}.txt"
    if not story_p.exists():
        raise SystemExit(f"story text not found: {story_p} "
                         "(one plain-text file per story under data/clean/)")
    text = story_p.read_text()
    words = text.split()
    n = len(words)

    if len(sys.argv) > 2:
        frac = float(sys.argv[2])
        freeze_info = {"approx_fraction": frac, "freeze_quote": "", "manual": True}
    else:
        freeze_cache = CACHE_DIR / f"{sid}.freeze.json"
        if freeze_cache.exists():
            print("freeze point from cache", freeze_cache)
            freeze_info = json.loads(freeze_cache.read_text())
        else:
            print("picking freeze point ...")
            freeze_info = jparse_retry(FREEZE_PROMPT, text,
                                       require=["approx_fraction"])
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            freeze_cache.write_text(json.dumps(freeze_info, indent=2,
                                               ensure_ascii=False))
        frac = float(freeze_info["approx_fraction"])
        # snap to the quote's actual position — but only when it lands near
        # the model's own estimate (a quote matching an EARLIER similar line
        # would otherwise drag the freeze far out of the 50-70% target zone)
        q = freeze_info.get("freeze_quote", "")
        if q and freeze_info.get("freeze_word_pinned") is None:
            pos = text.find(q)
            if pos >= 0:
                snapped = len(text[:pos].split()) / n
                if abs(snapped - frac) <= 0.15:
                    frac = snapped
                    freeze_info["snapped_fraction"] = round(frac, 3)
                else:
                    freeze_info["snap_rejected"] = round(snapped, 3)

    # a pinned word index (seeded by strip_script_headers --apply) overrides
    # the fraction arithmetic entirely: the freeze is never re-chosen after a
    # header strip, only shifted — and int(n * cut/n) can drop a word to
    # floating point, so the exact index is authoritative
    pinned = (freeze_info or {}).get("freeze_word_pinned")
    cut = int(pinned) if pinned is not None else int(n * frac)
    pre_text = " ".join(words[:cut])
    print(f"freeze at word {cut}/{n} ({frac:.0%})  quote={freeze_info.get('freeze_quote', '')!r}")

    print("extracting scoped world state + persona cards (sees pre-freeze text only) ...")
    # up to 8 character cards with many list fields: 4096 output tokens can
    # truncate the JSON (-> parse failure -> another full-text send)
    state = jparse_retry(STATE_PROMPT, pre_text,
                         require=["characters", "world"], max_tokens=8192)

    sim_chars = [c["name"] for c in state["characters"] if c.get("simulate")]
    result = {"story_id": sid, "freeze": freeze_info, "freeze_word": cut,
              "total_words": n, "state": state}
    out_path = ROOT / "data" / "meta" / f"resim_{sid}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"simulated characters: {sim_chars}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
