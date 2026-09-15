"""Neutral (de-literarized) role prompts for ANIMASK.

Derived from role_agent_prompt_en by surgically removing the drama pumps:
  - "engaging dialogue / advance the plot"
  - "create tension, resolve issues, or introduce dramatic twists"
  - "do not prepare/confirm; act immediately and draw conclusions"
  - "literary narrative-style" output framing
Everything else (format contracts, JSON fields) is kept identical so the
engine parses outputs unchanged. Select via ANIMASK_PROMPT_STYLE=neutral.
"""
from modules.prompt.role_agent_prompt_en import *  # noqa: F401,F403
import modules.prompt.role_agent_prompt_en as _en

_REPLACEMENTS = [
    ("The dialogue should be engaging, advance the plot, and reveal the "
     "character's emotions, intentions, or conflicts.",
     "Respond only as the character genuinely would in this situation."),
    ("4. **Substance:** Ensure your responses are meaningful, create tension, "
     "resolve issues, or introduce dramatic twists.",
     "4. **Fidelity over drama:** Act strictly from your goals, knowledge, and "
     "personality. Caution, hesitation, refusal, waiting, or de-escalation are "
     "all acceptable whenever they are what the character would genuinely do. "
     "Never add drama for its own sake."),
    ("5. **Avoid Repetition:** Avoid repeating information from the dialogue "
     "history, and refrain from vague or generic responses. Do not "
     "“prepare,” “ask for opinions,” or “confirm”; "
     "instead, act immediately and draw conclusions.",
     "5. **No filler:** Do not restate information already present in the "
     "history."),
    ("A literary narrative statement containing",
     "A plain factual statement containing"),
    ("a literary narrative-style statement containing",
     "a plain factual statement containing"),
    ("a literary and narrative description that includes",
     "a plain factual description that includes"),
]

_TARGETS = ["ROLE_PLAN_PROMPT", "ROLE_SINGLE_ROLE_RESPONSE_PROMPT",
            "ROLE_MULTI_ROLE_RESPONSE_PROMPT", "ROLE_NPC_RESPONSE_PROMPT"]

_unused = {old for old, _ in _REPLACEMENTS}
for _name in _TARGETS:
    _p = getattr(_en, _name)
    for _old, _new in _REPLACEMENTS:
        if _old in _p:
            _unused.discard(_old)
            _p = _p.replace(_old, _new)
    globals()[_name] = _p

if _unused:
    print(f"[neutral prompts] WARNING: {len(_unused)} replacement anchors not "
          f"found (the original prompt text may have changed): "
          f"{[u[:60] for u in _unused]}")
