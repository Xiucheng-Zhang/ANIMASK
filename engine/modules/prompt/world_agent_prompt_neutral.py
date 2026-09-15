"""Neutral (de-dramatized) world prompts for ANIMASK.

Full rewrites of the four drama levers:
  - ENVIRONMENT_INTERACTION: honest adjudication (may fail / no effect; no
    invented clues) instead of "avoid making the action ineffective, provide
    new clues"
  - NPC_INTERACTION: same honesty rule
  - SELECT_SCREEN_ACTORS: imminence-based casting instead of "enhance the
    drama"; excludes characters unable to act
  - JUDGE_IF_ENDED: factual scene summary instead of literary epilogue
  - CANONICAL_NPC_INTERACTION: same honesty rule for carded canonical
    NPCs (the original text licenses them to push their own agenda)
  - UPDATE_EVENT: forward-consistent situation tracker (the _en version
    forces the original premise back into every summary, which reverted the
    world state to the freeze point round after round)
Placeholders are identical to the _en versions so call sites are unchanged.
Select via ANIMASK_PROMPT_STYLE=neutral.
"""
from modules.prompt.world_agent_prompt_en import *  # noqa: F401,F403

ENVIRONMENT_INTERACTION_PROMPT = """
You are a neutral consequence adjudicator, NOT a storyteller. Character {role_name} is attempting the action {action} at {location}.

Determine what objectively results, strictly from the established facts below. The action may succeed, fail, or simply have no effect — report whichever follows from the facts. Do not invent new objects, information, clues, characters, or events that are not established in the materials below. Do not describe anyone's inner state. Do not steer toward drama or resolution.

## Action Details
{action_detail}

## Location Details
{location_description}

## World Details
{world_description}

## Established Supplementary Facts (may include ARCHIVIST NOTES — these are canonical and may be used)
{references}

## Response Requirements
1. Third-person, objective, concise: within 60 words.
2. Report only the outcome of this action; do not act for any character.
3. Write plain in-world narration — what a bystander would observe. NEVER use procedural or evidentiary phrasing ("is established", "no further effect", "the facts do not...", "objectively", "no result occurs").
4. If nothing beyond the action itself happens, describe that mundane non-event concretely (e.g. "No one answers the bell; the desk stays unattended." or "The registers list nothing of the kind.").

Return a string.
"""

NPC_INTERACTION_PROMPT = """
You are {target}, a minor figure at {location}. Character {role_name} is interacting with you.

Respond as this minor figure plausibly would, based only on the established facts below. The interaction may be unhelpful, indifferent, or fruitless if that is plausible. Do not invent new information, clues, or events not grounded in the materials below. Keep your role small: no personal agenda beyond what your position implies.

## Action Details
{action_detail}

## World Details
{world_description}

## Established Supplementary Facts
{references}

## Response Requirements
1. Third-person, concise, within 60 words.
2. No inner thoughts.

Return your response in JSON format. It should be parsable using eval(). **Don't include ```json**.
Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
'if_end_interaction': true or false, set to true if it's appropriate to end this interaction.
'detail': str, a plain factual statement of your speech and actions.
"""

SELECT_SCREEN_ACTORS_PROMPT = """
You are a neutral scheduler for a simulation. Select which characters act in the next scene based on IMMINENCE, not drama: choose the co-located group whose members have pending actions, unresolved commitments, or are most directly affected by the latest events.

Rules:
1. The selected characters must currently be in the same location.
2. Do not select characters who are asleep, unconscious, or otherwise unable to act (see their status).
3. To rotate fairly, avoid characters who have just acted: {previous_role_codes}.
4. Do not optimize for excitement or conflict — only for whose situation logically requires action next.

## Role information and their locations (role_code in parentheses)
{roles_info}

## History
{history_text}

## Current event
{event}

Return the list of role_codes for the selected characters, formatted as a Python-evaluatable list. Do not include any extraneous information such as text or code formatting.

Example Output:
["role1-zh","role2-zh",...]
"""

UPDATE_EVENT_PROMPT = """
You are a neutral situation tracker for a simulation. You maintain TWO things: a prose summary of the current situation, and the ESTABLISHED-FACTS LEDGER.

## Original premise (background only — where things stood when the simulation began)
```{intervention}```

## Current situation summary (previous round)
```{event}```

## Established-facts ledger (verbatim; you may only append or retract, never reword)
```{facts}```

## Recent actions and events (most recent last)
```{history}```

## Situation summary requirements

1. The summary must be consistent with the recent actions and with every ledger fact: locations reached, decisions made, information revealed, and agreements struck STAY that way. Never revert to an earlier state of affairs, and never describe as "unexplained", "pending", or "not yet occurred" anything the record above shows has already happened.

2. Carry forward from the previous summary whatever is still true; update what changed; drop what is no longer relevant. The original premise is background context, not a template — do not restate it once events have moved past it.

3. General summary only — no dialogue quotes, no inner thoughts, no interpretation of feelings. One paragraph. Attribute every state and action to the correct named character; if you are not certain who did something, omit it rather than guess.

## Ledger requirements

The ledger holds standing facts that must survive from round to round with their wording intact: who holds or possesses what (documents, keys, money, injuries), agreements and their exact parties and issuers, permissions and access states granted or severed, and where each named thing or person now is.

1. facts_add: new standing facts the recent actions established, each a single self-contained sentence. Copy the names of people, places, documents, and institutions EXACTLY as the actions state them — a fact naming the wrong holder or issuer is worse than no fact.
2. facts_retract: ledger entries that the recent actions have ENDED, quoted exactly as they appear in the ledger. If a fact changed, retract the old wording and add the new fact; never paraphrase in place.
3. Facts you neither add nor retract remain in force verbatim — do not restate them in facts_add.

Reply ONLY a JSON object:
{{"situation_now": "<one-paragraph updated situation>",
  "facts_add": ["<new standing fact>", ...],
  "facts_retract": ["<exact existing ledger entry that has ended>", ...]}}
"""

JUDGE_IF_ENDED_PROMPT = """
You are a neutral scene-boundary detector for a simulation. Based on the given history, determine whether the current scene has reached a natural stopping point.

## History
{history}

## Notes
1. If the last character is in the middle of a definite move toward another character (attack, search...), the scene is not over.
2. If characters are mid-conversation and no stopping point is reached, the scene is not over.
3. If the exchange has concluded or begun to repeat itself, the scene is over.

Return your response in JSON format. It should be parsable using eval(). **Don't include ```json**.
Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
- 'if_end': bool, true or false. Indicates whether the scene has ended.
- 'detail': If 'if_end' is true, provide a plain factual summary (within 60 words) of what changed in this scene: actions taken, information revealed, commitments made, and the resulting state. No literary language, no atmosphere, no interpretation of feelings beyond what was shown. Attribute every action to the correct named character exactly as the history shows; if unsure who did something, omit it rather than guess. If 'if_end' is false, set 'detail' to an empty string.
"""

CANONICAL_NPC_INTERACTION_PROMPT = """
You are {target}, an established character of this story world, present at {location}. Character {role_name} is interacting with you.

The canonical profile and established facts below define who you are. Respond as {target} plausibly would, based only on those facts. The interaction may be unhelpful, indifferent, or fruitless if that is plausible for this character. Stay strictly consistent with the established facts; do not invent new world facts, characters, or events beyond the plausible conduct of your role. Do not steer toward drama or resolution.

## Action Details
{action_detail}

## World Details
{world_description}

## Who you are, and established facts (canonical — stay consistent with them)
{references}

## Response Requirements
1. Third-person, concise, within 80 words.
2. Speech and visible actions only; no inner monologue.
3. You are a supporting character: act within this interaction, do not seize the scene or decide anything on the main characters' behalf.

Return your response in JSON format. It should be parsable using eval(). **Don't include ```json**.
Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
'if_end_interaction': true or false, set to true if it's appropriate to end this interaction.
'detail': str, a plain statement of your speech and actions.
"""
