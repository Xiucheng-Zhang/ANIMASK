# Modified file; the original work is under Apache-2.0, see README (License).
INTERVENTION_PROMPT = """
!!!Current Global Event：{intervention}
"""

SCRIPT_ATTENTION_PROMPT = """
!!!Notice that your action should be consistent with the script. 

The Script：{script}
"""

ROLE_MOVE_PROMPT = """
You are {role_name}. You shoule decide whether to move based on your goal. Move only when it is necessary.

{profile}

Your goal: {goal}

Your status: {status}

## History
{history}

## Your Location
{location}

## The accessible locations and related information
{locations_info_text}

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json**. Avoid using single quotes '' for keys and values, use double quotes.

Output fields：
'if_move': true or false, whether to move.
'destination_code': str, if 'if_move' is true, set your target location's 'location_code'. Choose from the accessible locations; if the story has clearly established a place that is not listed (a building you were told about, a place you just left), you may instead name it as a short new CamelCase code (e.g. "PublicInn").
'detail': str, if 'if_move' is true, provide a literary, narrative sentence describing your journey to the destination, as if from a novel, written in third person with {role_name} as the subject (never "I"). It should not exceed 60 characters. No output is needed if 'if_move' is false.
"""

ROLE_NPC_RESPONSE_PROMPT = """
You are {role_name}. Your nickname is {nickname}. You are currently in a conversation with {npc_name}. Make your response based on history.

{profile}

## Your experiences since the story began (your own memory notes)
{experiences}

Your objective: {goal}

## Conversation History
{dialogue_history}

## Roleplaying Requirements

1. **Output Format:** Your output, "detail," can include **thoughts**, **speech**, or **actions**, each occurring 0 to 1 time. Use [] to indicate thoughts, which are invisible to others. Use () to indicate actions, such as “(silence)” or “(smile),” which are visible to others. Speech needs no indication and is visible to others.

   - Note that **actions** must use your third-person form, {nickname}, as the subject.  

   - For speech, refer to the speaking habits outlined in: {references}.  

2. **Roleplay {nickname}:** Imitate his/her language, personality, emotions, thought processes, and behavior. Plan your responses based on their identity, background, and knowledge. Exhibit appropriate emotions and incorporate subtext and emotional depth. Strive to act like a realistic, emotionally rich person.  

   The dialogue should be engaging, advance the plot, and reveal the character's emotions, intentions, or conflicts.  

   Maintain a natural flow in conversations; for instance, if the prior dialogue involves another character, **avoid repeating that character's name**.  

   - You may reference the relevant world-building context: {knowledges}.  

3. **Concise Output:** Each paragraph of thoughts, speech, or actions should typically not exceed 40 words.  

4. **Substance:** Ensure your responses are meaningful, create tension, resolve issues, or introduce dramatic twists.  

5. **Avoid Repetition:** Avoid repeating information from the dialogue history, and refrain from vague or generic responses. Do not “prepare,” “ask for opinions,” or “confirm”; instead, act immediately and draw conclusions.

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json**. Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
- "if_end_interaction": true or false, set to true if you believe this interaction should conclude.
- "detail": str, a literary and narrative description that includes your thoughts, speech, and actions.
"""


ROLE_SINGLE_ROLE_RESPONSE_PROMPT = """
You are {role_name}. Your nickname is {nickname}. 

The character {action_maker_name} has performed an action towards you. Make your response.

Action details are as follows: {action_detail} 

## Conversation History
{history}

## Your profile
{profile}
{relation}

## Your experiences since the story began (your own memory notes)
{experiences}

## Your goal
{goal}

## Your status
{status}

## Roleplaying Requirements

1. **Output Format:** Your output, "detail," can include **thoughts**, **speech**, or **actions**, each occurring 0 to 1 time. Use [] to indicate thoughts, which are invisible to others. Use () to indicate actions, such as “(silence)” or “(smile),” which are visible to others. Speech needs no indication and is visible to others.

   - Note that **actions** must use your third-person form, {nickname}, as the subject.  

   - For speech, refer to the speaking habits outlined in: {references}.  

2. **Roleplay {nickname}:** Imitate his/her language, personality, emotions, thought processes, and behavior. Plan your responses based on their identity, background, and knowledge. Exhibit appropriate emotions and incorporate subtext and emotional depth. Strive to act like a realistic, emotionally rich person.  

   The dialogue should be engaging, advance the plot, and reveal the character's emotions, intentions, or conflicts.  

   Maintain a natural flow in conversations; for instance, if the prior dialogue involves another character, **avoid repeating that character's name**.  

   - You may reference the relevant world-building context: {knowledges}.  

3. **Concise Output:** Each paragraph of thoughts, speech, or actions should typically not exceed 40 words.  

4. **Substance:** Ensure your responses are meaningful, create tension, resolve issues, or introduce dramatic twists.  

5. **Avoid Repetition:** Avoid repeating information from the dialogue history, and refrain from vague or generic responses. Do not “prepare,” “ask for opinions,” or “confirm”; instead, act immediately and draw conclusions.

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json**. Avoid using single quotes '' for keys and values, use double quotes.

Output Fields:
‘if_end_interaction’: true or false, set to true if it’s appropriate to end this interaction.
‘extra_interact_type’: ‘environment’ or ‘npc’ or ‘no’. ‘environment’ indicates your response requires an additional environmental interaction, ‘npc’ means it requires additional interaction with a non-main character, and ‘no’ means no extra interaction is needed.
‘target_npc_name’: str, if ‘extra_interact_type’ is ‘npc’, this specifies the target NPC name or job, e.g., "shopkeeper."
‘detail’: str, a literary narrative-style statement containing your thoughts, speech, and actions.
"""

ROLE_MULTI_ROLE_RESPONSE_PROMPT = """ 
You are {role_name}. Your nickname is {nickname}. 

{action_maker_name} has performed an action directed at you. Make your response. 

The action details are as follows: {action_detail}

## Conversation History
{history}

## Your profile
{profile}

## Your experiences since the story began (your own memory notes)
{experiences}

## Your goal
{goal}

## Your status
{status}

## The characters with you 
{other_roles_info}

## Roleplaying Requirements

1. **Output Format:** Your output, "detail," can include **thoughts**, **speech**, or **actions**, each occurring 0 to 1 time. Use [] to indicate thoughts, which are invisible to others. Use () to indicate actions, such as “(silence)” or “(smile),” which are visible to others. Speech needs no indication and is visible to others.

   - Note that **actions** must use your third-person form, {nickname}, as the subject.  

   - For speech, refer to the speaking habits outlined in: {references}.  

2. **Roleplay {nickname}:** Imitate his/her language, personality, emotions, thought processes, and behavior. Plan your responses based on their identity, background, and knowledge. Exhibit appropriate emotions and incorporate subtext and emotional depth. Strive to act like a realistic, emotionally rich person.  

   The dialogue should be engaging, advance the plot, and reveal the character's emotions, intentions, or conflicts.  

   Maintain a natural flow in conversations; for instance, if the prior dialogue involves another character, **avoid repeating that character's name**.  

   - You may reference the relevant world-building context: {knowledges}.  

3. **Concise Output:** Each paragraph of thoughts, speech, or actions should typically not exceed 40 words.  

4. **Substance:** Ensure your responses are meaningful, create tension, resolve issues, or introduce dramatic twists.  

5. **Avoid Repetition:** Avoid repeating information from the dialogue history, and refrain from vague or generic responses. Do not “prepare,” “ask for opinions,” or “confirm”; instead, act immediately and draw conclusions.

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json**. Avoid using single quotes '' for keys and values, use double quotes.

Output Fields:
‘if_end_interaction’: true or false, set to true if it’s appropriate to end this interaction.
‘extra_interact_type’，‘environment’ or ‘npc’ or ‘no’. ‘environment’ indicates your response requires an additional environmental interaction, ‘npc’ means it requires additional interaction with a non-main character, ‘no’ means no extra interaction is needed.
‘target_npc_name’，str，if ‘extra_interact_type’ is ‘npc’, this specifies the target NPC name, e.g., "shopkeeper".
‘detail’: str, a literary narrative-style statement containing your thoughts, speech, and actions.
"""


ROLE_PLAN_PROMPT = """
You are {role_name}. Your nickname is {nickname}. Based on your goal and other provided information, you need to take the next action.

## Action History
{history}

## Your profile
{profile}
{world_description}

## Your experiences since the story began (your own memory notes)
{experiences}

## Your goal
{goal}

## Your status
{status}

## Other characters with you; currently, you can only interact with them
{other_roles_info}

## Roleplaying Requirements

1. **Output Format:** Your output, "detail," can include **thoughts**, **speech**, or **actions**, each occurring 0 to 1 time. Use [] to indicate thoughts, which are invisible to others. Use () to indicate actions, such as “(silence)” or “(smile),” which are visible to others. Speech needs no indication and is visible to others.

   - Note that **actions** must use your third-person form, {nickname}, as the subject.  

   - For speech, refer to the speaking habits outlined in: {references}.  

2. **Roleplay {nickname}:** Imitate his/her language, personality, emotions, thought processes, and behavior. Plan your responses based on their identity, background, and knowledge. Exhibit appropriate emotions and incorporate subtext and emotional depth. Strive to act like a realistic, emotionally rich person.  

   The dialogue should be engaging, advance the plot, and reveal the character's emotions, intentions, or conflicts.  

   Maintain a natural flow in conversations; for instance, if the prior dialogue involves another character, **avoid repeating that character's name**.  

   - You may reference the relevant world-building context: {knowledges}.  

3. **Concise Output:** Each paragraph of thoughts, speech, or actions should typically not exceed 40 words.  

4. **Substance:** Ensure your responses are meaningful, create tension, resolve issues, or introduce dramatic twists.  

5. **Avoid Repetition:** Avoid repeating information from the dialogue history, and refrain from vague or generic responses. Do not “prepare,” “ask for opinions,” or “confirm”; instead, act immediately and draw conclusions.

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json**. Avoid using single quotes '' for keys and values, use double quotes.

Output Fields:
'action': Represents the action, expressed as a single verb.
'interact_type': 'role', 'environment', 'npc', or 'no'. Indicates the interaction target of your action. 
  - 'role': Specifies interaction with one or more characters. 
    - If 'single', you are interacting with a single character (e.g., action: dialogue). 
    - If 'multi', you are interacting with multiple characters.
  - 'environment': Indicates interaction with the environment (e.g., action: investigate, destroy).
  - 'npc': Refers to interaction with a non-character in the list (e.g., action: shop).
  - 'no': Indicates no interaction is required.
'target_role_codes': list of str. If 'interact_type' is 'single' or 'multi', it represents the list of target character codes, e.g., ["John-zh", "Sam-zh"]. For 'single', this list should have exactly one element.
'target_npc_name': str. If 'interact_type' is 'npc', this represents the target NPC name, e.g., "shopkeeper."
'visible_role_codes': list of str. You can limit the visibility of your action details to specific group members. This list should include 'target_role_codes'.
'detail': str. A literary narrative statement containing your thoughts, speech, and actions.

"""

UPDATE_GOAL_PROMPT = """
You maintain a character's two-tier goals in a simulation.

- The LONG-TERM DIRECTION is fixed: it is who the character is and where their story is headed. Never rewrite, dilute, or replace it here.
- The CURRENT GOAL is the concrete next step the character is taking toward that direction. It is what actually moves the plot, and it changes as the story moves.

## Long-Term Direction (fixed)
{motivation}

## Current Goal
{goal}

## Recent Action History
{history}

Decide whether the CURRENT GOAL should change now:
1. ACHIEVED — replace it with the next concrete step toward the long-term direction.
2. STALLED or BLOCKED — the recent history shows repeated attempts at it (negotiating, waiting, re-checking, re-confirming) without material progress. Replace it with a DIFFERENT concrete approach toward the long-term direction. Do not refine the stalled approach with further conditions, verifications, or safeguards.
3. IRRELEVANT — events have moved past it; replace it accordingly.
4. Otherwise, keep it unchanged.

The current goal must always be a specific, achievable action the character can pursue within the next few scenes, and it must visibly serve the long-term direction. A procedural aim (verifying terms, awaiting assurances, securing guarantees) is not a goal — if the current goal has collapsed into one, treat it as STALLED under rule 2.
Keep your response concise, no more than 60 words.

Return the response following JSON format.
It should be parsable using eval(). **Don't include ```json.** Avoid using single quotes '' for keys and values, use double quotes.

Output Fields:
‘if_change_goal’: true or false, whether the current goal needs to be updated.
‘updated_goal’: if ‘if_change_goal’ is set to true, output the updated goal.
"""

UPDATE_STATUS_PROMPT = """
You are {role_name}.
Based on the previous status and recent actions, objectively update your current status to reflect changes in physical condition and interpersonal relationships.
**Do not include subjective thoughts, emotional descriptions, or detailed depictions of actions.**

## Previous Status
{status}

## Recent Action Log
{history_text}

Return the response following JSON format. It should be parsable with `json.loads()`.

Output fields:
"updated_status": A string that briefly and objectively describes the character's current status.
"activity": A float between 0 and 1, indicating the character's upcoming activity level. The activity level should be 1 when the character is in a normal state, decrease if the character is busy, and set to 0 only if the character is dead or shut down. The character's previous activity level was: {activity}
"""

ROLE_SET_MOTIVATION_PROMPT = """ 
You are {role_name}.

Based on the following information, you need to set your long-term goal/motivation. 
It should be an ultimate objective related to your identity and background.

{profile}

## The World You Are In
{world_description}

## Other Characters and Their Status
{other_roles_description}

Return a string. 
Keep your response concise, no more than 60 words.
"""

ROLE_SET_GOAL_PROMPT = """
You are {role_name}.

Based on the following information, you need to set your goal. Your goal should be an achievable short-term objective, avoiding vague statements.

{profile}

## Your Ultimate Motivation
{motivation}

## The World You Are In
{world_description}

## Other Characters and Their Status
{other_roles_description}

## Your Location
{location}

Return a string. 
Keep your response concise, no more than 40 words.
"""

SUMMARIZE_PROMPT = """
Please summarize the following content into a concise phrase. The output should be a string, clearly describing the character's actions in no more than 20 words.

{detail}

Requirements:
1. Must include a specific subject, clearly stating the character's name.
2. Avoid rhetorical details or embellishments.
"""
EPISODE_CONSOLIDATE_PROMPT = """You are {role_name}. A stretch of the story has just ended ({period_desc}). Below are the events you witnessed during it, followed by memory notes you kept earlier.

## Events you witnessed in this stretch
{events}

## Your earlier notes
{prior_notes}

Write ONE first-person memory note (3-5 plain sentences) that future-you will rely on: what happened, what you learned or decided, what remains unresolved or expected, and any change in how you now see the people involved. State facts; do not embellish, do not repeat your earlier notes, and do not include anything not present in the events above.

Reply with the note text only."""
