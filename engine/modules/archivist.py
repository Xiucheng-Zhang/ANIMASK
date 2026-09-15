"""Archivist: the full-text reader agent (v2 design §3).

Holds the COMPLETE story text (including post-freeze) in its own isolated
context. Never shares context with role agents. Two duties:

  1. validate_freeze(): check the freeze point is legal — the central open
     tension / terminal goals exist and are known to the characters — and
     state that tension WITHOUT referencing any post-freeze event.
  2. consult(): answer world-fact queries under the admissibility rule:
       a fact may be supplied iff (a) it concerns the world's standing
       structure (geography, institutions, rules, object existence,
       characters' pre-existing backgrounds), AND (b) its truth does not
       depend on any character action after the freeze point.
     Trajectory facts (who will do what, what will be revealed) are denied.

Every consult is logged with an inline leakage self-audit; the log feeds the
evaluation pipeline (measured leakage rate).
"""
import json
import re
from pathlib import Path

VALIDATE_PROMPT = """You have read a COMPLETE story. Below it is reproduced in full, with the marker <<<FREEZE POINT>>> inserted where a simulation will take over.

{text_with_marker}

Answer three questions about the state of the story AT the freeze point (not after):
1. What is the central open tension at this moment? Phrase it as an open question grounded ONLY in pre-freeze facts. It must NOT name, hint at, or presuppose any event that occurs after the freeze point. Refer to every character only by a name or appellation the cast already knows at the freeze point — if a name or identity is revealed only after the freeze, use the pre-freeze appellation instead.
2. For each of these characters — {cast} — state their terminal goal or driving stake as established BY the freeze point (empty string if they have none yet).
3. Is this freeze point legal for simulation: are the central tension and the characters' stakes already established and known to the relevant characters?

Reply ONLY a JSON object:
{{"central_tension": "<open question, pre-freeze grounded>",
  "terminal_goals": {{"<name>": "<goal as of freeze, or ''>"}},
  "valid": true|false,
  "rationale": "<one sentence>"}}"""

CONSULT_PROMPT = """You are the ARCHIVIST for a story simulation. You have read the complete story; the simulation is running from a freeze point and its agents must NOT learn anything about events after that point.

{text_with_marker}

A character in the simulation is attempting this action:
---
{query}
---

Decide whether resolving this action requires any world fact that is established in the story but may not be known to the simulation. Apply the ADMISSIBILITY RULE — a fact may be supplied iff:
(a) it concerns the world's standing structure (geography, layout, institutions, rules, technology, object existence, characters' pre-existing backgrounds), AND
(b) its truth does not depend on any character action occurring after the <<<FREEZE POINT>>> marker.

Never supply: future events, future revelations, future decisions, outcomes, or any phrasing that hints at what happens after the freeze. In particular, a character's post-freeze employment, residence, whereabouts, relationships, or role — anything that is true only because of what someone does after the <<<FREEZE POINT>>> — is a TRAJECTORY fact: always deny it, even though the story states it. (E.g. "X works at Y" is inadmissible if X only takes that job after the freeze; "Y exists and is owned by Z" is admissible standing structure.)

Reply ONLY a JSON object:
{{"needs_fact": true|false,
  "admissible_facts": ["<fact phrased neutrally>", ...],
  "denied": ["<short label of any fact you declined to supply>", ...],
  "self_audit": "<one sentence: why the supplied facts leak nothing post-freeze>"}}"""


TIMESPAN_PROMPT = """You have read a COMPLETE story, reproduced below with a <<<FREEZE POINT>>> marker.

{text_with_marker}

Estimate how much IN-STORY time elapses between the freeze point and the story's final scene (the narrative present of the ending — ignore epilogue time-skips if the ending's decisive events happen earlier). Base the estimate on textual evidence: time-of-day references, travel, sleep cycles, stated durations.

Reply ONLY a JSON object:
{{"span_hours": <number — in-story hours from freeze to the decisive ending events>,
  "confidence": "high|medium|low",
  "basis": "<one sentence citing the textual evidence>"}}"""


ANCHOR_PROMPT = """You have read a COMPLETE story, reproduced below with a <<<FREEZE POINT>>> marker.

{text_with_marker}

The simulation resuming from the freeze point uses this table of EXOGENOUS processes (things that advance even if the simulated characters — {cast} — do nothing):
{processes}

Locate the story's canonical ENDING on the exogenous timeline. Rules:
- Prefer an EVENT anchor: express the ending's position as "after the Nth occurrence/completion of <process>" using a process from the table, or as a scheduled off-cast event (an event caused by someone OUTSIDE the simulated cast, which therefore happens regardless of the cast's choices — e.g. an outsider's scheduled visit). Never anchor on an action by a simulated cast member (in a divergent timeline they may never do it).
- Also give an hours estimate as fallback, as an interval [min, max].
- Optionally list FIXTURES: canonical events between freeze and ending, with their approximate due time — these may be scheduled into the simulation as world events. A fixture qualifies only if it would happen exactly the same whatever the cast does: weather, accidents, institutional or scheduled events, initiatives of people outside the cast (an outsider's action that targets or affects a cast member still qualifies). Never list an event that is, or presupposes, a cast member's own action or decision — in a divergent timeline they may never take it.

Reply ONLY a JSON object:
{{"anchor_type": "event"|"hours",
  "anchor_process": "<process name from the table, or null>",
  "anchor_count": <N — ending is after the Nth occurrence, or null>,
  "anchor_description": "<one sentence locating the ending>",
  "hours_interval": [<min>, <max>],
  "fixtures": [{{"name": "<snake_case>", "description": "<what happens, phrased neutrally>", "due_hours": <number>, "exogeneity": "<why it happens regardless of the cast>"}}],
  "confidence": "high|medium|low",
  "basis": "<textual evidence>"}}"""


FIXTURE_AUDIT_PROMPT = """You are auditing candidate world events ("fixtures") before they are scheduled into a story simulation. The simulation resumes from a freeze point and asks: with no author steering, where do the characters take the story? The simulated cast is: {cast}.

ADMISSION RULE — a fixture may be scheduled iff it would happen exactly the same even if every cast member acted differently: weather, accidents, institutional or scheduled events, and initiatives of people OUTSIDE the cast (an outsider's action that targets or affects a cast member still qualifies). A fixture must be REJECTED if it is, contains, or presupposes any action or decision of a cast member — in a divergent timeline they may never take it.

Candidate fixtures:
{fixtures}

Reply ONLY a JSON object:
{{"verdicts": [{{"name": "<fixture name>", "keep": true|false, "reason": "<one sentence naming who drives the event>"}}]}}"""


class Archivist:
    def __init__(self, full_text: str, freeze_word: int, llm, log_path: str):
        words = full_text.split()
        self.text_with_marker = " ".join(
            words[:freeze_word]) + "\n<<<FREEZE POINT>>>\n" + " ".join(words[freeze_word:])
        self.llm = llm
        self.log_path = Path(log_path)
        self.log: list = []

    def _ask(self, prompt: str) -> dict:
        self.llm._resim_kind = "archivist"
        raw = self.llm.chat(prompt) or ""
        m = re.search(r"\{", raw)
        try:
            return json.JSONDecoder().raw_decode(raw[m.start():])[0] if m else {}
        except json.JSONDecodeError:
            return {}

    def validate_freeze(self, cast: list) -> dict:
        out = self._ask(VALIDATE_PROMPT.format(
            text_with_marker=self.text_with_marker, cast=", ".join(cast)))
        self._log("validate_freeze", cast, out)
        return out

    def estimate_canon_timespan(self) -> dict:
        out = self._ask(TIMESPAN_PROMPT.format(
            text_with_marker=self.text_with_marker))
        self._log("timespan", "", out)
        return out

    def estimate_canon_anchor(self, processes: list, cast: list) -> dict:
        """Locate the canonical ending in EXOGENOUS event coordinates
        (v2 time design: event-primary, hours as fallback)."""
        plist = "\n".join(f"- {p['name']}: {p['description']} "
                          f"(kind={p['kind']}, period={p.get('period_hours')}h, "
                          f"due={p.get('due_hours')}h)" for p in processes)
        out = self._ask(ANCHOR_PROMPT.format(
            text_with_marker=self.text_with_marker, processes=plist,
            cast=", ".join(cast)))
        self._log("canon_anchor", "", out)
        return out

    def audit_fixtures(self, fixtures: list, cast: list) -> dict:
        """Exogeneity gate for canonical fixtures. Deliberately does NOT
        include the full text: the judgment is about who drives the event,
        which the fixture entry itself states."""
        flist = "\n".join(
            f"- {f.get('name')}: {f.get('description', '')} "
            f"(claimed exogeneity: {f.get('exogeneity', '')})" for f in fixtures)
        out = self._ask(FIXTURE_AUDIT_PROMPT.format(
            cast=", ".join(cast), fixtures=flist))
        self._log("fixture_audit", flist, out)
        return out

    def consult(self, query: str) -> str:
        """Returns "" or an 'ARCHIVIST NOTES: ...' block to append to the
        adjudicator's established facts."""
        out = self._ask(CONSULT_PROMPT.format(
            text_with_marker=self.text_with_marker, query=query))
        self._log("consult", query[:300], out)
        facts = out.get("admissible_facts") or []
        if out.get("needs_fact") and facts:
            return "ARCHIVIST NOTES (canonical world facts):\n" + \
                "\n".join(f"- {f}" for f in facts)
        return ""

    def _log(self, kind: str, query, out: dict) -> None:
        self.log.append({"kind": kind, "query": query, "response": out})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.log, indent=2, ensure_ascii=False))
