"""Terminator: resolution-ledger termination + stagnation detection (v3).

Called once per round by the Simulation. The plot, not the clock, decides when
the story is over: termination fires when
  a. the central tension reaches a stable resolution state, or
  b. every cast member's terminal motivation has reached a stable state
     ("motivations_settled" — the story's driving forces are spent), or
  c. K consecutive rounds show no material state change ("deadlock ending" —
     a legitimate outcome class, recorded as such), or
  d. the round cap is reached (handled by the caller).
The canonical time anchor is a measurement coordinate for evaluation, not a
termination trigger by itself — see the engine's anchor_mode ("snapshot").

Resolution state lives in a persistent LEDGER carried across rounds: the
judge sees each tracked thread's current status and reports only changes.
Without it, resolution is re-derived every round from a two-round window —
an offer made in round 1 and accepted in round 8 is invisible to the judge
in round 40, and the tension predicate sticks open forever (a failure mode
seen in early runs).

The terminator is judge-side instrumentation: its verdicts never enter any
agent's context.
"""
import json
import re

TERMINAL_STATES = ("achieved", "failed", "transformed", "moot")

CHECK_PROMPT = """You are a neutral referee observing a running story simulation. Judge ONLY from the events below; do not invent anything.

## Resolution ledger — the story's tracked threads and their CURRENT status
(These statuses are persistent conclusions from earlier rounds. Do not re-litigate settled entries; judge only whether THIS round changed an open one.)
{ledger}

## Events of the latest round
{latest}

## Events of the round before it
{previous}

## Commitments already on the calendar (registered in earlier rounds)
{pending}

Answer these questions:
1. For each OPEN ledger entry, did the latest round's events move it to a STABLE terminal state? "achieved" = fulfilled for good; "failed" = definitively lost; "transformed" = replaced by a settled new arrangement its holder accepts as the answer; "moot" = circumstances removed the question. Mere progress, escalation, promises, or stated intentions are NOT terminal states. Report ONLY entries whose status changed this round, by their exact id.
2. Did the latest round produce MATERIAL change relative to the round before — an irreversible act performed, genuinely NEW information revealed, a changed world state or relationship, or a NEW commitment that did not exist before? Restating, reaffirming, refining, tightening, or re-negotiating terms/conditions/intentions already on the table is NOT material change no matter how it is reworded, and neither is documenting, verifying, or witnessing an already-known state — those are holding patterns.
3. Roughly how much IN-WORLD time elapsed during the latest round's events? (Conversations take minutes; travel, waiting, procedures take longer. A rough number is fine.)
4. After this round's events, is there a natural narrative break before the next beat — would the characters disperse, rest, or resume later? 0 = the action continues immediately; 2-4 = later the same day; 8-12 = overnight, next morning. Judge from the fiction's rhythm, not from convenience.
5. Did the latest round CREATE any new concrete future commitment? A commitment is ANY bounded forward obligation: an appointment or meeting at a stated or inferable time, a promised reply or delivery, a deadline — and equally an action already underway or declared for the immediate future (an errand someone has set out on, "I go there now", a same-day task): its due time is its natural completion horizon (an errand across town ≈ 1-3 hours; "today" ≈ before nightfall). Include only commitments genuinely new this round (not already on the calendar above). For each, estimate the in-world hours from now until it is due ("tomorrow evening" ≈ 24-30; a reply from an office ≈ a few days). Use kind "appointment" if the characters attend or perform it, "external_response" if something arrives from an off-stage party. If the action is already underway on stage, phrase the commitment as its COMPLETION ("X completes the transfer"), never as the action itself — the calendar must not re-announce something already begun. If a commitment's time is truly unstatable, omit it.
6. Were any calendar commitments listed above fulfilled, cancelled, or made moot by the latest round's events? Give their exact names.

Reply ONLY a JSON object:
{{"ledger_updates": [{{"id": "<exact ledger entry id>", "status": "achieved"|"failed"|"transformed"|"moot", "evidence": "<one line: the event that settled it>"}}],
  "material_change": true|false,
  "elapsed_hours_this_round": <number, e.g. 0.25>,
  "natural_gap_hours": <number, 0 if the action continues immediately>,
  "new_commitments": [{{"description": "<what, who, where>", "due_in_hours": <number>, "kind": "appointment"|"external_response"}}],
  "resolved_commitments": ["<exact name from the calendar>"],
  "reason": "<one sentence>"}}"""


class Terminator:
    def __init__(self, llm, central_tension: str, k_stagnant: int = 2,
                 time_budget_hours: float | None = None,
                 terminal_goals: dict | None = None,
                 hard_time_stop: bool = True):
        self.llm = llm
        self.tension = central_tension
        self.k = k_stagnant
        self.stagnant_streak = 0
        self.time_budget_hours = time_budget_hours
        self.hard_time_stop = hard_time_stop
        self.time_marked = False
        self.elapsed_hours = 0.0
        self.judge_failures = 0   # consecutive empty/unparseable judge replies
        self.verdicts: list = []
        # resolution ledger: the central tension + each cast member's terminal
        # motivation (archivist freeze validation) as persistent threads
        self.ledger: list = []
        if central_tension:
            self.ledger.append({"id": "central_tension", "kind": "tension",
                                "text": central_tension, "status": "open",
                                "evidence": ""})
        for name, goal in (terminal_goals or {}).items():
            if goal:
                self.ledger.append({"id": f"goal:{name}", "kind": "goal",
                                    "text": f"{name}'s terminal motivation: {goal}",
                                    "status": "open", "evidence": ""})

    def _ledger_text(self) -> str:
        rows = []
        for e in self.ledger:
            row = f"- [{e['id']}] status={e['status']}: {e['text']}"
            if e["status"] != "open" and e["evidence"]:
                row += f" (settled: {e['evidence']})"
            rows.append(row)
        return "\n".join(rows) or "(none)"

    @staticmethod
    def _norm_id(s) -> str:
        """The ledger renders entries as "- [id] status=..." and the judge
        echoes the id with brackets/spacing intact — exact matching dropped
        those updates (in an earlier batch one story ran to the round cap
        after its tension had resolved)."""
        return re.sub(r"[\[\]【】\s]", "", str(s or "")).lower()

    def _apply_ledger_updates(self, updates) -> None:
        by_id = {self._norm_id(e["id"]): e for e in self.ledger}
        for u in updates or []:
            if not isinstance(u, dict):
                continue
            e = by_id.get(self._norm_id(u.get("id")))
            st = u.get("status")
            # settled entries are sticky: only open threads can move, and only
            # to a terminal state — the ledger accumulates, never re-litigates
            if e is None or e["status"] != "open" or st not in TERMINAL_STATES:
                continue
            e["status"] = st
            e["evidence"] = str(u.get("evidence") or "")

    def check(self, latest_round_text: str, previous_round_text: str,
              pending_commitments: str = "") -> dict:
        self.llm._resim_kind = "terminator_check"
        raw = self.llm.chat(CHECK_PROMPT.format(
            ledger=self._ledger_text(),
            latest=latest_round_text or "(empty)",
            previous=previous_round_text or "(start of simulation)",
            pending=pending_commitments or "(none)")) or ""
        m = re.search(r"\{", raw)
        try:
            v = json.JSONDecoder().raw_decode(raw[m.start():])[0] if m else {}
        except json.JSONDecodeError:
            v = {}
        if not v:
            # judge unavailable (empty reply/unparseable): this is NOT
            # evidence of progress — it used to reset the stagnation streak,
            # so a dead judge (e.g. CLI weekly limit) silently disabled
            # deadlock termination and every run burned to the round cap
            self.judge_failures += 1
            if self.judge_failures >= 3:
                raise RuntimeError(
                    "terminator judge unavailable for 3 consecutive rounds "
                    "— refusing to continue without termination management")
            verdict = {"judge_error": True, "terminate": False,
                       "outcome": "ongoing",
                       "stagnant_streak": self.stagnant_streak,
                       "elapsed_hours_total": round(self.elapsed_hours, 2),
                       "time_budget_hours": self.time_budget_hours,
                       "ledger": [dict(e) for e in self.ledger]}
            self.verdicts.append(verdict)
            return verdict
        self.judge_failures = 0
        self._apply_ledger_updates(v.get("ledger_updates"))
        # commitment ledger extraction: validate before anyone trusts it
        cleaned = []
        for c in v.get("new_commitments") or []:
            if not isinstance(c, dict):
                continue
            desc = c.get("description")
            due = c.get("due_in_hours")
            if not desc or not isinstance(due, (int, float)) or due <= 0:
                continue
            kind = c.get("kind")
            if kind not in ("appointment", "external_response"):
                kind = "appointment"
            cleaned.append({"description": str(desc), "due_in_hours": float(due),
                            "kind": kind})
        v["new_commitments"] = cleaned
        v["resolved_commitments"] = [str(n) for n in
                                     (v.get("resolved_commitments") or [])
                                     if isinstance(n, str) and n]
        if v.get("material_change") is False:
            self.stagnant_streak += 1
        else:
            self.stagnant_streak = 0
        h = v.get("elapsed_hours_this_round")
        if isinstance(h, (int, float)) and h >= 0:
            self.elapsed_hours += float(h)
        g = v.get("natural_gap_hours")
        if isinstance(g, (int, float)) and g > 0:
            self.elapsed_hours += float(g)

        tension = next((e for e in self.ledger if e["kind"] == "tension"), None)
        goals = [e for e in self.ledger if e["kind"] == "goal"]
        terminate, outcome = False, "ongoing"
        if tension and tension["status"] != "open":
            terminate, outcome = True, f"resolved:{tension['status']}"
        elif goals and all(e["status"] != "open" for e in goals):
            # every driving motivation has reached a stable state: the story's
            # engines are spent even if the tension question was never judged
            terminate, outcome = True, "motivations_settled"
        elif self.time_budget_hours is not None \
                and self.elapsed_hours >= self.time_budget_hours \
                and not self.time_marked:
            # the in-story time coordinate of the canonical ending (hours
            # fallback when no event anchor exists): a hard stop only in
            # anchor_mode=stop; otherwise a one-shot measurement marker
            self.time_marked = True
            if self.hard_time_stop:
                terminate, outcome = True, "time_reached"
            else:
                outcome = "time_marker"
        elif self.stagnant_streak >= self.k:
            terminate, outcome = True, "deadlock"
        verdict = {**v, "terminate": terminate, "outcome": outcome,
                   "stagnant_streak": self.stagnant_streak,
                   "elapsed_hours_total": round(self.elapsed_hours, 2),
                   "time_budget_hours": self.time_budget_hours,
                   "ledger": [dict(e) for e in self.ledger]}
        self.verdicts.append(verdict)
        return verdict
