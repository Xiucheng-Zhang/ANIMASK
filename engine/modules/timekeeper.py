"""Timekeeper: the exogenous event queue — the spine of simulation time
(v2 design: event-primary, clock-auxiliary).

Holds the world process table (+ canonical off-cast fixtures). Time advances
either by fine-mode round estimates or by skips that land exactly on the next
exogenous event. Narrative position is denominated in event occurrences;
hours are bookkeeping.

The skip settlement is a neutral operation under the same contract as the
adjudicator: it may only advance declared processes and report the landed
event — never invent occurrences. Every skip is logged for audit.
"""
import json
import re

from utils import norm_event_key

VETO_PROMPT = """You are {name}. {profile_head}

Your current goal: {goal}
Your current status: {status}
Situation: {situation}

The coming period ({span}, until: {event_desc}) looks routine — no one is engaged with you and nothing demands your immediate attention. Answer privately and honestly, in character:

Do you intend to take a significant action BEFORE then, or would you let this time pass (living your routine)?

Reply ONLY a JSON object:
{{"let_time_pass": true|false,
  "routine": "<if letting time pass: how you spend it, one line>",
  "intended_action": "<if not: what you intend to do imminently, one line>"}}"""

SETTLE_PROMPT = """You are a neutral bookkeeper for a simulation (NOT a storyteller). {span} of in-world time pass quietly. You must settle this period.

Hard rules: only advance the processes listed below, the characters' stated routines, and the entailed pressures defined next; if the section below names a concrete landed event it occurs, otherwise NOTHING lands and the period simply ends; NOTHING else happens — no new events, discoveries, arrivals, accidents, or decisions. Characters' unresolved intentions remain unresolved. Never state or imply that a story milestone, ending, or turning point has been reached.

Entailed pressures — the ordinary costs of living through this period, which are consequences of time rather than new events. For each character, advance what their status, means, and routine imply: wages earned or missed, money and provisions running down, rent or lodging coming due, an employer / host / landlord losing patience, fatigue accumulating, standing threats or obligations drawing nearer. State these concretely in per_character_status (amounts, dwindling reserves, sharpening moods), and surface in situation_now any pressure that now demands action. Pressures may only TIGHTEN a character's circumstances — never introduce new opportunities, new characters, windfalls, or relief that resolves any part of the story's open tensions.

One exception, for the landed event only: if it is an awaited external response or consequence (a reply, a delivery, a decision by an off-stage party), state concretely what arrives or happens — choose the most ordinary, procedurally plausible content given the situation, and report it as an arriving fact, not as a resolution of the story's central tension. If it is a scheduled appointment, the skip ends just as the appointment is about to begin — do not narrate the appointment itself.

Where characters spend the period: for a skip longer than a day, characters do not linger at the transient site of the last scene (a public building, someone else's room, a street); they return to wherever they ordinarily sleep and work, as implied by their status and routine, unless the situation states they are confined or have nowhere else to go. Place each character at a location code from the list below; if the situation has established a place that matches none of the known codes, coin a short new CamelCase code for it (e.g. ChurchGuestRoom) instead of forcing a wrong known code.

## World situation before the skip
{situation}

## Processes that advanced during the skip
{advanced}

## The event this skip lands on
{event_desc}

## Characters at the start of the skip (location; status; stated routine)
{characters}

## Known locations (use these codes)
{places}

## Canonical world facts (may be empty; these are established and may inform pressures and statuses)
{canon_notes}

## Established story facts (verbatim ledger — every one of these remains true through the skip unless the landed event itself changes it; situation_now must not contradict them)
{facts}

## Pending commitments on the calendar (none of these is fulfilled, expired, or forgotten by the mere passage of time; any not yet due survives the skip unchanged)
{pending}

In "narration" and "situation_now", refer to places ONLY by their plain display names — never write a location_code or CamelCase identifier into narrative text. "per_character_status" must be concrete and in-world (where they are, what they hold, what they did during the period); do not use clinical boilerplate such as "conscious and physically stable" or "physically unharmed" unless injury was genuinely in question.

Reply ONLY a JSON object:
{{"narration": "<2-4 sentences: time passing, process states, the landed event beginning — plain, factual>",
  "situation_now": "<updated one-paragraph situation at the landing moment>",
  "per_character_status": {{"<role_code>": "<updated objective status, one line>"}},
  "per_character_location": {{"<role_code>": "<location_code from Known locations, or a new short CamelCase code for a place the situation established>"}}}}"""


def span_text(hours: float) -> str:
    """Human-readable duration for prompts: '12 hours', '72 hours (~3 days)',
    '504 hours (~3 weeks)', '4380 hours (~6 months)'. Large skips read as
    calendar spans, not raw hour counts."""
    h = float(hours)
    base = f"{h:.0f} hours" if h >= 2 else f"{h:g} hour" + ("s" if h != 1 else "")
    if h < 48:
        return base
    if h < 24 * 14:
        return f"{base} (~{round(h / 24)} days)"
    if h < 24 * 60:
        return f"{base} (~{round(h / (24 * 7))} weeks)"
    if h < 24 * 365 * 1.5:
        return f"{base} (~{round(h / (24 * 30.4))} months)"
    years = h / (24 * 365)
    return f"{base} (~{years:.1f} years)" if years < 10 else f"{base} (~{round(years)} years)"


def _jparse(raw: str) -> dict:
    m = re.search(r"[\[{]", raw or "")
    if not m:
        return {}
    try:
        return json.JSONDecoder().raw_decode(raw[m.start():])[0]
    except json.JSONDecodeError:
        return {}


class Timekeeper:
    def __init__(self, processes: list, fixtures: list | None = None,
                 max_skip_hours: float = 24 * 21,
                 skip_cap_max_hours: float = 24 * 365,
                 target_skips: int = 4):
        # base cap for one skip (applies near the anchor / with no anchor);
        # skip_cap() stretches it when the anchor is far, so the anchor stays
        # reachable within ~target_skips skips instead of dozens of 3-week
        # hops — bounded by skip_cap_max_hours
        self.max_skip_hours = max_skip_hours
        self.skip_cap_max_hours = max(max_skip_hours, skip_cap_max_hours)
        self.target_skips = max(1, int(target_skips))
        self._anchor_d0 = None  # anchor distance when first measured
        self.now_hours = 0.0
        self.queue = []  # [{name, description, due_hours, period_hours|None}]
        self.occurrences: dict[str, int] = {}
        self.skip_log: list = []
        self._commit_seq = 0
        # every entry carries two identities: `name` is the DISPLAY name
        # (original, the only one that may reach prompts/records/logs) and
        # `key` is the unique internal id that occurrences are counted by —
        # a fixture sharing a process's name must not credit the process's
        # event-denominated anchor, and internal suffixes must never leak
        # into narrative text
        for p in processes or []:
            due = p.get("due_hours")
            if due is None and p.get("period_hours"):
                due = p["period_hours"]
            if due is not None:
                self.queue.append({"name": p["name"], "key": p["name"],
                                   "description": p.get("description", ""),
                                   "due_hours": float(due),
                                   "period_hours": p.get("period_hours"),
                                   "kind": "process"})
        for f in fixtures or []:
            if f.get("due_hours") is None:
                continue
            name, due = f["name"], float(f["due_hours"])
            # a fixture that lands exactly on a same-named process's schedule
            # is the same event declared twice (the archivist restating the
            # process table) — keeping it fires a second [WORLD EVENT] at
            # the same moment. Drop it.
            if any(e["kind"] == "process" and e["name"] == name
                   and self._on_schedule(due, e) for e in self.queue):
                print(f"[timekeeper] fixture '{name}' (due {due:g}h) "
                      f"restates the schedule of process '{name}' — dropped")
                continue
            self.queue.append({"name": name,
                               "key": self._unique_key(name, "_fixture"),
                               "description": f.get("description", ""),
                               "due_hours": due, "period_hours": None,
                               "kind": "fixture"})

    @staticmethod
    def _on_schedule(due: float, entry: dict, tol: float = 0.01) -> bool:
        """Does `due` coincide with one of the entry's firing times?"""
        if entry["period_hours"]:
            k = (due - entry["due_hours"]) / float(entry["period_hours"])
            return k > -tol and abs(k - round(k)) < tol
        return abs(due - entry["due_hours"]) < tol

    def _unique_key(self, name: str, suffix: str) -> str:
        key = name
        if any(e["key"] == key for e in self.queue):
            base, i = f"{name}{suffix}", 2
            key = base
            while any(e["key"] == key for e in self.queue):
                key, i = f"{base}{i}", i + 1
        return key

    # --- dynamic commitment ledger -------------------------
    # The fiction keeps creating time-bound facts (an appointment, a promised
    # reply, a deadline). They are registered here as ordinary one-shot queue
    # entries, so propose_skip clips skips to them and advance() fires them —
    # the same machinery that serves canonical fixtures.

    def add_event(self, description: str, due_in_hours,
                  kind: str = "appointment", name: str | None = None):
        """Register a commitment created during play. kind: "appointment"
        (the cast attends it) or "external_response" (something arrives from
        an off-stage party). Returns the entry name, or None if invalid."""
        try:
            due = float(due_in_hours)
        except (TypeError, ValueError):
            return None
        if due <= 0 or not description:
            return None
        if kind not in ("appointment", "external_response"):
            kind = "appointment"
        self._commit_seq += 1
        name = name or f"commitment_{self._commit_seq}"
        self.queue.append({"name": name,
                           "key": self._unique_key(name, "_c"),
                           "description": str(description),
                           "due_hours": due, "period_hours": None,
                           "kind": kind})
        return name

    def remove_event(self, name: str) -> bool:
        """Drop a one-shot commitment that was fulfilled or made moot.
        Only play-created commitments are removable — canonical fixtures
        (kind "fixture") are not the fiction's to cancel. The name comes
        from the judge as free text, so match on the normalized display
        name (exact matching silently bounced legitimate removals, leaving
        settled appointments in {pending} and firing them as phantom
        events at their due time)."""
        want = norm_event_key(name)
        if not want:
            return False
        for e in list(self.queue):
            if norm_event_key(e["name"]) == want and not e["period_hours"] \
                    and e.get("kind") in ("appointment", "external_response"):
                self.queue.remove(e)
                return True
        return False

    def pending_commitments(self) -> list:
        return sorted((e for e in self.queue
                       if not e["period_hours"]
                       and e.get("kind") in ("appointment", "external_response")),
                      key=lambda e: e["due_hours"])

    def pending_text(self) -> str:
        rows = [f"- {e['name']}: {e['description']} "
                f"(due in ~{e['due_hours']:.0f}h; {e['kind']})"
                for e in self.pending_commitments()]
        return "\n".join(rows) or "(none)"

    def next_event(self):
        if not self.queue:
            return None
        return min(self.queue, key=lambda e: e["due_hours"])

    def advance(self, hours: float) -> list:
        """Advance the clock; return events whose due time was crossed, each
        with a `count` of occurrences the advance crossed (a periodic event
        can occur many times inside one long skip — all must be credited,
        or occurrence-denominated anchors never arrive and the leftover
        negative due time re-fires the event on every later advance).
        Periodic events reschedule past the new now; one-shot events leave
        the queue."""
        self.now_hours += hours
        passed = []
        for e in list(self.queue):
            e["due_hours"] -= hours
            if e["due_hours"] > 0:
                continue
            if e["period_hours"]:
                n = 0
                while e["due_hours"] <= 0:
                    n += 1
                    e["due_hours"] += float(e["period_hours"])
                self.occurrences[e["key"]] = self.occurrences.get(e["key"], 0) + n
                passed.append(dict(e, count=n))
            else:
                self.occurrences[e["key"]] = self.occurrences.get(e["key"], 0) + 1
                passed.append(dict(e, count=1))
                self.queue.remove(e)
        return passed

    def anchor_target_hours(self, anchor: dict | None):
        """Hours remaining until the canonical anchor coordinate, if known."""
        if not anchor:
            return None
        if anchor.get("anchor_type") == "event" and anchor.get("anchor_process"):
            name, need = anchor["anchor_process"], anchor.get("anchor_count") or 1
            done = self.occurrences.get(name, 0)
            if done >= need:
                return 0.0
            for e in self.queue:
                if e["name"] == name and e["period_hours"]:
                    return (need - done - 1) * float(e["period_hours"]) + e["due_hours"]
            # anchor process has no clock in the queue (quantification failed
            # upstream) — fall back to the hours interval rather than losing
            # the anchor entirely
        iv = anchor.get("hours_interval") or []
        if len(iv) == 2 and isinstance(iv[0], (int, float)):
            return max(0.0, float(iv[0]) - self.now_hours)
        return None

    def skip_cap_for(self, remaining, initial=None) -> float:
        """Effective cap for one skip. Near the anchor (or without one) it
        is the base cap. Far from it, the stride is initial_distance /
        target_skips — fixed for the run, so the anchor is reached in about
        target_skips skips (the last one lands exactly: propose_skip clips
        to the anchor candidate) — never above skip_cap_max_hours."""
        if remaining is None or remaining <= self.max_skip_hours:
            return self.max_skip_hours
        d0 = initial if initial is not None else remaining
        return min(self.skip_cap_max_hours,
                   max(self.max_skip_hours, d0 / self.target_skips))

    def skip_cap(self, anchor: dict | None = None) -> float:
        remaining = self.anchor_target_hours(anchor)
        if remaining is not None and remaining > 0 and self._anchor_d0 is None:
            self._anchor_d0 = remaining
        return self.skip_cap_for(remaining, self._anchor_d0)

    def estimate_skips(self, anchor: dict | None = None, limit: int = 200) -> int:
        """Skips needed to reach the anchor under the dynamic cap (0 if
        reached or unknown) — the round-budget estimator's input."""
        remaining = self.anchor_target_hours(anchor)
        if remaining is None or remaining <= 0:
            return 0
        d0 = self._anchor_d0 if self._anchor_d0 is not None else remaining
        n = 0
        while remaining > 0 and n < limit:
            remaining -= self.skip_cap_for(remaining, d0)
            n += 1
        return n

    def propose_skip(self, anchor: dict | None = None):
        """Skip target = nearest of {one-shot events (fixtures/scheduled),
        anchor coordinate}, capped. Minor periodic processes do NOT gate the
        skip — their occurrences accumulate inside it and are settled in the
        narration. Falls back to the nearest periodic event; None if there is
        nothing in the future at all.

        Two invariants:
        - anchor_description is post-freeze canon and must NEVER reach a
          generative prompt — the anchor candidate carries a neutral,
          diegetic description instead (it is a termination coordinate, not
          an event that imports the canonical ending into this timeline).
        - if the cap truncates the skip, nothing lands at its end: the
          proposal becomes an interim waypoint, not the (unreached) event."""
        candidates = [(e["due_hours"], e) for e in self.queue
                      if not e["period_hours"]]
        tgt = self.anchor_target_hours(anchor)
        if tgt is not None and tgt > 0:
            candidates.append((tgt, {"name": "canon_anchor_point",
                                     "description": "an uneventful stretch of "
                                                    "time; the period simply "
                                                    "ends",
                                     "due_hours": tgt, "period_hours": None}))
        if not candidates:
            nxt = self.next_event()
            if nxt is None:
                return None
            candidates = [(nxt["due_hours"], nxt)]
        hours, event = min(candidates, key=lambda x: x[0])
        cap = self.skip_cap(anchor)
        if hours > cap:
            return {"hours": cap,
                    "event": {"name": "interim_waypoint",
                              "description": "an uneventful stretch of time; "
                                             "the period simply ends",
                              "due_hours": cap,
                              "period_hours": None},
                    "landed": False, "cap": cap}
        return {"hours": hours, "event": event, "landed": True, "cap": cap}

    def anchor_reached(self, anchor: dict) -> bool:
        if not anchor:
            return False
        if anchor.get("anchor_type") == "event" and anchor.get("anchor_process"):
            name = anchor["anchor_process"]
            has_clock = any(e["name"] == name for e in self.queue) \
                or self.occurrences.get(name, 0) > 0
            if has_clock:
                return self.occurrences.get(name, 0) >= (anchor.get("anchor_count") or 1)
            # no clock for the anchor process — fall through to hours
        iv = anchor.get("hours_interval") or []
        if len(iv) == 2 and isinstance(iv[0], (int, float)):
            return self.now_hours >= iv[0]
        return False

    def log_skip(self, record: dict) -> None:
        self.skip_log.append(record)
