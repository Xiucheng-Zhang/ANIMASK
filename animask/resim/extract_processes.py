"""Extract the world's EXOGENOUS process table for a pack (v2 time design).

Exogeneity criterion: the process advances or occurs even if every simulated
character does nothing. These processes form the event queue that is the
spine of simulation time; endogenous (character-driven) events are never
timeline anchors.

Writes engine/data/worlds/<pack_id>/processes.json.

Usage (from the repository root): python3 -m animask.resim.extract_processes story01__oracle
  (story text is resolved from the base sid before "__")
"""
import json
import re
import sys
from pathlib import Path

from animask.llm import GPT, query  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = ROOT / "engine"

PROCESSES_PROMPT = (
    "The piped text is the beginning of a story, cut at a freeze point. "
    "Extract the world's EXOGENOUS PROCESSES: things that advance or occur on "
    "their own schedule even if every character does nothing — periodic "
    "cycles, scheduled deadlines/reviews, ongoing depletions or accumulations, "
    "institutional routines with fixed cadence.\n"
    "Only include processes grounded in the text. For each, explain WHY it "
    "proceeds without character action (the exogeneity check) — drop it if "
    "you cannot.\n"
    "Descriptions must be phrased as recurring/ongoing facts valid at ANY "
    "later time — no deictic wording ('current', 'now', 'today') and no "
    "reference to the freeze moment's time of day or season phase (the "
    "description will be re-emitted every time the process occurs, weeks "
    "later included).\n"
    "Reply ONLY a JSON array:\n"
    '[{"name": "<snake_case>", "description": "<one line>", '
    '"kind": "periodic|scheduled|ongoing", '
    '"period_hours": <number — REQUIRED for kind=periodic: infer the cycle '
    "length in hours from the text's stated cadence (e.g. 'three times every "
    "day' -> 8, 'weekly review' -> 168); null only for kind=ongoing>, "
    '"due_hours": <number — REQUIRED for kind=scheduled (hours from the '
    "freeze moment until it occurs/completes); for periodic: hours until the "
    "next occurrence (estimate; use period_hours/2 if unknown); null only "
    'for kind=ongoing>, '
    '"exogeneity": "<why it advances without any character\'s action>"}]'
)


def main() -> None:
    pack_id = sys.argv[1]
    base_sid = pack_id.split("__")[0]
    meta = json.loads((ROOT / "data" / "meta" / f"resim_{base_sid}.json").read_text())
    text = (ROOT / "data" / "clean" / f"{base_sid}.txt").read_text()
    pre_text = " ".join(text.split()[: meta["freeze_word"]])

    # fail loudly instead of writing an empty processes.json: the batch
    # runner gates this stage on file existence, so a backend error or an
    # unparseable reply persisted here would leave the timekeeper without
    # exogenous events forever ("<<ERROR: ...>>" contains no '[' and used to
    # decode to []; with a '[' it used to crash on raw JSONDecodeError)
    processes = None
    for attempt in range(3):
        out = query(PROCESSES_PROMPT, model=GPT, stdin=pre_text)
        if (out or "").startswith("<<ERROR"):
            raise SystemExit(f"backend error: {out[:300]}")
        m = re.search(r"\[", out or "")
        if m:
            try:
                processes = json.JSONDecoder().raw_decode(out[m.start():])[0]
                break
            except json.JSONDecodeError:
                pass
        print(f"  (unparseable process table, attempt {attempt + 1}/3; "
              f"reply head: {(out or '')[:200]!r})")
    if not isinstance(processes, list) or not processes:
        raise SystemExit("no usable process table extracted — refusing to "
                         "write an empty processes.json")

    wdir = ENGINE / "data" / "worlds" / pack_id
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "processes.json").write_text(
        json.dumps(processes, indent=2, ensure_ascii=False))
    for p in processes:
        print(f"  {p['name']:28s} {p['kind']:9s} period={p.get('period_hours')} "
              f"due={p.get('due_hours')} | {p['description'][:60]}")
    print(f"saved {len(processes)} processes -> {wdir / 'processes.json'}")


if __name__ == "__main__":
    main()
