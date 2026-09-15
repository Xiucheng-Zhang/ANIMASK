"""Regenerate engine/data/locations/<pack>.json from the pack-build
cache, using the current location-code rule (build_world_pack.
build_location_table). No LLM calls: the LOCATIONS_PROMPT reply is read back
from data/cache/world_pack/<pack>.json.

Why: the old rule `re.sub(r"[^A-Za-z0-9]", "", name)` turned every CJK-only
location name into "" so all zh packs collapsed to a single location. Packs
built before the fix need their location table rewritten; role cards are
untouched (they store location *names*, not codes).

Usage:
  python3 -m animask.resim.rebuild_location_codes --check            # all packs, report only
  python3 -m animask.resim.rebuild_location_codes story01__oracle ...  # rewrite these packs
  python3 -m animask.resim.rebuild_location_codes --write-changed    # rewrite every pack whose
                                                             # table would change
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
from animask.resim.build_world_pack import build_location_table, jparse  # noqa: E402

CACHE = ROOT / "data" / "cache" / "world_pack"
LOCDIR = ROOT / "engine" / "data" / "locations"


def cached_location_lists(pack: str) -> list:
    """Every parseable LOCATIONS reply in the pack's build cache (usually 1;
    packs re-extracted with a different text scope can hold 2)."""
    path = CACHE / f"{pack}.json"
    if not path.exists():
        return []
    out = []
    for v in json.loads(path.read_text()).values():
        if not (isinstance(v, str) and '"location_name"' in v):
            continue
        parsed = jparse(v)
        if isinstance(parsed, list) and parsed \
                and all(isinstance(x, dict) for x in parsed):
            out.append(parsed)
    return out


def pick(lists: list, current: dict) -> list:
    """When several replies are cached, take the one whose names overlap the
    existing table most (the one the pack was actually built from)."""
    if len(lists) == 1:
        return lists[0]
    names_now = {e.get("location_name") for e in current.values()}
    return max(lists, key=lambda L: len(
        {x.get("location_name") for x in L} & names_now))


def plan(pack: str):
    loc_path = LOCDIR / f"{pack}.json"
    current = json.loads(loc_path.read_text()) if loc_path.exists() else {}
    if "locations" in current and isinstance(current["locations"], dict):
        current = current["locations"]
    lists = cached_location_lists(pack)
    if not lists:
        return None, current, "no cached LOCATIONS reply"
    new = build_location_table(pick(lists, current))
    return new, current, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="*")
    ap.add_argument("--check", action="store_true",
                    help="report only, write nothing")
    ap.add_argument("--write-changed", action="store_true",
                    help="rewrite every pack whose table would change")
    args = ap.parse_args()

    packs = args.packs or sorted(p.stem for p in CACHE.glob("*.json"))
    changed = []
    for pack in packs:
        new, current, note = plan(pack)
        if new is None:
            print(f"{pack:30s} SKIP ({note})")
            continue
        same = (new == current)
        names_kept = ({e['location_name'] for e in new.values()}
                      == {e.get('location_name') for e in current.values()})
        print(f"{pack:30s} {len(current):3d} -> {len(new):3d} locations "
              f"{'unchanged' if same else 'CHANGED'}"
              f"{'' if same or names_kept else '  (name set differs!)'}")
        if same:
            continue
        changed.append(pack)
        if args.check:
            continue
        if args.packs or args.write_changed:
            (LOCDIR / f"{pack}.json").write_text(
                json.dumps(new, indent=2, ensure_ascii=False))
            print(f"{'':30s} wrote {LOCDIR / (pack + '.json')}")
    print(f"\n{len(changed)} pack(s) differ from the current rule: "
          f"{', '.join(changed) or '-'}")


if __name__ == "__main__":
    main()
