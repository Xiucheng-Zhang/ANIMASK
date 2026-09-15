"""Rewrite every existing pack's role `pre_freeze.txt` as a raw, line-break-
preserving slice of the story — the form build_world_pack now emits —
without any LLM call.

Background: the file used to be written as " ".join(words[:freeze_word]),
i.e. one newline-free line. The engine chunks role data BY LINE
(utils.split_text_by_max_words), so the whole span became a single chunk
that top-k retrieval injected verbatim into every role prompt as the
"speaking habits" reference.

Safety: a pack is skipped (and reported) when its current file does not
cover the same token span as data/meta/resim_<sid>.json — that pack was
built against a different freeze and must not be silently re-based.

Usage (from the repository root): python3 -m animask.resim.refresh_pre_freeze [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

from animask.resim.build_world_pack import raw_prefix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = ROOT / "engine"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rewritten = unchanged = skipped = 0
    texts: dict = {}
    for f in sorted((ENGINE / "data" / "roles").glob("*/*/pre_freeze.txt")):
        pack = f.parent.parent.name
        sid = pack.split("__")[0]
        meta_p = ROOT / "data" / "meta" / f"resim_{sid}.json"
        text_p = ROOT / "data" / "clean" / f"{sid}.txt"
        if not (meta_p.exists() and text_p.exists()):
            print(f"  SKIP {pack}/{f.parent.name}: no meta/text for {sid}")
            skipped += 1
            continue
        if sid not in texts:
            texts[sid] = raw_prefix(
                text_p.read_text(),
                json.loads(meta_p.read_text())["freeze_word"])
        raw = texts[sid]
        old = f.read_text()
        if old.split() != raw.split():
            print(f"  SKIP {pack}/{f.parent.name}: existing span differs "
                  f"from meta freeze ({len(old.split())} vs "
                  f"{len(raw.split())} tokens)")
            skipped += 1
            continue
        if old == raw:
            unchanged += 1
            continue
        if not args.dry_run:
            f.write_text(raw)
        rewritten += 1
        print(f"  {'would rewrite' if args.dry_run else 'rewrote'} "
              f"{pack}/{f.parent.name}: {old.count(chr(10))} -> "
              f"{raw.count(chr(10))} lines")
    print(f"done: rewritten={rewritten} unchanged={unchanged} skipped={skipped}"
          f"{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
