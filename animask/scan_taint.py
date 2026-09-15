"""Channel-injection fingerprint scanner for simulation batches.

Scans a batch's engine actor calls (llm_calls_<tag>.jsonl, tag=role:*) and
persona-probe interview answers (persona_probe_<tag>.l2.json, ans|* keys) for
provider-side artifacts (identity leaks, refusals, system-prompt mentions).

Motivated by a case where a handful of interview answers in one actor batch
were tainted by a wrapper prompt leaking into the replies. Use as the
release gate for every new actor batch:

    python3 scan_taint.py mybatch_claudesonnet5_r1
    python3 scan_taint.py <newtag> --packs pack1 pack2   # first-2-books gate

Exit code 1 when any strong fingerprint hits, 0 when clean.
STRONG patterns = wrapper identity/refusal leaks (near-zero false positives).
WEAK patterns (reported, not gating) = 参考材料/搜索结果 style grounding echoes;
in-story actions like a character "搜索"-ing trip these legitimately.
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"

STRONG = re.compile(
    r"Claude|Anthropic|\bCLI\b|system prompt|系统提示"
    r"|cannot and will not continue|as an AI|AI language model|AI assistant"
    r"|我是一个AI|作为AI|作为一个人工智能|无法扮演|I cannot role|I can't role",
    re.I,
)
WEAK = re.compile(r"参考材料|参考资料|提供的资料|根据搜索|搜索结果|search results", re.I)


def iter_texts(tag, packs):
    for f in sorted(glob.glob(str(ROOT / "*" / f"llm_calls_{tag}.jsonl"))):
        pack = Path(f).parent.name
        if packs and pack not in packs:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("tag", "")
            if not t.startswith("role:"):
                continue
            yield pack, "engine/" + t.split(":", 1)[1], r.get("response") or ""
    for f in sorted(glob.glob(str(ROOT / "*" / f"persona_probe_{tag}.l2.json"))):
        pack = Path(f).parent.name
        if packs and pack not in packs:
            continue
        d = json.load(open(f))
        for k, v in d.get("items", {}).items():
            if not k.startswith("ans|") or not isinstance(v, str):
                continue
            yield pack, "interview/" + k.split("|")[1], v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--packs", nargs="*", default=None)
    ap.add_argument("--show", type=int, default=3, help="examples per role")
    args = ap.parse_args()

    n = 0
    strong_hits = defaultdict(list)
    weak_hits = defaultdict(int)
    for pack, who, text in iter_texts(args.tag, args.packs):
        n += 1
        key = f"{pack}/{who}"
        if STRONG.search(text):
            strong_hits[key].append(text[:100].replace("\n", " "))
        elif WEAK.search(text):
            weak_hits[key] += 1

    print(f"scanned {n} texts for tag {args.tag}")
    if strong_hits:
        total = sum(len(v) for v in strong_hits.values())
        print(f"STRONG: {total} hits across {len(strong_hits)} roles")
        for key, exs in sorted(strong_hits.items(), key=lambda x: -len(x[1])):
            print(f"  {key}: {len(exs)}")
            for e in exs[: args.show]:
                print(f"    | {e}")
    else:
        print("STRONG: clean")
    if weak_hits:
        print(f"WEAK (informational): {dict(weak_hits)}")
    sys.exit(1 if strong_hits else 0)


if __name__ == "__main__":
    main()
