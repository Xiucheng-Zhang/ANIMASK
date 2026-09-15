"""Strip the script's OWN header from screenplay clean texts: title line,
tag/episode-count meta lines, 简纲 (whole-series synopsis — contains the
ending), and 人物设定 (character bios — some state the ending outright).
With the header in place the pre-freeze window leaks the ending, and persona
extraction sees ready-made inner-life bios that the screenplay medium, by
the paper's argument, does not provide.

SCOPE: the simulation pipeline only — data/clean/<yq|yc sid>.txt and
data/meta/resim_<sid>.json. data/clean_zh (contamination-probe corpus) is
deliberately untouched. Novels have no such header and are never touched.

The freeze point is NOT re-chosen: stripping removes a word-prefix, so the
freeze index shifts by exactly the header's word count; the freeze_quote
(when present) independently verifies the new position. No LLM calls.

After --apply, built packs are stale until rebuilt (paid):
  python3 -m animask.resim.extract_state <sid>          # state re-extraction
  python3 -m animask.resim.build_world_pack <sid>   # persona/world extraction
  python3 -m animask.resim.extract_processes <sid>

Usage (from the repository root):
  python3 -m animask.resim.strip_script_headers            # check all yq/yc, no writes
  python3 -m animask.resim.strip_script_headers <sid> ...  # check specific sids
  python3 -m animask.resim.strip_script_headers --apply <sid ...|all>
"""
import json
import time
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN = ROOT / "data" / "clean"
META = ROOT / "data" / "meta"
WORLDS = ROOT / "engine" / "data" / "worlds"
FREEZE_CACHE = ROOT / "data" / "cache" / "extract_state"

TITLE_RE = re.compile(r"^《[^》]{1,50}》")
META_LINE_RE = re.compile(
    r"^(标签|题材|类型|风格|集数|篇幅|字数|时长|频道|出品|编剧|导演|原著|改编|作者)"
    r"\s*[：:]")
SECTION_RE = re.compile(
    r"^(?:[一二三四五六七八九十]{1,3}[、.．]|\([一二三四五六七八九十]{1,3}\)|\d{1,2}[、.．])?\s*"
    r"[【\[]?\s*(简纲|梗概|故事梗概|故事简介|故事大纲|内容简介|内容介绍|剧情简介"
    r"|剧情介绍|剧情脉络|分集梗概|分集大纲|简介|大纲"
    r"|人物设定|人物介绍|人物小传|角色设定|角色介绍|主要人物|人物简介)"
    r"\s*[】\]]?\s*[：:]?\s*$")
SECTION_INLINE_RE = re.compile(SECTION_RE.pattern.rstrip("$")[:-len(r"\s*[：:]?\s*")]
                               + r"\s*[】\]]?\s*[：:]")
BODY_MARK_RE = re.compile(r"^[【\[]?\s*(正剧|正文|剧本正文)\s*[】\]]?\s*$")
# episode headings are ambiguous: the body uses them ("第一集" then scenes),
# but so do per-episode outline sections ("第一集：曹阳到莞城投靠…") — they
# are resolved by what follows (see _episode_starts_body)
EPISODE_RE = re.compile(r"^第\s*[0-9一二三四五六七八九十百千]+\s*[集幕场]"
                        r"\s*[：:]?\s*(.*)$")
SCENE_RES = [
    # "1-1 夏星辰住处 内 夜" — but not the outline style "1-10 集脉络：…"/"1-10：…"
    re.compile(r"^\d+\s*[-–—]\s*\d+(?!\d)(?!\s*[：:集])"),
    re.compile(r"^\d+\s*[，,、.．]\s*场景"),              # "1，场景：渭县—县狱刑房"
    re.compile(r"^场景\s*[：:]"),
    re.compile(r"^场\s*\d+"),                            # "场 1-1 夜/内 …"
    re.compile(r"^\d+\s*[.．、]\s*(内|外)[景]?\b"),
]
# lines that only script BODY contains (not outline prose)
BODY_SIGNAL_RES = [
    re.compile(r"^[△Δ]"),                                # action lines
    re.compile(r"^(人物|时间|地点)\s*[：:]"),
]
MAX_HEADER_LINES = 300   # a "header" longer than this is a parse failure


def _clean(raw: str) -> str:
    """Normalize a line for pattern matching: strip whitespace, leading
    markdown #'s, and surrounding ** decorations."""
    s = raw.strip()
    s = re.sub(r"^#+\s*", "", s)
    return s.strip("*").strip()


def _is_scene(line: str) -> bool:
    return any(r.match(line) for r in SCENE_RES)


def _episode_starts_body(lines: list, i: int) -> bool:
    """An episode heading at lines[i] starts the BODY iff it has no long
    trailing summary on the same line and a scene/action/cast line follows
    within the next few nonblank lines; otherwise it is outline prose."""
    m = EPISODE_RE.match(_clean(lines[i]))
    if not m or len(m.group(1)) > 10:
        return False
    seen = 0
    for raw in lines[i + 1:]:
        al = _clean(raw)
        if not al:
            continue
        if EPISODE_RE.match(al):
            return False   # the scene lines beyond belong to a later heading
        if _is_scene(al) or any(r.match(al) for r in BODY_SIGNAL_RES):
            return True
        seen += 1
        if seen >= 4:
            break
    return False


def _is_section(line: str) -> bool:
    return bool(SECTION_RE.match(line) or SECTION_INLINE_RE.match(line))


def strip_script_header(text: str) -> dict:
    """Returns {body, dropped, flags, title_line_rewritten}. `dropped` is
    the list of removed lines (verbatim). If no header is recognized the
    body is the input text unchanged."""
    lines = text.split("\n")
    flags: set = set()
    i = 0
    in_header = False
    rewritten_first = None

    while i < len(lines):
        if i >= MAX_HEADER_LINES:
            return {"body": text, "dropped": [], "title_line_rewritten": None,
                    "flags": {"parse_failure_header_too_long"}}
        line = _clean(lines[i])
        if not line:
            i += 1
            continue
        if BODY_MARK_RE.match(line):
            flags.add("body_marker")
            i += 1
            break
        if EPISODE_RE.match(line):
            if not in_header or _episode_starts_body(lines, i):
                break
            flags.add("episode_outline")   # "第X集：<summary>" outline row
            i += 1
            continue
        if _is_scene(line):
            break
        if not in_header and i < 5 and TITLE_RE.match(line):
            rest = TITLE_RE.sub("", line).strip()
            _em = EPISODE_RE.match(rest) if rest else None
            if rest and (_is_scene(rest) or BODY_MARK_RE.match(rest)
                         or (_em and len(_em.group(1)) <= 10)):
                # "《佳偶天成》第一集": keep the marker, drop the title
                flags.add("title_prefix_removed")
                rewritten_first = rest
                break
            flags.add("title")     # title alone, or title + synopsis run-on
            in_header = True
            i += 1
            continue
        if META_LINE_RE.match(line):
            flags.add("meta_lines")
            in_header = True
            i += 1
            continue
        if _is_section(line):
            m = SECTION_RE.match(line) or SECTION_INLINE_RE.match(line)
            flags.add("section:" + m.group(1))
            in_header = True
            i += 1
            continue
        if not in_header and not flags and len(line) <= 20 \
                and not re.search(r"[，。！？；：,.!?]", line):
            # bare title without 《》 — only if a labeled header section
            # follows before any scene line
            for al in lines[i + 1: i + 1 + 40]:
                al = _clean(al)
                if not al:
                    continue
                if _is_scene(al) or EPISODE_RE.match(al):
                    break
                if _is_section(al) or META_LINE_RE.match(al) \
                        or BODY_MARK_RE.match(al):
                    flags.add("bare_title")
                    in_header = True
                    break
            if not in_header:
                break
            i += 1
            continue
        if in_header:
            # synopsis / bio prose between recognized header parts: header
            # continues until a scene line or body marker (bounded above)
            flags.add("unlabeled_prose")
            i += 1
            continue
        break                     # no header at all

    dropped = lines[:i]
    kept = lines[i:]
    if rewritten_first is not None:
        dropped = lines[:i + 1]
        kept = [rewritten_first] + lines[i + 1:]
    body = "\n".join(kept).strip("\n")
    if dropped and len(body) < 200:
        # outline-only document (some incomplete scripts are an outline plus
        # character bios with no script
        # body at all) — stripping would leave nothing; refuse
        return {"body": text, "dropped": [], "title_line_rewritten": None,
                "flags": {"no_body_found"}}
    return {"body": body, "dropped": dropped, "flags": flags,
            "title_line_rewritten": rewritten_first}


def relocate_freeze(meta: dict, old_text: str, new_text: str) -> dict:
    """Shift freeze_word by the removed prefix's word count; verify against
    the freeze_quote when one exists. Returns an update report."""
    old_words, new_words = old_text.split(), new_text.split()
    shift = len(old_words) - len(new_words)
    old_cut = meta["freeze_word"]
    new_cut = old_cut - shift
    rep = {"shift_words": shift, "old_cut": old_cut, "new_cut": new_cut,
           "old_total": len(old_words), "new_total": len(new_words)}
    if new_cut <= 0:
        rep["error"] = "freeze fell inside the stripped header"
        return rep
    q = (meta.get("freeze") or {}).get("freeze_quote") or ""
    if q:
        if old_text.find(q) < 0:
            rep["quote_absent_pre_existing"] = True   # never located; the
            # cut is fraction-based and the exact shift still applies
        else:
            pos = new_text.find(q)
            rep["quote_found"] = pos >= 0
            if pos >= 0 and "snapped_fraction" in (meta.get("freeze") or {}):
                quote_cut = len(new_text[:pos].split())
                rep["quote_cut"] = quote_cut
                # the original snap put the cut AT the quote; after a pure
                # prefix removal the two must still agree
                rep["quote_agrees"] = abs(quote_cut - new_cut) <= 2
    approx = (meta.get("freeze") or {}).get("approx_fraction")
    if approx is not None and len(new_words) > 0:
        new_frac = new_cut / len(new_words)
        rep["new_fraction"] = round(new_frac, 3)
        # extract_state re-runs will re-snap: warn when the new quote
        # position drifts beyond its 0.15 acceptance window
        rep["resnap_ok"] = abs(new_frac - float(approx)) <= 0.15
    return rep


def process(sid: str, apply: bool = False) -> dict:
    path = CLEAN / f"{sid}.txt"
    old_text = path.read_text()
    res = strip_script_header(old_text)
    body, dropped, flags = res["body"], res["dropped"], res["flags"]
    changed = bool(dropped)
    out = {"sid": sid, "changed": changed,
           "dropped_lines": len([l for l in dropped if l.strip()]),
           "flags": sorted(flags),
           "first_kept": (body.split("\n", 1)[0][:44] if body else ""),
           "pack_built": (WORLDS / f"{sid}__oracle").exists()}
    if not changed:
        return out

    # suffix property: stripping must only remove a word-prefix (modulo the
    # rewritten first token when a title prefix was cut from a mixed line)
    ow, nw = old_text.split(), body.split()
    start = len(ow) - len(nw)
    tail_from = 1 if res["title_line_rewritten"] is not None else 0
    out["prefix_only"] = (start >= 0
                          and ow[start + tail_from:] == nw[tail_from:])

    meta_path = META / f"resim_{sid}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        out["freeze"] = relocate_freeze(meta, old_text, body)

    if apply:
        if not out["prefix_only"]:
            out["applied"] = False
            out["error"] = "suffix check failed — not applying"
            return out
        fr = out.get("freeze", {})
        blockers = []
        if "error" in fr:
            blockers.append(fr["error"])
        # every failed verification blocks the write — a wrong freeze_word
        # here silently poisons the whole (paid) pack rebuild
        if fr.get("quote_found") is False:
            blockers.append("freeze quote no longer found in stripped text")
        if fr.get("quote_agrees") is False:
            blockers.append("freeze quote position disagrees with the "
                            "shifted cut")
        if fr.get("resnap_ok") is False:
            blockers.append("new freeze fraction outside extract_state's "
                            "re-snap window")
        if blockers:
            out["applied"] = False
            out["error"] = "; ".join(blockers) + " — not applying"
            return out
        path.write_text(body + "\n")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["freeze_word"] = fr["new_cut"]
            meta["total_words"] = fr["new_total"]
            if "snapped_fraction" in (meta.get("freeze") or {}):
                meta["freeze"]["snapped_fraction"] = round(
                    fr["new_cut"] / fr["new_total"], 3)
            meta["header_stripped"] = {
                "date": time.strftime("%Y-%m-%d"),
                "removed_words": fr["shift_words"],
                "old_freeze_word": fr["old_cut"],
                "old_total_words": fr["old_total"]}
            meta_path.write_text(json.dumps(meta, indent=2,
                                            ensure_ascii=False))
            # seed the extract_state freeze cache with the SHIFTED cut so the
            # rebuild's `extract_state <sid>` re-extracts a clean state (the
            # cached meta["state"] saw the header) without re-choosing the
            # freeze point — the pinned index bypasses fraction re-snapping,
            # which drifts for snap_rejected picks and loses a word to int()
            cache_dir = FREEZE_CACHE
            cache_dir.mkdir(parents=True, exist_ok=True)
            seed = dict(meta.get("freeze") or {})
            seed["freeze_word_pinned"] = fr["new_cut"]
            seed["seeded_by"] = "strip_script_headers"
            (cache_dir / f"{sid}.freeze.json").write_text(
                json.dumps(seed, indent=2, ensure_ascii=False))
        out["applied"] = True
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    if not args or args == ["all"]:
        sids = sorted(p.stem for p in CLEAN.glob("y[qc]*.txt"))
    else:
        sids = args
    n_changed = n_flagged = 0
    rebuilt = []   # applied sids with a built pack -> rebuild checklist
    for sid in sids:
        out = process(sid, apply=apply)
        if out.get("applied") and out["pack_built"]:
            rebuilt.append(sid)
        if not out["changed"]:
            if "parse_failure_header_too_long" in out["flags"]:
                print(f"{sid}  !! header-too-long — left untouched")
                n_flagged += 1
            elif "no_body_found" in out["flags"]:
                print(f"{sid}  !! outline-only file (no script body) — "
                      f"left untouched")
                n_flagged += 1
            continue
        n_changed += 1
        fr = out.get("freeze") or {}
        warn = []
        if not out["prefix_only"]:
            warn.append("SUFFIX-CHECK-FAILED")
        if "error" in fr:
            warn.append(fr["error"])
        if fr and not fr.get("quote_found", True):
            warn.append("quote-not-found")
        if fr and not fr.get("quote_agrees", True):
            warn.append("quote-disagrees")
        if fr and not fr.get("resnap_ok", True):
            warn.append("resnap-window-exceeded")
        if warn:
            n_flagged += 1
        mark = " [pack]" if out["pack_built"] else ""
        print(f"{sid}{mark:7s} -{out['dropped_lines']:3d} lines  "
              f"{','.join(out['flags']) or '-':46s} "
              f"cut {fr.get('old_cut', '-')}->{fr.get('new_cut', '-')} "
              f"{'APPLIED' if out.get('applied') else ''}"
              f"{' !! ' + ' '.join(warn) if warn else ''}")
        print(f"        first kept: {out['first_kept']!r}")
    mode = "applied" if apply else "check only — nothing written"
    print(f"\n{n_changed}/{len(sids)} files have a header ({mode}); "
          f"{n_flagged} flagged for review")
    if rebuilt:
        # stale artifacts of a rebuilt pack must go, or the batch runner's
        # existence gates silently reuse the pre-strip pack; the freeze cache
        # seeded above pins the shifted cut for the fresh extract_state
        print("\n== rebuild checklist (paid) — per sid, in order ==")
        for sid in rebuilt:
            pack = f"{sid}__oracle"
            print(f"\n# {sid}")
            print(f"rm -f  results/runs/{pack}/freeze_validation.json"
                  f"   # stale tension/goals/anchor from the un-stripped text")
            print(f"rm -rf engine/data/roles/{pack}"
                  f"       # cast may change -> orphan role dirs")
            print(f"rm -rf engine/data/worlds/{pack}")
            print(f"rm -f  engine/presets/"
                  f"{pack}.json")
            print(f"(cd scripts && python3 -m animask.resim.extract_state {sid} && "
                  f"python3 -m animask.resim.build_world_pack {sid} "
                  f"--persona-scope full --language zh && "
                  f"python3 -m animask.resim.extract_processes {pack})")


if __name__ == "__main__":
    main()
