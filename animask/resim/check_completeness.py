"""Artifact-completeness audit for a batch — the deliverable is completeness,
not evaluation: every item should have its full chain of raw artifacts so any
metric can be recomputed later. No scores are aggregated or interpreted here.

    python3 -m animask.resim.check_completeness <batchtag> [--runs 1]

Per item it checks (existence + parseable + non-trivially-empty):
  extract     data/meta/resim_<sid>.json
  pack        engine/presets/<pack>.json
              + data/roles/<pack>/ (>=1 role dir)
  processes   engine/data/worlds/<pack>/processes.json
  run_rK      transcript/run_meta(status=done)/llm_calls(nonempty)/
              history/timeline/verdicts/archivist_log
  probe_rK    persona_probe merged + .l1/.l2/.l3 layer stores
              + atoms cache per simulated role

Output: results/batch/<name>/completeness.md (+ .json), missing items listed
explicitly so they can be re-queued.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = ROOT / "engine"


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def check_item(sid: str, k: int, batch: str, cond: str = "") -> dict:
    pack = sid + "__oracle"
    tag = f"{batch}_{cond}_r{k}" if cond else f"{batch}_r{k}"
    out = ROOT / "results" / "runs" / pack
    missing, notes = [], []

    if jload(ROOT / "data" / "meta" / f"resim_{sid}.json") is None:
        missing.append("extract")

    preset = jload(ENGINE / "presets" / f"{pack}.json")
    roles_dir = ENGINE / "data" / "roles" / pack
    role_dirs = [d for d in roles_dir.iterdir() if d.is_dir()] \
        if roles_dir.is_dir() else []
    if preset is None or not role_dirs:
        missing.append("pack")

    procs = jload(ENGINE / "data" / "worlds" / pack / "processes.json")
    if not procs:   # None, [] or {}: an empty table means no exogenous
        missing.append("processes")   # events — not a complete stage

    meta = jload(out / f"run_meta_{tag}.json")
    transcript = jload(out / f"transcript_{tag}.json")
    if meta is None or transcript is None or meta.get("status") != "done":
        missing.append(f"run_r{k}")
        if meta is not None and meta.get("status") != "done":
            notes.append(f"run status={meta.get('status')}")
    else:
        calls = out / f"llm_calls_{tag}.jsonl"
        if not calls.exists() or calls.stat().st_size == 0:
            notes.append("llm_calls missing/empty")
        for aux in ("history", "timeline", "verdicts", "archivist_log"):
            if not (out / f"{aux}_{tag}.json").exists():
                notes.append(f"{aux} missing")


    probe = jload(out / f"persona_probe_{tag}.json")
    if probe is None:
        missing.append(f"probe_r{k}")
    else:
        for layer in ("l1", "l2", "l3"):
            lp = out / f"persona_probe_{tag}.{layer}.json"
            if isinstance(probe.get(layer), dict) \
                    and "skipped" not in probe[layer] and jload(lp) is None:
                notes.append(f"probe layer store {layer} missing")
        roles = probe.get("roles") or []
        for r in roles:
            if jload(ROOT / "data" / "cache" / "persona_probe" / pack /
                     f"{r}.atoms.json") is None:
                notes.append(f"atoms missing for {r}")

    return {"sid": sid, "missing": missing, "notes": notes,
            "complete": not missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    bdir = ROOT / "results" / "batch" / args.name
    spec = json.loads((bdir / "spec.json").read_text())

    rows = []
    for it in spec:
        for k in range(1, args.runs + 1):
            from animask.resim.batch_runner import cond_suffix
            rows.append(check_item(it["sid"], k, args.name, cond_suffix(it)))

    n_ok = sum(1 for r in rows if r["complete"])
    lines = [f"# {args.name} 产物完整性核对",
             f"\n完整 {n_ok}/{len(rows)}\n"]
    for r in rows:
        if r["complete"] and not r["notes"]:
            continue
        flag = "OK " if r["complete"] else "MISS"
        lines.append(f"- [{flag}] {r['sid']}: "
                     + (f"缺 {','.join(r['missing'])} " if r['missing'] else "")
                     + ("; ".join(r["notes"]) if r["notes"] else ""))
    if len(lines) == 3:
        lines.append("(全部条目产物齐全,无备注)")
    (bdir / "completeness.md").write_text("\n".join(lines), encoding="utf-8")
    (bdir / "completeness.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
