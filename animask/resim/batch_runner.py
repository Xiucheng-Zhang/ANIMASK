"""Parallel batch orchestrator: story -> world pack -> simulation -> persona
probe, stage-gated and resumable.

Every stage is skipped when its output artifact already exists, so killing and
relaunching the batch never repeats finished work. Per-item stage logs and a
single state.json record what happened.

Spec file (JSON list), e.g. results/batch/<name>/spec.json:
  [{"sid": "story01", "language": "en", "runs": 2},
   {"sid": "story02", "language": "zh", "runs": 2},
   {"sid": "story02", "language": "zh", "runs": 1,        # condition arms:
    "actor_model": "claude-sonnet-5", "persona": "P0"}, ...]
Optional per-item condition fields: actor_model (defaults to the engine
default), persona ("P3"/"P0"), judge_model (defaults to DEFAULT_JUDGE).
Conditions are encoded into the run tag (<batch>_<cond>_r<k>), so arms of
the same pack never overwrite each other; freeze_validation stays pack-level
and shared across arms.

Stages per item:
  extract   scripts:  python3 -m animask.resim.extract_state <sid>
  pack      scripts:  python3 -m animask.resim.build_world_pack <sid>
                          --persona-scope full --language <lang>
  processes scripts:  python3 -m animask.resim.extract_processes <sid>__oracle
  run[k]    engine:   python3 run_resim.py <sid>__oracle --tag <batch>_r<k>
  probe[k]  scripts:  python3 -m animask.persona_probe.run_persona_probe <pack> <tag>
                          --layers <probe-layers>   (skip with --no-probe)

Usage: python3 -m animask.resim.batch_runner <batch_name> [--spec path]
         [--build-slots 3] [--sim-slots 6] [--runs-default 1]
         [--probe-layers l2] [--no-probe]
Stop gracefully: touch results/batch/<name>/STOP  (finishes in-flight stages)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE = ROOT / "engine"

# judge != actor: the terminator judge is a different model family from the
# actor by default; the spec's judge_model overrides it per item
DEFAULT_JUDGE = "deepseek-v4-pro"


def preflight(spec: list[dict], root: Path = ROOT) -> list[str]:
    """Problems that would fail the batch on its first stage: a story text
    that is not there, or a provider that has no key in .env for one of the
    models the batch will call. Empty list means ready."""
    from animask import llm  # animask/llm.py
    problems = []
    for sid in sorted({it["sid"] for it in spec}):
        p = root / "data" / "clean" / f"{sid}.txt"
        if not p.exists():
            problems.append(f"{sid}: story text not found at {p}")
    models = {llm.GPT, DEFAULT_JUDGE}
    for it in spec:
        models.update(m for m in (it.get("actor_model"), it.get("judge_model")) if m)
    for m in sorted(models):
        try:
            llm.resolve(m)
        except llm.LLMConfigError as e:
            problems.append(str(e))
    return problems


def cond_suffix(item: dict) -> str:
    """Condition part of tags/stage keys (actor model + persona level).
    Empty for the default condition, so plain <batch>_r<k> names stay stable; two
    conditions on the same pack get distinct *_<tag>.* files and never
    overwrite each other. Spec fields: actor_model, persona ("P3"/"P0"),
    judge_model."""
    parts = []
    if item.get("actor_model"):
        parts.append(re.sub(r"[^A-Za-z0-9]+", "", item["actor_model"]).lower())
    if item.get("persona") and item["persona"] != "P3":
        parts.append(item["persona"])
    return "_".join(parts)


class Batch:
    def __init__(self, name: str, spec: list[dict], build_slots: int,
                 sim_slots: int, probe: bool, probe_layers: str = "l2"):
        self.name = name
        self.dir = ROOT / "results" / "batch" / name
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.probe = probe
        self.probe_layers = probe_layers
        self.state_path = self.dir / "state.json"
        self.state = (json.loads(self.state_path.read_text())
                      if self.state_path.exists() else {})
        self.lock = threading.Lock()
        self.build_sem = threading.Semaphore(build_slots)
        self.sim_sem = threading.Semaphore(sim_slots)

    # ---------- state ----------
    def mark(self, sid: str, stage: str, status: str, note: str = ""):
        with self.lock:
            rec = self.state.setdefault(sid, {})
            rec[stage] = {"status": status,
                          "t": time.strftime("%m-%d %H:%M:%S")}
            if note:
                rec[stage]["note"] = note[:300]
            self.state_path.write_text(
                json.dumps(self.state, indent=1, ensure_ascii=False))

    def stopped(self) -> bool:
        return (self.dir / "STOP").exists()

    # ---------- condition-aware naming ----------
    def run_tag(self, item: dict, k: int) -> str:
        c = cond_suffix(item)
        return f"{self.name}_{c}_r{k}" if c else f"{self.name}_r{k}"

    def stage(self, item: dict, name: str, k: int) -> str:
        c = cond_suffix(item)
        return f"{name}_{c}_r{k}" if c else f"{name}_r{k}"

    # ---------- shell ----------
    def sh(self, sid: str, stage: str, cmd: list[str], cwd: Path,
           timeout: int) -> bool:
        log = self.dir / "logs" / f"{sid}_{stage}.log"
        self.mark(sid, stage, "running")
        env = dict(os.environ)
        # per-process worker pools multiply across concurrent stages; cap
        # them so one provider never sees a thundering herd of full-text calls
        env.setdefault("LLM_MAX_WORKERS", "6")
        # token accounting for the scripts-side stages (extract/pack/
        # processes/probe): one JSONL per batch, records tagged by
        # item+stage (animask/llm.py usage_log). The sim stage has its own
        # per-run llm_calls_<tag>.jsonl, which now carries `usage` too.
        env.setdefault("LLM_USAGE_LOG", str(self.dir / "usage.jsonl"))
        env["LLM_USAGE_TAG"] = f"{sid}:{stage}"
        try:
            with open(log, "a") as lf:
                lf.write(f"\n===== {time.strftime('%F %T')} {' '.join(cmd)}\n")
                lf.flush()
                r = subprocess.run(cmd, cwd=cwd, stdout=lf, env=env,
                                   stderr=subprocess.STDOUT, timeout=timeout)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
            self.mark(sid, stage, "timeout")
            return False
        self.mark(sid, stage, "done" if ok else "failed",
                  "" if ok else f"exit={r.returncode}, see logs/{sid}_{stage}.log")
        return ok

    # ---------- stages ----------
    def ensure_extract(self, item) -> bool:
        sid = item["sid"]
        if (ROOT / "data" / "meta" / f"resim_{sid}.json").exists():
            return True
        with self.build_sem:
            if self.stopped():
                return False
            return self.sh(sid, "extract",
                           [sys.executable, "-m", "animask.resim.extract_state", sid],
                           ROOT, 3600)

    def ensure_pack(self, item) -> bool:
        sid, pack = item["sid"], item["sid"] + "__oracle"
        if (ENGINE / "presets" / f"{pack}.json").exists():
            return True
        with self.build_sem:
            if self.stopped():
                return False
            return self.sh(sid, "pack",
                           [sys.executable, "-m", "animask.resim.build_world_pack",
                            sid, "--persona-scope", "full",
                            "--language", item.get("language", "en")],
                           ROOT, 7200)

    def ensure_processes(self, item) -> bool:
        sid, pack = item["sid"], item["sid"] + "__oracle"
        if (ENGINE / "data" / "worlds" / pack / "processes.json").exists():
            return True
        with self.build_sem:
            if self.stopped():
                return False
            return self.sh(sid, "processes",
                           [sys.executable, "-m", "animask.resim.extract_processes",
                            pack], ROOT, 3600)

    def ensure_run(self, item, k: int) -> bool:
        sid, pack = item["sid"], item["sid"] + "__oracle"
        tag = self.run_tag(item, k)
        out = ROOT / "results" / "runs" / pack
        if (out / f"transcript_{tag}.json").exists():
            # a transcript from a crashed run must not count as done —
            # set it aside (preserved as *.err*) and re-run
            meta_p = out / f"run_meta_{tag}.json"
            status = None
            if meta_p.exists():
                try:
                    status = json.loads(meta_p.read_text()).get("status")
                except json.JSONDecodeError:
                    pass
            if status != "error":
                return True
            stamp = time.strftime("%m%d%H%M")
            for f in out.glob(f"*_{tag}.*"):
                f.rename(f.with_name(f"err{stamp}_" + f.name))
            self.mark(sid, self.stage(item, "run", k), "retry_after_error")
        else:
            # a timed-out / killed attempt leaves partial files with NO
            # transcript (run_meta status="running", a half-written call
            # log) — isolate them too, or the new attempt mixes with them
            leftovers = [f for f in out.glob(f"*_{tag}.*")
                         if not f.name.startswith("err")]
            if leftovers:
                stamp = time.strftime("%m%d%H%M")
                for f in leftovers:
                    f.rename(f.with_name(f"err{stamp}_" + f.name))
                self.mark(sid, self.stage(item, "run", k), "retry_after_partial")
        with self.sim_sem:
            if self.stopped():
                return False
            cmd = [sys.executable, "run_resim.py", pack, "--tag", tag,
                   "--judge_model", item.get("judge_model", DEFAULT_JUDGE)]
            if item.get("actor_model"):
                cmd += ["--actor_model", item["actor_model"]]
            if item.get("persona"):
                cmd += ["--persona_level", item["persona"]]
            return self.sh(sid, self.stage(item, "run", k), cmd,
                           ENGINE, 6 * 3600)

    def ensure_probe(self, item, k: int) -> bool:
        if not self.probe:
            return True
        if not (ROOT / "animask" / "persona_probe" / "run_persona_probe.py").exists():
            self.mark(item["sid"], self.stage(item, "probe", k),
                      "skipped", "plugin missing")
            return True
        sid, pack = item["sid"], item["sid"] + "__oracle"
        tag = self.run_tag(item, k)
        out = ROOT / "results" / "runs" / pack
        probe_p = out / f"persona_probe_{tag}.json"
        if probe_p.exists():
            # completeness, not existence: a probe json written with failed
            # calls (complete=false) must be re-run — it resumes from the
            # layer stores. Files predating the flag count as complete.
            try:
                complete = json.loads(probe_p.read_text()).get("complete",
                                                               True)
            except json.JSONDecodeError:
                complete = False
            if complete:
                return True
        if not (out / f"transcript_{tag}.json").exists():
            return False
        # the paper's measurements use the interview layer (l2); l1 and l3
        # are optional extra layers of the probe (see run_persona_probe)
        extra = ["--layers", self.probe_layers]
        if "l3" in self.probe_layers:
            extra += ["--l3-checkpoints", "4"]
        if item.get("actor_model"):
            # L2/L3 re-generate AS the character: the probing actor must be
            # the model that played it in this run
            extra += ["--actor-model", item["actor_model"]]
        with self.build_sem:
            return self.sh(sid, self.stage(item, "probe", k),
                           [sys.executable, "-m",
                            "persona_probe.run_persona_probe", pack, tag]
                           + extra, ROOT, 2 * 3600)

    # ---------- item pipeline ----------
    def run_item(self, item):
        sid = item["sid"]
        # stagger cold starts: the first wave of items otherwise fires all
        # its full-text extraction calls at the provider simultaneously
        time.sleep(min(item.get("_index", 0) * 5, 90))
        try:
            for fn in (self.ensure_extract, self.ensure_pack,
                       self.ensure_processes):
                if self.stopped() or not fn(item):
                    return
            ok = True
            for k in range(1, item.get("runs", 1) + 1):
                if self.stopped() or not self.ensure_run(item, k):
                    return
                ok = self.ensure_probe(item, k) and ok
            # a failed probe must not be swallowed into ALL=done, or "what
            # still needs a re-run" cannot be read off the state
            self.mark(sid, "ALL", "done" if ok else "done_with_failures",
                      "" if ok else "a probe stage failed — "
                      "re-launch retries it (stages are artifact-gated)")
        except Exception as e:  # never let one item kill the pool
            self.mark(sid, "ALL", "error", f"{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_name")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--build-slots", type=int, default=3)
    ap.add_argument("--sim-slots", type=int, default=6)
    ap.add_argument("--runs-default", type=int, default=1)
    ap.add_argument("--probe-layers", default="l2",
                    help="persona-probe layers to run per finished run "
                         "(default l2, the interview layer the paper uses)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the probe stage; run it later with "
                         "persona_probe.run_persona_probe")
    args = ap.parse_args()

    bdir = ROOT / "results" / "batch" / args.batch_name
    spec_path = Path(args.spec) if args.spec else bdir / "spec.json"
    spec = json.loads(spec_path.read_text())
    for i, it in enumerate(spec):
        it.setdefault("runs", args.runs_default)
        it["_index"] = i

    problems = preflight(spec)
    if problems:
        print("batch not started; fix these first:")
        for pr in problems:
            print("  -", pr)
        sys.exit(2)
    b = Batch(args.batch_name, spec, args.build_slots, args.sim_slots,
              not args.no_probe, probe_layers=args.probe_layers)
    print(f"batch {args.batch_name}: {len(spec)} items, "
          f"build_slots={args.build_slots} sim_slots={args.sim_slots}")
    with ThreadPoolExecutor(max_workers=args.build_slots + args.sim_slots) as ex:
        list(ex.map(b.run_item, spec))
    done = sum(1 for s in b.state.values()
               if s.get("ALL", {}).get("status") == "done")
    print(f"batch finished: {done}/{len(spec)} items complete "
          f"({'stopped early' if b.stopped() else 'natural end'})")


if __name__ == "__main__":
    main()
