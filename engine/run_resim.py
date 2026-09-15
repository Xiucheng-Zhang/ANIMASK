"""Re-simulation runner for the engine.

What it adds to the base engine:
  - --prompt_style neutral|original : de-literarized prompt set (ablation switch)
  - roles start at freeze-moment locations, with status/consciousness
    initialized from extraction (asleep roles cannot be cast)
  - archivist attached: freeze validation (cached) + on-demand canonical
    facts for the adjudicator, with leakage-audited log
  - terminator attached: tension-resolution + stagnation termination

Usage: python3 run_resim.py <story_id> [--rounds N] [--model M]
       [--prompt_style neutral|original] [--scene_mode 0|1] [--tag T]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
ROOT = ENGINE.parent
os.chdir(ENGINE)
sys.path.insert(0, str(ENGINE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("story_id")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--prompt_style", default="neutral",
                    choices=["neutral", "original"])
    ap.add_argument("--scene_mode", type=int, default=1)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--actor_model", default=None,
                    help="model for role agents (the experimental variable); "
                         "defaults to --model")
    ap.add_argument("--env_model", default=None,
                    help="model for world agent + archivist (the fixed "
                         "environment side); defaults to --model")
    ap.add_argument("--persona_level", default="P3", choices=["P3", "P0"],
                    help="P0 ablates card material at load time (profile -> "
                         "placeholder, no relation/motivation/goal seed); "
                         "runtime machinery is untouched")
    ap.add_argument("--actor_temperature", type=float, default=0.7,
                    help="sampling temperature for role agents (fixed "
                         "experimental setting; judge/archivist run at 0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="replicate label recorded in run_meta (not a "
                         "reproducibility guarantee); defaults to the rN "
                         "suffix of --tag")
    ap.add_argument("--time_align", type=int, default=1,
                    help="1: track the canonical freeze->ending time "
                         "coordinate (archivist-estimated); 0: off")
    ap.add_argument("--anchor_mode", default="snapshot",
                    choices=["snapshot", "stop"],
                    help="what reaching the canonical time coordinate does. "
                         "snapshot: record the time-aligned state and keep "
                         "playing until the plot settles (resolution ledger / "
                         "deadlock / round cap) — the clock paces the story "
                         "but does not end it; stop: legacy hard termination "
                         "at the coordinate")
    ap.add_argument("--judge_model", default="deepseek-v4-pro",
                    help="model for the referee (judge-side only); judge "
                         "verdicts gate skips and termination, so it defaults "
                         "to the same tier as the actors")
    ap.add_argument("--cascade_cap", type=int, default=4)
    ap.add_argument("--sub_rounds", type=int, default=2)
    ap.add_argument("--max_skip_hours", type=float, default=24 * 21,
                    help="base cap for one time skip (near the anchor)")
    ap.add_argument("--skip_cap_max_hours", type=float, default=24 * 365,
                    help="hard ceiling of the dynamic skip cap: far from the "
                         "anchor the cap stretches to remaining/target_skips "
                         "so the anchor is reached in ~target_skips skips")
    ap.add_argument("--target_skips", type=int, default=4)
    ap.add_argument("--max_rounds", type=int, default=40,
                    help="ceiling of the auto round budget")
    args = ap.parse_args()
    # story_id may be a pack id like "<sid>__oracle" (persona-scope variants);
    # story text and freeze meta always live under the base sid
    base_sid = args.story_id.split("__")[0]

    os.environ["ANIMASK_PROMPT_STYLE"] = args.prompt_style
    os.environ["ANIMASK_PERSONA_LEVEL"] = args.persona_level
    # model decoupling: actor is the experimental variable; world/archivist
    # are the fixed environment side; terminator has its own --judge_model
    actor_model = args.actor_model or args.model
    env_model = args.env_model or args.model

    from simulation import Simulation
    from utils import (load_json_file, get_models, as_bool,
                          pick_location_by_name, norm_event_key,
                          seed_from_tag)
    from modules.archivist import Archivist
    from modules.terminator import Terminator
    from modules.timekeeper import Timekeeper

    out_dir = ROOT / "results" / "runs" / args.story_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- run manifest: everything needed to interpret/resume this run later ---
    import time as _time
    seed = args.seed if args.seed is not None else seed_from_tag(args.tag)
    run_meta = {"args": vars(args).copy(), "base_sid": base_sid,
                "started": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_style_env": args.prompt_style,
                # replicate label; sampling is NOT reproducible from it
                "seed": seed,
                "temperatures": {"actor": args.actor_temperature,
                                 "world": 0.7, "archivist": 0.0,
                                 "terminator_judge": 0.0},
                "models": {"actor": actor_model, "world": env_model,
                           "archivist": env_model,
                           "terminator_judge": args.judge_model},
                "persona_level": args.persona_level,
                "status": "running"}
    meta_path = out_dir / f"run_meta_{args.tag}.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    # raw LLM call log for this run (post-hoc analysis without re-querying);
    # respect an explicit external setting, including "0" to disable
    if "ANIMASK_CALL_LOG" not in os.environ:
        call_log = out_dir / f"llm_calls_{args.tag}.jsonl"
        if call_log.exists():
            # the log is opened in append mode per call: a leftover from an
            # earlier attempt of this tag would get the new run appended to
            # it, corrupting position-addressed probe replay — rotate it
            call_log.rename(call_log.with_name(
                f"llm_calls_{args.tag}.prev{_time.strftime('%m%d%H%M%S')}.jsonl"))
        os.environ["ANIMASK_CALL_LOG"] = str(call_log)

    preset_path = f"./presets/{args.story_id}.json"
    server = Simulation(preset_path=preset_path, world_llm_name=env_model,
                    role_llm_name=actor_model, embedding_name="naive")
    server.cascade_cap = args.cascade_cap
    server.sub_rounds = args.sub_rounds
    # temperature policy: actors sample at the fixed experimental temperature;
    # judgment/extraction components run deterministic (0) — a judge sampling
    # at 0.7 adds noise to termination decisions
    for _agent in server.role_agents.values():
        _agent.llm.temperature = args.actor_temperature

    # --- archivist: full text (incl. post-freeze) in an isolated context ---
    full_text = (ROOT / "data" / "clean" / f"{base_sid}.txt").read_text()
    freeze_word = json.loads(
        (ROOT / "data" / "meta" / f"resim_{base_sid}.json").read_text())["freeze_word"]
    arch_llm = get_models(env_model)
    arch_llm.temperature = 0.0  # extraction/validation: deterministic
    setattr(arch_llm, "_resim_tag", "archivist")
    arch = Archivist(full_text, freeze_word, arch_llm,
                     str(out_dir / f"archivist_log_{args.tag}.json"))
    server.world_agent.archivist = arch

    cast = [server.role_agents[c].role_name for c in server.role_codes]
    validation_path = out_dir / "freeze_validation.json"
    validation = json.loads(validation_path.read_text()) \
        if validation_path.exists() else {}
    # existence alone is not enough: a cached file missing the ledger fields
    # (an archivist parse failure persisted forever) leaves the resolution
    # ledger empty — plot-driven termination can then never fire
    if not (validation.get("central_tension")
            and validation.get("terminal_goals")):
        print("[archivist] validating freeze point ...")
        fresh = arch.validate_freeze(cast)
        fresh["x_model"] = env_model  # provenance: who computed this
        if not (fresh.get("central_tension") and fresh.get("terminal_goals")):
            print("[archivist] ERROR: freeze validation returned no "
                  "central_tension/terminal_goals — refusing to run with an "
                  "empty resolution ledger")
            sys.exit(1)
        # keep cached extras (canon_anchor, fixture_audit) — they are paid
        # results and independent of the ledger fields
        validation = {**validation, **fresh}
        validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"[archivist] valid={validation.get('valid')} "
          f"tension={validation.get('central_tension', '')[:100]!r}")
    if not validation.get("valid"):
        print("[archivist] WARNING: freeze point judged invalid — continuing anyway")

    # --- v2 time design: exogenous process table + event-coordinate anchor ---
    proc_path = ENGINE / "data" / "worlds" / args.story_id / "processes.json"
    processes = json.loads(proc_path.read_text()) if proc_path.exists() else []
    if not processes:
        print("[timekeeper] WARNING: no processes.json — run "
              "resim.extract_processes first; skips will be unavailable")

    time_budget, anchor = None, None
    if args.time_align:
        # truthiness, not key presence: an archivist parse failure returns {}
        # and a cached {"canon_anchor": {}} would otherwise run the whole
        # simulation with no anchor, no time budget and no fixtures
        if not validation.get("canon_anchor"):
            print("[archivist] locating canonical ending on the exogenous timeline ...")
            fresh_anchor = arch.estimate_canon_anchor(processes, cast)
            if not fresh_anchor:
                print("[archivist] ERROR: canon anchor estimate returned "
                      "nothing — refusing to run time-aligned with an empty "
                      "anchor (retry later or pass --time_align 0)")
                sys.exit(1)
            fresh_anchor["x_model"] = env_model
            validation["canon_anchor"] = fresh_anchor
            validation_path.write_text(json.dumps(validation, indent=2,
                                                  ensure_ascii=False))
        anchor = validation["canon_anchor"]
        # normalize the anchor process name against the actual process table —
        # the archivist sometimes invents a variant name, which silently breaks
        # event-denominated termination (occurrences of a nonexistent process
        # never accumulate)
        ap = anchor.get("anchor_process")
        table_names = {p["name"] for p in processes}
        if ap and processes and ap not in table_names:
            def _toks(s):
                return set(s.lower().replace("-", "_").split("_"))
            best = max(processes, key=lambda p: len(_toks(p["name"]) & _toks(ap)))
            if len(_toks(best["name"]) & _toks(ap)) >= 2:
                print(f"[anchor] process '{ap}' not in table; matched to "
                      f"'{best['name']}'")
                anchor["anchor_process"] = best["name"]
            else:
                print(f"[anchor] WARNING: process '{ap}' not in table, no close "
                      f"match; falling back to hours_interval")
                anchor["anchor_process"] = None
        print(f"[anchor] type={anchor.get('anchor_type')} "
              f"process={anchor.get('anchor_process')}x{anchor.get('anchor_count')} "
              f"hours={anchor.get('hours_interval')} "
              f"({anchor.get('confidence')}: {anchor.get('anchor_description', '')[:90]})")
        if anchor.get("anchor_type") != "event":
            iv = anchor.get("hours_interval") or []
            # numeric guard: the archivist occasionally writes strings
            # ("~72") — a non-numeric budget would TypeError mid-run in the
            # terminator's >= comparison; 0 would fire time_marker on round 1
            time_budget = (iv[0] if iv and isinstance(iv[0], (int, float))
                           and iv[0] > 0 else None)  # hours fallback

    # exogeneity gate: a fixture is scheduled only if it happens regardless
    # of any cast member's action or decision (outsiders acting ON a cast
    # member still qualify); the archivist's plot beats do not. Verdicts are
    # cached in freeze_validation.json; a failed audit schedules nothing
    # rather than letting unaudited canon into the simulation.
    fixtures = (anchor or {}).get("fixtures") or []
    admitted_fixtures, dropped_fixtures = [], []
    if fixtures:
        # norm_event_key: the auditor echoes names with case/spacing variants
        # ("Winter School Break" for winter_school_break) — exact matching
        # re-fired the audit every run, never cached, and dropped everything
        names = {norm_event_key(f.get("name")) for f in fixtures}
        audit = validation.get("fixture_audit") or {}

        def _verdict_keys(a):
            return {norm_event_key(v.get("name")) for v in a.get("verdicts", [])}

        if names - _verdict_keys(audit):
            print("[fixtures] auditing exogeneity ...")
            audit = arch.audit_fixtures(fixtures, cast)
            audit["x_model"] = env_model
            if not (names - _verdict_keys(audit)):
                validation["fixture_audit"] = audit
                validation_path.write_text(json.dumps(validation, indent=2,
                                                      ensure_ascii=False))
        verdicts = {norm_event_key(v.get("name")): v
                    for v in audit.get("verdicts", [])}
        if not verdicts:
            print("[fixtures] WARNING: exogeneity audit failed — scheduling "
                  "no fixtures this run")
        for f in fixtures:
            v = verdicts.get(norm_event_key(f.get("name")))
            # as_bool: a judge replying "keep": "false" (string) must drop —
            # bare truthiness let every rejected fixture through
            if v and as_bool(v.get("keep")):
                admitted_fixtures.append(f)
            else:
                dropped_fixtures.append(f.get("name"))
                if verdicts:
                    print(f"[fixtures] dropped '{f.get('name')}': "
                          f"{(v or {}).get('reason', 'no audit verdict')}")
    server.timekeeper = Timekeeper(processes, admitted_fixtures,
                                   max_skip_hours=args.max_skip_hours,
                                   skip_cap_max_hours=args.skip_cap_max_hours,
                                   target_skips=args.target_skips)
    server.canon_anchor = anchor
    server.anchor_mode = args.anchor_mode
    # auto budget: rounds scale with anchor distance (skips needed x overhead);
    # the skip cap is dynamic (Timekeeper.skip_cap), so a far anchor costs
    # ~target_skips skips, not remaining/3-weeks of them
    remaining = server.timekeeper.anchor_target_hours(anchor)
    if remaining and remaining > 0:
        est_skips = server.timekeeper.estimate_skips(anchor)
        auto_rounds = min(args.max_rounds, max(args.rounds, 12 + 6 * est_skips))
        if auto_rounds != args.rounds:
            print(f"[budget] anchor ~{remaining:.0f}h away -> est {est_skips} "
                  f"skips (first cap {server.timekeeper.skip_cap(anchor):.0f}h)"
                  f" -> rounds {args.rounds} -> {auto_rounds}")
            args.rounds = auto_rounds
    # v3: termination is plot-driven — the central tension AND each cast
    # member's terminal motivation (both from freeze validation) form the
    # resolution ledger; the story ends when they settle, deadlock, or the
    # round cap — not when the clock runs out (see --anchor_mode)
    judge_llm = get_models(args.judge_model)
    judge_llm.temperature = 0.0  # judgment: deterministic
    setattr(judge_llm, "_resim_tag", "terminator")
    run_meta["temperatures"]["terminator_judge"] = judge_llm.temperature
    # judge preflight: one tiny call BEFORE any actor spend. A dead judge
    # used to be invisible: every round's empty verdict counted as progress,
    # the run burned actor tokens to the round cap with no termination
    # management and still read "done"
    setattr(judge_llm, "_resim_kind", "judge_preflight")
    if not (judge_llm.chat("Reply with exactly: OK") or "").strip():
        print(f"[terminator] ERROR: judge model {args.judge_model!r} "
              "returned nothing on preflight — refusing to start a run "
              "without termination management (check the key and quota)")
        sys.exit(1)
    # actor preflight: one tiny actor call BEFORE any spend, so a
    # misconfigured or unavailable actor provider fails at the gate with
    # zero actor tokens burned instead of mid-flight
    _pre = get_models(actor_model)
    setattr(_pre, "_resim_tag", "preflight")
    setattr(_pre, "_resim_kind", "actor_preflight")
    if not (_pre.chat("Reply with exactly: OK") or "").strip():
        print(f"[preflight] ERROR: actor model {actor_model!r} returned "
              "nothing — provider unavailable, refusing to start")
        sys.exit(1)
    server.terminator = Terminator(judge_llm,
                                   validation.get("central_tension", ""),
                                   time_budget_hours=time_budget,
                                   terminal_goals=validation.get("terminal_goals"),
                                   hard_time_stop=(args.anchor_mode == "stop"))

    # --- freeze-faithful placement + status/consciousness init ---
    def place_roles():
        loc_by_code = server.world_agent.locations_info
        default_code = server.world_agent.locations[0]
        for role_code, agent in server.role_agents.items():
            info = load_json_file(f"./data/roles/{args.story_id}/{role_code}/role_info.json")
            ag = info.get("x_agentic", {})
            wanted = (ag.get("location_now") or "").lower()
            # longest contained name wins (first containment used to land
            # "皇宫内侍局宿舍门口" on 皇宫 instead of 内侍局宿舍); otherwise
            # >=2 overlapping tokens, else the default — see utils
            chosen = pick_location_by_name(wanted, loc_by_code, default_code)
            agent.set_location(chosen, server.world_agent.find_location_name(chosen))
            agent.status = ag.get("state_now", "")
            agent.asleep = not ag.get("conscious", True)
            print(f"[place] {role_code} -> {chosen}"
                  f"{' (asleep)' if agent.asleep else ''}")
    server.init_role_locations = place_roles

    # --- run to completion ---
    transcript = []
    gen = server.simulate_generator(rounds=args.rounds, if_save=0,
                                    mode="free", scene_mode=args.scene_mode)
    try:
        for actor_type, code, text, _rid in gen:
            transcript.append({"actor_type": actor_type, "code": code, "text": text})
            print(f"[{code or actor_type}] {text[:160]}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        transcript.append({"actor_type": "error", "code": "",
                           "text": f"{type(e).__name__}: {e}", "traceback": tb})
        print("SIMULATION ERROR:", type(e).__name__, e)
        print(tb)

    (out_dir / f"transcript_{args.tag}.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False))
    md = [f"# {args.story_id} — resim run ({args.model}, {args.prompt_style}, "
          f"rounds={args.rounds}, scene_mode={args.scene_mode})",
          f"central tension: {validation.get('central_tension', '')}", ""]
    for t in transcript:
        md.append(f"**[{t['code'] or t['actor_type']}]** {t['text']}")
        md.append("")
    (out_dir / f"transcript_{args.tag}.md").write_text("\n".join(md))
    (out_dir / f"history_{args.tag}.json").write_text(json.dumps(
        server.history_manager.detailed_history, indent=2,
        ensure_ascii=False, default=str))
    (out_dir / f"verdicts_{args.tag}.json").write_text(json.dumps(
        server.terminator.verdicts, indent=2, ensure_ascii=False))
    (out_dir / f"timeline_{args.tag}.json").write_text(json.dumps({
        "now_hours": server.timekeeper.now_hours,
        "occurrences": server.timekeeper.occurrences,
        "skips": server.timekeeper.skip_log,
        "anchor": anchor,
        "anchor_mode": args.anchor_mode,
        "fixtures_admitted": [f.get("name") for f in admitted_fixtures],
        "fixtures_dropped": dropped_fixtures,
        "goal_updates": getattr(server, "goal_log", []),
        "anchor_snapshot": getattr(server, "anchor_snapshot", None),
        "resolution_ledger": server.terminator.ledger},
        indent=2, ensure_ascii=False))
    run_meta["finished"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["status"] = ("error" if transcript and
                          transcript[-1]["actor_type"] == "error" else "done")
    run_meta["n_messages"] = len(transcript)
    run_meta["rounds_effective"] = args.rounds
    # the authoritative reason is the engine's explicit marker line — the
    # last verdict reads "ongoing" for round-cap and stop-mode time_reached
    # exits (an earlier batch mislabeled such runs), because those paths
    # terminate without a judge verdict
    term = None
    for t in reversed(transcript):
        m_term = re.search(r"SIMULATION TERMINATED: (\S+)", t.get("text", ""))
        if m_term:
            term = m_term.group(1)
            break
    run_meta["termination"] = term or (server.terminator.verdicts[-1].get("outcome")
                                       if server.terminator.verdicts else None)
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print(f"\nsaved {len(transcript)} messages -> {out_dir}/transcript_{args.tag}.md")
    if run_meta["status"] == "error":
        # exit non-zero so callers (batch_runner.ensure_run) do not treat an
        # errored transcript as success and eval/probe it in the same pass
        sys.exit(1)


if __name__ == "__main__":
    main()
