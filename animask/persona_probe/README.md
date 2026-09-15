# persona_probe — evaluation chain for finished replays

`persona_probe` is the measurement side of ANIMASK: a plugin over finished
simulation runs that reads the artifacts a run left behind, calls LLM judges
and writes new files next to them; the engine and its outputs are never
modified. Persona presence: at checkpoints of each character's acting
sequence the live context is forked and the character is interviewed with a
fixed battery; answers are penalty-scored against the card, summarised per
facet (traits and values, relations, knowledge) and gated before a role-run
enters the decision-point pool. Knowledge audit: at the first checkpoint,
before anything could have propagated inside the story, a judge checks
whether the character already possesses a fact belonging to another
character; the verdict feeds the gate. Story curves: replay and canon
continuation are rated at five checkpoints on four seven-point scales and
compared per actor model. Decision points: each consequential choice gets
the action the card implies (judged blind), the action taken and the action
of the same model without the persona; the five-label v2 relabel
(`yield < hold < press < oppose < cross`) yields the paper's adherence,
efficacy and shift numbers, the eight-class v1 labels are kept for
comparison. L1 (utterance stream) and L3 (ablation arm) are optional
per-run layers. The overall recipe is in the root
[`README.md`](../../README.md).

## Inputs (read-only)

Per run, under `results/runs/<pack>/` (`<pack>` is `<sid>__oracle`):
`transcript_<tag>.json` (utterance timeline, read by L1 and the replay-side
curves); `llm_calls_<tag>.jsonl` (call log, the live contexts for
interviews, ablation replays and the decision-point scan; an acting call has
`tag` `role:<code>` and a JSON `response` with a `"detail"` field, records
carry `messages`, `response`, `temperature`, `round`; without it L2/L3
return `{"skipped": "call_log_missing"}` and decision points skip);
`run_meta_<tag>.json` (`status`, where `done` marks a finished run,
`models.actor`, persona level); `verdicts_<tag>.json` (terminator verdicts
per round, hints for the scan); `freeze_validation.json` (`central_tension`
for the story-curve judge). Per pack:
`engine/data/roles/<pack>/<role>/role_info.json` (the frozen card:
`role_name`, `profile`, `relation`, `motivation`; `profile` appears verbatim
in acting prompts and is the ablation target) and
`engine/presets/<pack>.json` (`language`,
`role_agent_codes`). The canon side reads `data/clean/<sid>.txt` and the
freeze point in `data/meta/resim_<sid>.json`. Paths resolve against the
repository root; `ANIMASK_ROOT` overrides it (the tests point it at a
temporary tree); `LLM_MAX_WORKERS` caps the thread pool (default 10).

## Commands

Run everything from the repository root. Per-run probe (the batch runner calls it
with `--layers l2` after every finished simulation); values shown are the
defaults:

```bash
python3 -m animask.persona_probe.run_persona_probe <pack> <tag> \
    [--layers l1,l2,l3] [--actor-model gpt-5.6-sol] \
    [--l1-judge gpt-5.6-luna] [--main-judge deepseek-v4-pro] \
    [--second-judge gpt-5.6-luna] [--lang auto|zh|en] \
    [--l2-checkpoints 5] [--l2-paraphrases 2] [--l3-checkpoints 2] \
    [--max-utterances-l1 0] [--seed 0] [--flush-every 50] \
    [--no-bfi] [--force-atomize] [--dry-run]
```

`--actor-model` must be the model that played the characters in that run;
`--main-judge` also atomizes the cards; `--lang auto` reads the preset's
`language`; `--l2-paraphrases` counts variants per question including the
original; `--dry-run` makes no LLM call and only prints the planned counts.

Batch evaluation over every run whose `run_meta` says `done` and whose tag
is `<batch>` or starts with `<batch>_`:

```bash
python3 -m animask.persona_probe.run_eval <batch> [--judge deepseek-v4-pro] \
    [--no-b1] [--no-curves] [--legacy]
```

Steps: per run, the eight-class decision-point pipeline (skipped when its
output has `complete: true`) and the replay-side story curves; per story,
the canon-side curves; then `facet_profile`, `leak_audit`, `facet_profile`
again (so the audit enters the gate), the v1 decision-point aggregate and
`curve_compare`; `--legacy` adds the superseded `c3_scoreboard` and
`c0_convergence`. Failures are reported and skipped, never fatal. The
five-label relabel and its aggregate are separate:

```bash
python3 -m animask.persona_probe.run_b1v2_all <tag> [<tag> ...] [--parallel 2]
python3 -m animask.persona_probe.b1v2_decisions --aggregate <batch>
```

`run_b1v2_all` runs `b1v2_decisions <pack> <tag>` for every pack that has a
`b1_decisions_<tag>` file, one subprocess per pack, skipping complete
outputs. Every module is also runnable on its own (`python3 -m animask.persona_probe.<m> -h`).

Resume. Every LLM-backed module keeps a `ResumableStore`, a JSON file
`{"items": {}, "meta": {}}` keyed by one string per call and written
atomically every N puts (50 for the probe layers, 10 elsewhere); present
keys are skipped on re-entry, while transport failures (`<<ERROR`) and
unparseable judge replies are not stored and so get retried. Calls are
tallied in `CALL_STATS` (`ok`, `error`, `parse_skip`); a `CREDIT_EXHAUSTED`
error latches and short-circuits the rest. Final files carry
`complete: true` only when a pass had no error and no parse skip; otherwise
the probe runner exits with code 2 and the batch runner re-runs it. Batch
drivers are artifact-gated: a finished sub-result is never recomputed.

## Outputs and schema

Per run, `results/runs/<pack>/persona_probe_<tag>.json`:

```
{ pack, tag, lang, roles, generated_at, params, complete,
  call_stats: {by_phase, ok, error, parse_skip},
  atoms: {role: {n_atoms, by_type, cache_path}},
  l1: {                                  # or {"skipped": reason}
    per_agent: {role: {n_utterances,
      series: [{n, t_index, score,       # score = 1 - contradicted/applicable
                n_entail, n_contradict, n_applicable,
                features: {n_chars, n_tokens, ttr, punct_density, quote_ratio}}],
      stats: {n, mean, theil_sen_slope, mk: {s, z, p, n}, auc, total_change},
      feature_means}},
    calibration: {n_pairs, percent_agreement, cohens_kappa, judge, calib_judge},
    n_batches, failures, judge_model, lang},
  l2: {                                  # or {"skipped": "call_log_missing"}
    per_agent: {role: {
      checkpoints: [{call_index, frac, is_t0,
                     normative_score,    # penalty rubric vs the card, 0..100
                     self_score,         # agreement with the t0 answers
                     n_scored, violations_total}],
      stats: {normative: {...series stats...}, self: {...}},
      bfi: [{call_index, frac, raw, dims: {O, C, E, A, N}}],
      probe_set: [{qid, expectation, atom_id, source_role, variants}]}},
    skipped_agents, judge_agreement: {n_pairs, binary_agreement,
      cohens_kappa, score_mad, main_judge, second_judge}, lang, actor_model},
  l3: {                                  # or {"skipped": ...}
    per_agent: {role: {
      checkpoints: [{call_index, frac, next_call_index, note,
                     persona_votes, votes, order_consistent,  # 2AFC, swap x2
                     ablated_probe_score, persona_probe_score, l2_call_index,
                     drift_gap, cohens_d, len_persona, len_default}],
      persona_rate, persona_votes, votes, mean_drift_gap}},
    skipped_agents, ablation_notes, l2_available_for_gap, lang,
    actor_model, main_judge},
  stats_summary: {l1_mean_score_ci, l2_normative_ci,     # clustered bootstrap
                  l3_arm_gap_permutation: {role: {mean_diff, p, n, p_holm}}},
  verdict: {run_verdict, per_agent: {role: {verdict, evidence}},
            verdict_counts, inert_agents, degradations, thresholds,
            rules_version} }
```

Per-agent verdicts: `persona_inert` (2AFC persona-arm rate <= 0.60 on >= 8
votes and mean drift gap <= 10), else `drift_associated` (an L1 or L2 series
with n >= 3, Mann-Kendall p < 0.05 and Theil-Sen total change <= -0.05 on
L1 or <= -5 on L2), else `persona_stable_model_trend` (a trajectory plus
rate >= 0.75 on >= 6 votes), else `persona_stable_attribution_pending`, else
`insufficient_evidence`. The run verdict is `drift_associated` when at least
half of the evaluable agents are, `persona_inert` or
`persona_stable_model_trend` when all are, `mixed` otherwise.

The layer stores `persona_probe_<tag>.{l1,l2,l3}.json` hold the raw items
under keys such as `ans|<role>|<ci>|<qid>|<vi>` and `score|...` (L2) or
`act|<role>|<ci>` and `afc|<role>|<ci>|<order>` (L3); card atoms are cached
in `data/cache/persona_probe/<pack>/<role>.atoms.json`. Also per run:
`story_curves_<tag>.json` (`curves: {mood, plot_intensity,
tension_progress, relationship_warmth}`, five values each) and
`b1_decisions_<tag>.json` / `b1v2_decisions_<tag>.json` (`points` with
`implied`, `implied_set`, `actual`, `nocard`, `cell`, `cell_strict`; v1 adds
`summary`, v2 adds `*_toward`), each with a `.store.json`. Canon curves are
cached in `data/cache/canon_curves/<sid>.json`.

Batch-level files in `results/persona_probe/`:

- `facets_<batch>.json` + `.md` — `runs` (per run and role: `facets` with
  a `series` of `{call_index, frac, slot, score, n_scored}` and `stats`,
  `fk_violations`, `leak`, `gate: {pass, components, untested}`) and
  `groups` keyed `<actor_model>|<persona_level>` (`facet_means` with
  clustered bootstrap, cluster = pack; `drift_curve` per nominal slot;
  `gate_pass_rate`, `gate_failed_roles`, `leak_t0`). Incomplete probes are
  skipped unless `--include-incomplete`.
- `leak_audit_<batch>.json` (+ `.store.json`) — `roles` keyed
  `<pack>:<tag>:<role>` with `valid`, `reason`, `statement`, `verdicts:
  [{possesses, legitimately_known, evidence}]` and `leaked_t0` (`null` when
  the sample is invalid, pending, or legitimately known).
- `curves_<batch>.json` + `.md` — per actor model: `dispersion` and
  `dispersion_gap` (canon minus replay, bootstrap `lo`/`hi`) per checkpoint,
  `deviation` per score (`mean_diff`, `ci`, `n_pos`, `n_neg`,
  `p_permutation`, `p_holm` across the four scores).
- `b1_<batch>.json` + `.md` — eight-class aggregate per actor: `summary`,
  `adherence_ci`, `efficacy_ci`, `implied_x_actual`, `by_phase`.
- `b1v2_<batch>.json` + `.md` — five-label aggregate per actor:
  `adherence`, `disagreement_share`, `efficacy`, `overridden_share`,
  `persona_wins_at_disagreements`, `shift` (label index of the actual minus
  the persona-free action), each a clustered bootstrap interval (cluster =
  story). Both aggregates apply the presence gate when `facets_<batch>.json`
  exists: gate-failed role-runs are excluded, unknown ones quarantined.
- `aggregate_<batch>.json` + `.md` — cross-run pooling of probe verdicts;
  `c3_<batch>.*`, `c0_<batch>.*` only with `--legacy`.

## Design notes and known limitations

- **Interviews (L2).** Checkpoints sit at evenly spaced quantiles of each
  character's acting-call sequence; t0 is the first acting call. A fork
  replays the checkpoint call's full `messages` plus its actual response,
  appends one out-of-scene question and asks the actor model to answer in
  character at temperature 0; the finished run is untouched. The battery
  (identity, relation, should-know and should-not-know knowledge, and
  conditional questions, with paraphrase variants) is fixed per character
  and reused verbatim at every checkpoint. Scoring is a penalty rubric
  against the atomized card, `score = max(0, 100 - 5 * sum(severity))`; a
  seeded 25% subsample is re-scored by the second judge. The mini-BFI is
  auxiliary only.
- **Facets and gate.** Knowledge expectations and `K*` atoms map to
  knowledge, `R*` atoms to relations, all else to traits and values. A
  role-run passes when the traits-and-values and relations facets score
  >= 80 at >= 80% of their scored checkpoints and no t0 leak was audited
  (constants in `facet_profile.py`); a component without questions in a
  pack passes vacuously (`untested`).
- **Knowledge leak.** A raw violation on a should-not-know answer is only a
  diagnostic (`fk_violations`): it conflates foreign atoms about the probed
  character, in-story propagation, self-disclosure and trait slips. Samples
  whose source atom or question names the probed character are ruled
  invalid; valid ones are judged at t0 only for possession of the specific
  fact, with a `legitimately_known` escape that voids the sample.
- **Utterance stream (L1).** LLM-NLI judging (eight utterances per call)
  replaces a trained NLI model; a seeded 10% subsample is re-judged by the
  main judge, agreement and Cohen's kappa are reported, and low kappa calls
  for manual review.
- **Ablation arm (L3) and persona-free replays.** Card material is removed
  the way the engine's P0 level removes it (`profile` becomes a neutral
  placeholder, relation lines and motivation are deleted; `_persona_blocks`
  / `_ablate`, shared with the decision-point modules); a prompt without the
  verbatim profile is skipped with `profile_not_found_in_context`. Replays
  use the original call's temperature (0.7 when unrecorded). The L3 persona
  arm is the logged response, not a resample; its probe score comes from the
  nearest L2 checkpoint, so `drift_gap` is `null` without L2. Checkpoint
  indices are capped at the second-to-last acting call so a real next
  behavior exists for the 2AFC, which runs twice with positions swapped;
  length is excluded by instruction and recorded, not matched.
- **Decision points.** The scan takes the union of two votes per round, at
  most three candidates per round, one point per (actor, round). The implied
  label is judged from the card and the persona-ablated situation, never the
  action; actual and persona-free actions are classified with the same
  prompt, so the classifier cannot tell which arm it grades. Adherence uses
  set membership (implied label plus admitted alternatives); the strict
  single-label cell is a sensitivity variant. The v2 relabel reuses the v1
  scan and replay entries, re-runs only the three classifier calls with an
  ordered checklist (cross, oppose, yield, press, hold) and at most one
  adjacent acceptable label, and never modifies v1 files.
- **Story curves and statistics.** Both sides pass through the same
  questionnaire, so a scorer bias cancels; no dispersion ratio is reported
  because a strong genre convention collapses the canon-side spread, so the
  gap is given with a bootstrap interval. Statistics are pure Python
  (`stats.py`): Theil-Sen with Mann-Kendall, trapezoid trajectory AUC,
  clustered bootstrap, paired sign-flip permutation with Holm, kappa, d.
- **Judges and backend.** The main judge (DeepSeek family) is distinct from
  every actor arm; the cheap-tier judge (L1 primary, L2 second rater) is
  GPT-family. Every judge prompt restricts drift to traits: in-story changes
  of emotion, plans, relationships or knowledge are not violations.
  Single-prompt calls go through `llm.query` / `llm.query_many`; context
  forks need a messages array and go through `llm_chat` / `llm_chat_many`,
  which call `llm.chat` with the same credentials and routing; tests
  monkeypatch these four shims. Runs without `llm_calls_<tag>.jsonl`
  support L1 only and record `call_log_missing`.

## Module map

| Module | Role |
|---|---|
| `__init__.py` | Model constants, paths and `ANIMASK_ROOT`, call-log helpers, `ResumableStore`, `CALL_STATS`, the LLM shims |
| `run_persona_probe.py` | Per-run CLI: atomize, L1/L2/L3, stats summary, verdict, `--dry-run` |
| `atomize.py` | Character card to typed atoms, cached per pack and role |
| `l1_stream.py` | L1 utterance-level LLM-NLI stream with calibration subsample |
| `l2_checkpoint.py` | L2 interviews on the forked context, penalty scoring, self-baseline, second rating, mini-BFI |
| `l3_ablation.py` | L3 persona ablation, 2AFC and probe gap; ablation helpers shared with the decision-point modules |
| `facet_profile.py` | Presence profile per facet, gate constants, batch rollup |
| `leak_audit.py` | t0 knowledge-possession audit feeding the gate |
| `story_curves.py` | Five-checkpoint questionnaire on replay and canon |
| `curve_compare.py` | Dispersion and paired deviations across stories, Holm-adjusted |
| `b1_decisions.py` | Eight-class decision-point scan, labels, cells, aggregate |
| `b1v2_decisions.py` | Five-label relabel of the same points and its `--aggregate` |
| `run_b1v2_all.py` | Parallel, resumable relabel over every pack of the given tags |
| `run_eval.py` | One-command batch driver, `--legacy` for the endpoint chain |
| `aggregate.py` | Cross-run pooling of the per-run probe verdicts |
| `c3_scoreboard.py` | Legacy outcome scoreboard; `story_curves` borrows its canon segmentation |
| `c0_convergence.py` | Legacy convergence ratio over outcome cards |
| `stats.py` | Pure-Python statistics |

## Tests

From the repository root, `python3 -m pytest tests/`; no network or data
needed (the LLM shims are monkeypatched, `ANIMASK_ROOT` points at a
temporary tree). Files covering this package:

- `tests/test_persona_probe.py` — atomize, L1/L2/L3 pipelines, resume,
  language routing, verdicts, dry-run, statistics, cross-run aggregate.
- `tests/test_facet_profile.py` — facet profile, gate, `fk_validity`, leak audit.
- `tests/test_b1_decisions.py` — eight-class decision-point pipeline.
- `tests/test_eval_entry_points.py` — `run_eval`, `curve_compare` Holm
  p-values, five-label aggregate.
- `tests/test_condition_arms.py` — ablation helpers match the engine's P0 level.
- `tests/test_run_guards.py`, `tests/test_core_regressions.py` — parse-skip
  accounting, `<<ERROR` tally, credit latch.
- `tests/test_c3_scoreboard.py`, `tests/test_c0_convergence.py` — legacy modules.
