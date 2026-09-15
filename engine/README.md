# Simulation engine

The multi-agent engine that replays a story from a world package, driven
by `run_resim.py`, normally through `animask/resim/batch_runner.py`.

## Components

- **Retrieval**: `modules/db/NaiveDB.py`, a dependency-free lexical
  retriever (CJK bigrams for Chinese), replaces the vector store; no
  embedding model is loaded (`embedding_name="naive"`).
- **Archivist** (`modules/archivist.py`): holds the full story text in an
  isolated context, validates the freeze point, answers factual questions
  for the world agent, and estimates the canonical anchor.
- **Timekeeper** (`modules/timekeeper.py`): the exogenous process table
  and audited canonical fixtures drive time skips; agents can veto a skip.
- **Terminator** (`modules/terminator.py`): a judge model keeps a
  resolution ledger (central tension, terminal goals) and closes the replay
  on resolution, settled goals, standstill, or the round cap.
- **Prompt styles**: `--prompt_style neutral` (default) uses the
  `*_neutral.py` prompt modules; `original` keeps upstream wording. Both
  have English and Chinese variants.
- **Persona levels**: `--persona_level P3` (full character card) or `P0`
  (card material replaced by a placeholder; runtime machinery untouched).
- **Model decoupling**: `--actor_model` (the experimental variable),
  `--env_model` (world agent and archivist), `--judge_model` (terminator).
- **Provenance**: `run_meta_<tag>.json` records models, temperatures, seed
  and termination; `llm_calls_<tag>.jsonl` logs every model call with its
  round and attribution tag (`ANIMASK_CALL_LOG`).

## Running one simulation

From this directory, with a world pack already built for `<pack>`:

```bash
python3 run_resim.py <pack> --tag <batch>_r1 --actor_model gpt-5.5 \
    --judge_model deepseek-v4-pro
```

`python3 run_resim.py --help` lists the round budget, sub-round, time-skip
and anchor options. Outputs land in `results/runs/<pack>/`.
