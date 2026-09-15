# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/Xiucheng-Zhang/ANIMASK.git
cd ANIMASK
pip install -r requirements-dev.txt
python -m pytest tests/
```

The pipeline and the engine use the standard library only; keep it that way
unless a dependency is unavoidable, and never make a test call a model
provider (the tests monkeypatch the LLM shims and point `ANIMASK_ROOT` at a
temporary tree).

## Pull requests

- Keep the change focused; describe what it changes and why.
- Add or extend a test under `tests/` for behaviour you change.
- Prompts are part of the measurement: a change to any prompt constant
  should say so in the description, since it changes what the judges see.
- Run the full suite before opening the PR.

## Reporting a problem

Open an issue with the command you ran, the relevant stage log
(`results/batch/<name>/logs/`), and the model names involved. Never paste
the contents of your `.env`.
