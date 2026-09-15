"""ANIMASK: story re-simulation pipeline and evaluation chain.

Sub-packages: `resim` (freeze point, world package, event table, batch
runner), `persona_probe` (evaluation chain); `llm` is the shared model
client. Run entry points as modules from the repository root, e.g.
`python3 -m animask.resim.batch_runner`.
"""
