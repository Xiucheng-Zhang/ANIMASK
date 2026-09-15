"""LLM adapters for the simulation engine.

`ChatAPI` is the one adapter; it wraps the project-wide client in
animask/llm.py, which picks the provider from the model name. `BaseLLM`
defines the message-building interface the engine calls.
"""
