"""Neutral (de-literarized) Chinese role prompts for ANIMASK.

Mirror of role_agent_prompt_neutral: derived from role_agent_prompt_zh by
surgically removing the same drama pumps —
  - "引人入胜 / 推进剧情"
  - "制造张力、解决问题或引入戏剧性转折"
  - "不要准备/确认；立刻行动并得出结论"
  - "富有文学性的叙述性" output framing
Everything else (format contracts, JSON fields) is kept identical so the
engine parses outputs unchanged. Selected when language == "zh" and
ANIMASK_PROMPT_STYLE=neutral.
"""
from modules.prompt.role_agent_prompt_zh import *  # noqa: F401,F403
import modules.prompt.role_agent_prompt_zh as _zh

_REPLACEMENTS = [
    ("对话应当引人入胜、推进剧情，并揭示角色的情感、意图或冲突。",
     "只以这个角色在此情此景下真实会有的方式回应。"),
    ("4. **言之有物：**确保你的回应有实质内容，制造张力、解决问题或引入戏剧性转折。",
     "4. **忠实高于戏剧：**严格从你的目标、知识和性格出发行动。谨慎、犹豫、拒绝、"
     "等待或缓和事态，只要是这个角色真实会做的，都是可以的。绝不为了戏剧而添加戏剧。"),
    ("5. **避免重复：**不要重复对话历史中已有的信息，避免模糊或泛泛的回应。"
     "不要“准备”、“征求意见”或“确认”；而是立刻行动并得出结论。",
     "5. **不说废话：**不要复述历史记录中已有的信息。"),
    ("一个富有文学性的叙述性语句，包含",
     "一个平实的事实陈述，包含"),
    ("一个富有文学性的叙述性描述，包含",
     "一个平实的事实描述，包含"),
]

_TARGETS = ["ROLE_PLAN_PROMPT", "ROLE_SINGLE_ROLE_RESPONSE_PROMPT",
            "ROLE_MULTI_ROLE_RESPONSE_PROMPT", "ROLE_NPC_RESPONSE_PROMPT"]

_unused = {old for old, _ in _REPLACEMENTS}
for _name in _TARGETS:
    _p = getattr(_zh, _name)
    for _old, _new in _REPLACEMENTS:
        if _old in _p:
            _unused.discard(_old)
            _p = _p.replace(_old, _new)
    globals()[_name] = _p

if _unused:
    print(f"[neutral prompts zh] WARNING: {len(_unused)} replacement anchors "
          f"not found (base zh prompt text may have changed): "
          f"{[u[:60] for u in _unused]}")
