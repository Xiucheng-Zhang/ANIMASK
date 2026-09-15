# Modified file; the original work is under Apache-2.0, see README (License).
import sys
from collections import defaultdict
sys.path.append("../")
import os
from typing import Any, Dict, List, Optional, Literal
from modules.embedding import get_embedding_model
from modules.memory import build_role_agent_memory
from modules.history_manager import HistoryManager
from utils import *
import random
import warnings
warnings.filterwarnings("ignore")



class RPAgent:
    def __init__(self, 
                 role_code: str,
                 role_file_dir: str,
                 world_file_path: str,
                 source: str = "",
                 language: str = "en",
                 db_type: str = "chroma",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None
                 ):
        super(RPAgent, self).__init__()
        self.language: str  = language
        self.role_code: str = role_code
        
        self.history_manager = HistoryManager()
        self.prompts: List[Dict] = []
        self.acted: bool = False
        self.status: str = ""
        self.goal: str = ""
        # experience layer: personality is frozen; what the
        # character lives through accumulates here as first-person notes
        self.episode_notes: List[str] = []
        self._episode_start_idx: int = 0
        self.location_code: str = ""
        self.location_name: str = ""
        self.motivation: str = ""
        
        self._init_from_file(role_code, role_file_dir, world_file_path, source)
        self._init_prompt()
        
        self.llm_name = llm_name
        if llm == None:
            llm = get_models(llm_name)
        self.llm = llm
        # metadata tag for the ANIMASK_CALL_LOG (call attribution only; no
        # behavioral effect — see modules/llm/ChatAPI.log_call)
        setattr(self.llm, "_resim_tag", f"role:{role_code}")
        
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=self.language)
        
        self.db_name = clean_collection_name(f"role_{role_code}_{embedding_name}")
        self.db = build_db(data = self.role_data,
                           db_name = self.db_name,
                           db_type = db_type,
                           embedding = embedding)
        self.world_db = None
        self.world_db_name = ""
        self.memory = build_role_agent_memory(llm_name=llm_name,
                                              embedding_name = embedding_name,
                                              embedding = embedding,
                                              db_name = self.db_name.replace("role","memory"),
                                              language = self.language,
                                              type="naive"
                                              )
        
    def _init_prompt(self):
        # language x style matrix: zh is a
        # translation of the en set, and ANIMASK_PROMPT_STYLE applies to both
        # languages alike. Fallback default is "neutral", matching run_resim's
        # CLI default — server/__main__ entry points used to silently run
        # "original" while the batch ran "neutral"
        neutral = os.getenv("ANIMASK_PROMPT_STYLE", "neutral") == "neutral"
        if self.language == 'zh':
            if neutral:
                from modules.prompt import role_agent_prompt_zh_neutral as _p
            else:
                from modules.prompt import role_agent_prompt_zh as _p
        elif neutral:
            from modules.prompt import role_agent_prompt_neutral as _p
        else:
            from modules.prompt import role_agent_prompt_en as _p
        self._ROLE_SET_GOAL_PROMPT = _p.ROLE_SET_GOAL_PROMPT
        self._ROLE_PLAN_PROMPT = _p.ROLE_PLAN_PROMPT
        self._ROLE_SINGLE_ROLE_RESPONSE_PROMPT = _p.ROLE_SINGLE_ROLE_RESPONSE_PROMPT
        self._ROLE_MULTI_ROLE_RESPONSE_PROMPT = _p.ROLE_MULTI_ROLE_RESPONSE_PROMPT
        self._INTERVENTION_PROMPT = _p.INTERVENTION_PROMPT
        self._UPDATE_GOAL_PROMPT = _p.UPDATE_GOAL_PROMPT
        self._UPDATE_STATUS_PROMPT = _p.UPDATE_STATUS_PROMPT
        self._ROLE_SET_MOTIVATION_PROMPT = _p.ROLE_SET_MOTIVATION_PROMPT
        self._SCRIPT_PROMPT = _p.SCRIPT_ATTENTION_PROMPT
        self._ROLE_MOVE_PROMPT = _p.ROLE_MOVE_PROMPT
        self._SUMMARIZE_PROMPT = _p.SUMMARIZE_PROMPT
        self._ROLE_NPC_RESPONSE_PROMPT = _p.ROLE_NPC_RESPONSE_PROMPT
        self._EPISODE_CONSOLIDATE_PROMPT = _p.EPISODE_CONSOLIDATE_PROMPT
            
    def _init_from_file(self, 
                        role_code: str, 
                        role_file_dir: str, 
                        world_file_path: str,
                        source:str):
        if source and os.path.exists(os.path.join(role_file_dir, source)):
            for path in get_child_folders(os.path.join(role_file_dir, source)):
                if role_code in path:
                    role_path = path
                    break
        else:
            for path in get_grandchild_folders(role_file_dir):
                if role_code in path:
                    role_path = path
                    break
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        role_profile_path = os.path.join(base_dir, role_path,"role_info.json")
        
        role_info = load_json_file(role_profile_path)
        # self.role_info = role_info
        self.role_profile: str = role_info['profile']
        self.nickname: str = role_info["nickname"]
        self.role_name: str = role_info["role_name"]
        self.relation: str = role_info["relation"]
        self.motivation: str = role_info["motivation"] if "motivation" in role_info else ""
        # two-tier goals. motivation = the fixed long-term
        # direction (story-level); goal = the current concrete step
        # (plot-level). The freeze-moment objective from extraction seeds the
        # current goal — previously x_agentic.goal_now was generated by the
        # pack builder but never read by the runtime.
        self.x_agentic: Dict[str, Any] = role_info.get("x_agentic") or {}
        if self.x_agentic.get("goal_now"):
            self.goal = self.x_agentic["goal_now"]

        self._apply_persona_level()

        self.activity: float = float(role_info["activity"]) if "activity" in role_info else 1.0
        self.icon_path: str = os.path.join(base_dir, role_path,"icon.png")
        self.avatar_path: str = os.path.join(base_dir, role_path,"avatar.png")
        for image_type in ['jpg','png','bmp']:
            if os.path.exists(os.path.join(base_dir, role_path,f"./avatar.{image_type}")):
                self.avatar_path: str = os.path.join(base_dir, role_path,f"avatar.{image_type}")
            if os.path.exists(os.path.join(base_dir, role_path,f"./icon.{image_type}")):
                self.icon_path: str = os.path.join(base_dir, role_path,f"icon.{image_type}")

        self.role_data: List[str] = build_role_agent_data(os.path.join(base_dir, role_path))

    def _apply_persona_level(self):
        """Persona-level switch (P0 arm): CARD MATERIAL is ablated at load
        time, runtime machinery is untouched — the agent still derives
        goals/motivation at run time, from placeholder inputs (= model
        default). Keeps: name, location, references/knowledge retrieval,
        runtime experiences. Removes: profile (-> placeholder), relation
        map, card motivation, goal seed. Single choke point: everything
        downstream (prompts, other-role views, VETO profile_head) reads
        these fields. Definition mirrors the single-step ablation in
        animask/persona_probe/l3_ablation — keep the two in sync."""
        self.persona_level = os.getenv("ANIMASK_PERSONA_LEVEL", "P3")
        if self.persona_level == "P0":
            self.role_profile = P0_PLACEHOLDER.get(self.language,
                                                   P0_PLACEHOLDER["en"])
            self.relation = {}
            self.motivation = ""
            self.goal = ""
            self.x_agentic = dict(self.x_agentic, goal_now="")

    # Agent
    def set_motivation(self, 
                       world_description: str, 
                       other_roles_info: Dict[str, Any], 
                       intervention: str = "", 
                       script: str = ""):
        if self.motivation:
            return self.motivation
        other_roles_info_text = self.get_other_roles_info_text(other_roles_info)
        prompt = self._ROLE_SET_MOTIVATION_PROMPT.format(**
            {
                "role_name": self.role_name,
                "profile":self.role_profile,
                "world_description": world_description,
                "other_roles_description": other_roles_info_text,
                "location": self.location_name
            })  
        if script:
            script = self._SCRIPT_PROMPT.format(**
                {"script": script}
            )  
            prompt = prompt + script
        elif intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
            prompt = intervention + prompt + "\n**注意: 在你的动机中考虑全局事件的影响**" if self.language == "zh" else intervention + prompt + "\n**Notice that: You should take the global event into consideration.**"
        
        self.llm._resim_kind = "role_set_motivation"
        motivation = self.llm.chat(prompt)
        self.save_prompt(prompt = prompt, detail = motivation)
        self.motivation = motivation
        return motivation

    def set_goal(self,
                 world_description: str,
                 other_roles_info: Dict[str, Any]):
        """Two-tier goals: derive the initial CURRENT goal from the fixed
        long-term motivation (the original ROLE_SET_GOAL_PROMPT, previously
        imported but never called). The freeze-moment goal_now from
        extraction takes precedence when present."""
        if self.goal:
            return self.goal
        other_roles_info_text = self.get_other_roles_info_text(other_roles_info)
        prompt = self._ROLE_SET_GOAL_PROMPT.format(**{
            "role_name": self.role_name,
            "profile": self.role_profile,
            "motivation": self.motivation,
            "world_description": world_description,
            "other_roles_description": other_roles_info_text,
            "location": self.location_name
        })
        self.llm._resim_kind = "role_set_goal"
        goal = strip_wrapping_quotes((self.llm.chat(prompt) or "").strip())
        self.save_prompt(prompt=prompt, detail=goal)
        self.goal = goal
        return goal

    def plan(self, 
             other_roles_info: Dict[str, Any], 
             available_locations: List[str], 
             world_description: str, 
             intervention: str = ""):
        action_history_text = self.retrieve_history(query = "", retrieve=False)
        references = self.retrieve_references(query = action_history_text)
        knowledges = self.retrieve_knowledges(query = action_history_text)
        
        if len(other_roles_info) == 1:
            other_roles_info_text = "没有人在这里。你不能进行涉及角色的互动。" if self.language == "zh" else "No one else is here. You can not interact with roles."
        else:
            other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile = False)
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_PLAN_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "profile": self.role_profile,
                "experiences": self.get_experiences_text(),
                "goal": self.goal,
                "status": self.status,
                "history": action_history_text,
                "other_roles_info": other_roles_info_text,
                "world_description": world_description,
                "location": self.location_name,
                "references": references,
                "knowledges":knowledges,
            }
        )
        prompt = intervention + prompt
        max_tries = 3
        plan = {"action": "待机" if self.language == "zh" else "Stay", 
                "destination": None,
                "interact_type":'no',
                "target_role_codes": [],
                "target_npc_name":None,
                "detail": f"{self.role_name}原地不动，观察情况。" if self.language == "zh" else f"{self.role_name} stays put."
                }
        
        for i in range(max_tries):
            self.llm._resim_kind = "role_plan"
            response = self.llm.chat(prompt)
            try:
                plan.update(json_parser(response))
                break
            except Exception as e:
                print(self.role_name)
                print(f"Parsing failure! {i+1}th tries. Error:", e)   
                print(response)
        plan["role_code"] = self.role_code
        self.save_prompt(detail=plan["detail"],
                      prompt=prompt)
        return plan
    
    def npc_interact(self,
                     npc_name:str,
                     npc_response:str,
                     history:str,
                     intervention:str = ""
                     ):
        references = self.retrieve_references(npc_response)
        knowledges = self.retrieve_knowledges(query = npc_response)
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_NPC_RESPONSE_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "profile": self.role_profile,
                "experiences": self.get_experiences_text(),
                "goal": self.goal,
                "npc_name":npc_name,
                "npc_response":npc_response,
                "references": references,
                "knowledges":knowledges,
                "dialogue_history": history
            }
            )
        prompt = intervention + prompt
        interaction = {
                    "if_end_interaction": True,
                    "detail": "",
                    }
        max_tries = 3
        for i in range(max_tries):
            self.llm._resim_kind = "role_npc"
            response = self.llm.chat(prompt)
            try:
                interaction.update(json_parser(response))
                break
            except Exception as e:
                print(f"Parsing failure! {i+1}th tries. Error:", e)
                print(response)
                # the model often answers this prompt in plain prose — on the
                # final try, treat the text itself as the in-character reply
                # instead of raising out of the simulation loop
                if i == max_tries - 1 and isinstance(response, str) and response.strip():
                    interaction["detail"] = response.strip()
                    interaction["if_end_interaction"] = False
        self.save_prompt(detail = interaction["detail"],
                      prompt = prompt)
        return interaction
    
    def single_role_interact(self, 
                             action_maker_code: str, 
                             action_maker_name: str,
                             action_detail: str, 
                             action_maker_profile: str, 
                             intervention: str = ""):
        references = self.retrieve_references(action_detail)
        history = self.retrieve_history(query = action_detail)
        knowledges = self.retrieve_knowledges(query = action_detail)
        
        relation = f"role_code:{action_maker_code}\n" + self.search_relation(action_maker_code)
        
        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_SINGLE_ROLE_RESPONSE_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "action_maker_name": action_maker_name,
                "action_detail": action_detail,
                "profile": self.role_profile,
                "experiences": self.get_experiences_text(),
                "action_maker_profile": action_maker_profile,
                "relation": relation,
                "goal": self.goal,
                "status": self.status,
                "references": references,
                "knowledges":knowledges,
                "history": history
            }
            )
        prompt = intervention + prompt
        
        max_tries = 3
        interaction = {
                    "if_end_interaction": True,
                    "extra_interact_type":"no",
                    "target_npc_name":"",
                    "detail": "",
                    }
        
        for i in range(max_tries):
            self.llm._resim_kind = "role_single"
            response = self.llm.chat(prompt) 
            try:
                interaction.update(json_parser(response))
                break
            except Exception as e:
                print(f"Parsing failure! {i}th tries. Error:", e)
                print(response)
        # tolerate the pre-rename engine token 'enviroment' (old saves /
        # model echoes) — the dispatch vocabulary is 'environment'
        if interaction.get("extra_interact_type") == "enviroment":
            interaction["extra_interact_type"] = "environment"

        self.save_prompt(detail = interaction["detail"],
                      prompt = prompt)
        return interaction
    
    def multi_role_interact(self, 
                            action_maker_code: str, 
                            action_maker_name: str, 
                            action_detail: str, 
                            action_maker_profile: str, 
                            other_roles_info: Dict[str, Any], 
                            intervention: str = ""):
        references = self.retrieve_references(query = action_detail)
        history = self.retrieve_history(query = action_detail)
        knowledges = self.retrieve_knowledges(query = action_detail)
        
        other_roles_info_text = self.get_other_roles_info_text(other_roles_info, if_profile = False)

        if intervention:
            intervention = self._INTERVENTION_PROMPT.format(**
                {"intervention": intervention}
            )
        prompt = self._ROLE_MULTI_ROLE_RESPONSE_PROMPT.format(**
            {
                "role_name": self.role_name,
                "nickname": self.nickname,
                "action_maker_name": action_maker_name,
                "action_detail": action_detail,
                "profile": self.role_profile,
                "experiences": self.get_experiences_text(),
                "action_maker_profile": action_maker_profile,
                "other_roles_info":other_roles_info_text,
                "goal":self.goal,
                "status": self.status,
                "references": references,
                "knowledges":knowledges,
                "history": history
            }
            )
        prompt = intervention + prompt
        max_tries = 3
        interaction = {
                    "if_end_interaction": True,
                    "extra_interact_type":"no",
                    "target_role_code":"",
                    "target_npc_name":"",
                    "visible_role_codes":[],
                    "detail": "",
                    }
        
        for i in range(max_tries):
            self.llm._resim_kind = "role_multi"
            response = self.llm.chat(prompt) 
            try:
                interaction.update(json_parser(response))
                break
            except Exception as e:
                print(f"Parsing failure! {i}th tries. Error:", e)
                print(response)
        # same legacy-token tolerance as above
        if interaction.get("extra_interact_type") == "enviroment":
            interaction["extra_interact_type"] = "environment"
        self.save_prompt(detail = interaction["detail"], prompt=prompt)
        return interaction
    
    def get_experiences_text(self, max_notes: int = 8) -> str:
        if not self.episode_notes:
            # localized: an English empty-state inside a zh prompt tends to
            # flip the first consolidation note (and everything it seeds)
            # into English
            return ("（尚无——故事刚刚开始）" if self.language == "zh"
                    else "(none yet — the story has just begun)")
        return "\n".join(f"- {n}" for n in self.episode_notes[-max_notes:])

    def consolidate_episode(self, period_desc: str = ""):
        """Experience layer: compress the just-ended episode into
        one first-person note. Personality (profile/relation/motivation) stays
        frozen; lived experience accumulates in episode_notes and reaches
        prompts through the {experiences} block."""
        new_events = self.history_manager.get_subsequent_history(
            self._episode_start_idx)
        self._episode_start_idx = len(self.history_manager)
        if not new_events:
            return ""
        zh = self.language == "zh"
        period_desc = period_desc or ("一次场景切换" if zh else "a scene break")
        prompt = self._EPISODE_CONSOLIDATE_PROMPT.format(
            role_name=self.role_name,
            period_desc=period_desc,
            events="\n".join(str(e) for e in new_events[-80:]),
            prior_notes="\n".join(f"- {n}" for n in self.episode_notes[-6:])
                        or ("（无）" if zh else "(none)"))
        self.llm._resim_kind = "role_consolidate"
        note = (self.llm.chat(prompt) or "").strip()
        if note:
            self.episode_notes.append(note)
            self.save_prompt(prompt=prompt, detail=note)
        return note

    def update_status(self,):
        prompt = self._UPDATE_STATUS_PROMPT.format(**{
            "role_name":self.role_name,
            "status":self.status,
            "history_text":self.retrieve_history(query=""),
            "activity":self.activity
        })
        max_tries = 3
        for i in range(max_tries):
            self.llm._resim_kind = "role_update_status"
            response = self.llm.chat(prompt) 
            try:
                status = json_parser(response)
                self.status = status["updated_status"]
                self.activity = float(status["activity"])
                break
            except Exception as e:
                print(f"Parsing failure! {i}th tries. Error:", e)    
                print(response)
        
        return
    
    def update_goal(self,other_roles_status: str,instruction: str = ""):
        motivation = self.motivation
        if instruction:
            motivation = instruction
        history = self.retrieve_history(self.motivation)
        if len(history) == 0:
            # keep the freeze-moment current goal if extraction provided one;
            # fall back to the long-term motivation only when nothing seeded it
            if not self.goal:
                self.goal = motivation
            return self.goal
        
        prompt = self._UPDATE_GOAL_PROMPT.format(**{
            "history":history,
            "motivation":motivation,
            "goal":self.goal,
            "other_roles_status":other_roles_status,
            "location":self.location_name
        })
        self.llm._resim_kind = "role_update_goal"
        response = self.llm.chat(prompt) 
        try:
            new_plan = json_parser(response)
            if new_plan["if_change_goal"]:
                goal = new_plan["updated_goal"]
                self.save_prompt(prompt,response)
                self.goal = goal
                return goal
        except Exception as e:
            print(self.role_name)
            print(f"Parsing failure! Error:", e)    
            print(response)
        return ""
    
    def move(self, 
             locations_info_text: str, 
             locations_info: Dict[str, Any]):
        history_text = self.retrieve_history(query="")
        prompt = self._ROLE_MOVE_PROMPT.format(**{
            "role_name":self.role_name,
            "profile": self.role_profile,
            "goal":self.goal,
            "status":self.status,
            "history":history_text,
            "location":self.location_name,
            "locations_info_text":locations_info_text
            
        })
        self.llm._resim_kind = "role_move"
        response= self.llm.chat(prompt)
        try:
            result = json_parser(response)
            # an unlisted destination is legal when the fiction
            # has established the place — the caller resolves or registers it
            destination_code = str(result.get("destination_code") or "").strip()
            if result["if_move"] and destination_code and destination_code != self.location_code:
                self.save_prompt(detail = result["detail"],
                              prompt = prompt)
                return True, result["detail"], destination_code
        except Exception as e:
            print(f"Parsing failure! Error:", e)    
            print(response)
        return False, "",self.location_code
    
    def record(self,
                record):
        self.history_manager.add_record(record)
        # experience layer: index every witnessed event as it happens, so
        # older events stay recallable by relevance (the store was previously
        # only rebuilt on restore and never written during play)
        detail = record.get("detail") if isinstance(record, dict) else None
        if detail:
            try:
                self.memory.add_record(str(detail))
            except Exception:
                pass
        
    def save_prompt(self,prompt,detail):
        if prompt:
            self.prompts.append({"prompt":prompt,
                                 "response":detail})
    # Other
    def action_check(self,):
        if self.acted == False:
            self.acted = True
            return True
        dice = random.uniform(0,1)
        if dice > self.activity:
            self.acted = False
            return False
        return True
    
    def retrieve_knowledges(self, query:str, top_k:int=1, max_words = 100):
        if self.world_db is None:
            return ""
        knowledges = "\n".join(self.world_db.search(query, top_k,self.world_db_name))
        knowledges = knowledges[:max_words]
        return knowledges
    
    def retrieve_references(self, query: str, top_k: int = 3):
        if self.db is None:
            return ""
        references = "\n".join(self.db.search(query, top_k,self.db_name))
        return references
    
    def retrieve_history(self, query: str, top_k: int = 5, retrieve: bool = True):
        """Recent window plus relevance recall over the agent's own witnessed
        log. With an empty query this degrades to the plain
        recent window, matching the old behavior."""
        if len(self.history_manager) == 0: return ""
        recent = self.history_manager.get_recent_history(top_k)
        lines = list(recent)
        if retrieve and query and self.memory.len > top_k:
            seen = set(recent) | {"_init_"}
            recalled = [r for r in self.memory.search(query, top_k)
                        if r not in seen][:2]
            if recalled:
                if self.language == "zh":
                    lines = (["[回忆起的早先事件]"] + recalled
                             + ["[近期事件]"] + lines)
                else:
                    lines = (["[Recalled earlier events]"] + recalled
                             + ["[Recent events]"] + lines)
        return "\n" + "\n".join(lines) + "\n"
        
    def get_other_roles_info_text(self, other_roles: List[str], if_relation: bool = True, if_profile: bool = True):
        roles_info_text = ""
        for i, role_code in enumerate(other_roles):
            if role_code == self.role_code :continue
            name = other_roles[role_code]["nickname"]
            profile = other_roles[role_code]["profile"]  if if_profile else ""
            relation = self.search_relation(role_code) if if_relation else ""
            roles_info_text += f"\n{i+1}. {name}\nrole_code:{role_code}\n{relation}\n{profile}\n\n"

        return roles_info_text
    
    def search_relation(self, other_role_code: str):
        if self.language == 'en':
            if other_role_code in self.relation:
                relation_text = ",".join(self.relation[other_role_code]["relation"])
                detail_text = self.relation[other_role_code]["detail"]
                return f"This is your {relation_text}. {detail_text}\n"
            else:
                return ""
        elif self.language == 'zh':
            if other_role_code in self.relation:
                relation_text = ",".join(self.relation[other_role_code]["relation"])
                detail_text = self.relation[other_role_code]["detail"]
                return f"这是你的{relation_text}. {detail_text}\n"
            else:
                return ""
    def set_location(self, location_code, location_name):
        self.location_code: Optional[str] = location_code
        self.location_name: Optional[str] = location_name
            
    def __getstate__(self):
        states = {key: value for key, value in self.__dict__.items() \
            if isinstance(value, (str, int, list, dict, bool, type(None))) \
                and key not in ['role_info','role_data','llm','embedding','db',"memory"]
                and "PROMPT" not in key}
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)
        self._init_prompt()

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./roles/{self.role_code}.json")
        save_json_file(filename, self.__getstate__() )

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./roles/{self.role_code}.json")
        states = load_json_file(filename)
        self.__setstate__(states)     
        self.memory.init_from_data(self.history_manager.get_complete_history())

def build_role_agent_data(role_dir: str):
    role_data: List[str] = []
    for path in get_child_paths(role_dir):
        if os.path.splitext(path)[-1] == ".txt":
            text = load_text_file(path)
            role_data += split_text_by_max_words(text)
        elif os.path.splitext(path)[-1] == ".jsonl":
            role_data += [line["text"] for line in load_jsonl_file(path)]
    return role_data      


if __name__ == "__main__":
    agent = RPAgent(role_code='Harry-en')
    agent.single_role_interact("Hi,Harry, Who is Ron?")


    