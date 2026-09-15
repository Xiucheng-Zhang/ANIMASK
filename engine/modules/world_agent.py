import ast
import sys
sys.path.append("../")
import csv
import re
from typing import Any, Dict, List, Optional, Literal
from utils import *
from modules.embedding import get_embedding_model

class WorldAgent:
    # Init
    def __init__(self, 
                 world_file_path: str,
                 location_file_path: str,
                 map_file_path: Optional[str] = "",
                 world_description: str = "",
                 llm_name: str = "gpt-4o-mini",
                 llm = None,
                 embedding_name: str = "bge-small",
                 embedding = None,
                 db_type: str = "chroma",
                 language: str = "zh",
                 ):
        if llm is None:
            llm = get_models(llm_name)
        if embedding is None:
            embedding = get_embedding_model(embedding_name, language=language)
        self.llm = llm
        # metadata tag for the ANIMASK_CALL_LOG (call attribution only)
        setattr(self.llm, "_resim_tag", "world")
        self.world_info: Dict[str, Any] = load_json_file(world_file_path)
        self.world_name: str = self.world_info["world_name"]
        self.language: str = language
        self.description:str = self.world_info["description"] if world_description == "" else world_description
        source = self.world_info["source"]
        
        self.locations_info: Dict[str, Any] = {}
        self.locations: List[str] = []
        self.history: List[str] = []
        self.edges: Dict[tuple, int] = {}  # 地点间距离
        self.prompts: List[Dict] = []
        # standing world facts (verbatim ledger; diff-updated by update_event)
        self.established_facts: List[str] = []
        
        self.init_from_file(map_file_path = map_file_path,
                            location_file_path = location_file_path)
        self.init_prompt()

        # canonical NPC cards (non-cast characters) extracted by
        # the pack builder — ground world-played NPCs in identity and facts
        npc_path = os.path.join(os.path.dirname(world_file_path), "npc_cards.json")
        self.npc_cards: Dict[str, Any] = \
            load_json_file(npc_path) if os.path.exists(npc_path) else {}

        # identity registry (pack builder) — all appellations of
        # each cast entity, with knowledge scope. The world model is the
        # administrator: it always knows which names denote the same person,
        # and controls whether an NPC may voice that knowledge.
        id_path = os.path.join(os.path.dirname(world_file_path), "identity_map.json")
        id_map = load_json_file(id_path) if os.path.exists(id_path) else {}
        self.cast_aliases: Dict[str, list] = id_map.get("cast_aliases", {})
        notes = []
        for ent in id_map.get("registry") or []:
            if not isinstance(ent, dict) or not ent.get("cast_name"):
                continue
            apps = [a for a in ent.get("appellations") or [] if isinstance(a, dict)]
            if len(apps) < 2:
                continue
            alias_txt = "; ".join(
                f"'{a.get('name')}' (known to: {a.get('known_by', 'all')})"
                for a in apps)
            notes.append(f"- {alias_txt} — ALL denote the same person, the "
                         f"cast character {ent['cast_name']}.")
        self.identity_notes: str = (
            "CAST IDENTITY NOTES (administrator knowledge): the following "
            "appellations denote the SAME person — never treat them as "
            "different people, never route a character to seek themselves, "
            "and never let an NPC reveal an appellation to characters the "
            "scope says do not know it:\n" + "\n".join(notes)) if notes else ""

            
        self.world_data,self.world_settings = build_world_agent_data(world_file_path = world_file_path, max_words = 50)
        self.db_name = clean_collection_name(f"settings_{source}_{embedding_name}")
        self.db = build_db(data = [row for row in self.world_data], 
                           db_name = self.db_name, 
                           db_type = db_type, 
                           embedding = embedding)
        
    def init_from_file(self, map_file_path: str, location_file_path: str, default_distance: int = 1):
        if map_file_path and os.path.exists(map_file_path):
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            with open(map_file_path, mode='r',encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                locations = next(csv_reader)[1:]  
                for row in csv_reader:
                    loc1 = row[0]
                    if loc1 not in valid_locations:
                        print(f"Warning: The location {loc1} does not exist")
                        continue
                    self.locations_info[loc1] = valid_locations[loc1]
                    self.locations.append(loc1)
                    distances = row[1:]
                    for i, distance in enumerate(distances):
                        loc2 = locations[i]
                        if loc2 not in valid_locations:
                            print(f"Warning: The location {loc2} does not exist")
                            continue
                        if distance != '0':  # Skip self-loops
                            self._add_edge(loc1, loc2, int(distance))
        else:
            valid_locations = load_json_file(location_file_path) if "locations" not in load_json_file(location_file_path) else load_json_file(location_file_path)["locations"]
            for loc1 in valid_locations:
                self.locations_info[loc1] = valid_locations[loc1]
                self.locations.append(loc1)
                for loc2 in valid_locations:
                    if loc2 != loc1:
                        self._add_edge(loc1, loc2, default_distance)
                        
    def init_prompt(self,):
        # language x style matrix: zh is a
        # translation of the en set, and ANIMASK_PROMPT_STYLE applies to both
        # languages alike. Fallback default is "neutral" (matches run_resim's
        # CLI default); printed once so every entry point shows what it runs
        style = os.getenv("ANIMASK_PROMPT_STYLE", "neutral")
        neutral = style == "neutral"
        print(f"[prompts] style={style} language={self.language}")
        if self.language == "zh":
            if neutral:
                from modules.prompt import world_agent_prompt_zh_neutral as _p
            else:
                from modules.prompt import world_agent_prompt_zh as _p
        elif neutral:
            from modules.prompt import world_agent_prompt_neutral as _p
        else:
            from modules.prompt import world_agent_prompt_en as _p

        self._ENVIRONMENT_INTERACTION_PROMPT = _p.ENVIRONMENT_INTERACTION_PROMPT
        self._NPC_INTERACTION_PROMPT = _p.NPC_INTERACTION_PROMPT
        # canonical NPCs (those with cards) act with their full persona.
        # Strict attribute access: the old getattr fallback silently served
        # the ORIGINAL text whenever a module lacked the neutral rewrite
        self._CANONICAL_NPC_INTERACTION_PROMPT = _p.CANONICAL_NPC_INTERACTION_PROMPT
        self._SCRIPT_INSTRUCTION_PROMPT = _p.SCRIPT_INSTRUCTION_PROMPT
        self._SCRIPT_ATTENTION = _p.SCRIPT_ATTENTION_PROMPT
        self._DECIDE_NEXT_ACTOR_PROMPT= _p.DECIDE_NEXT_ACTOR_PROMPT
        self._LOCATION_PROLOGUE_PROMPT = _p.LOCATION_PROLOGUE_PROMPT
        self._GENERATE_INTERVENTION_PROMPT = _p.GENERATE_INTERVENTION_PROMPT
        self._UPDATE_EVENT_PROMPT = _p.UPDATE_EVENT_PROMPT
        self._SELECT_SCREEN_ACTORS_PROMPT = _p.SELECT_SCREEN_ACTORS_PROMPT
        self._JUDGE_IF_ENDED_PROMPT = _p.JUDGE_IF_ENDED_PROMPT
        self._LOG2STORY_PROMPT = _p.LOG2STORY_PROMPT
        
    # Agent
    def facts_text(self) -> str:
        facts = getattr(self, "established_facts", None) or []
        return "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"

    def compose_event(self, prose: str) -> str:
        """The event string agents see: prose situation + the verbatim
        standing-facts ledger (world-state that must not drift)."""
        facts = getattr(self, "established_facts", None) or []
        if not facts:
            return prose
        return (prose + "\n[ESTABLISHED FACTS]\n"
                + "\n".join(f"- {f}" for f in facts))

    def _apply_fact_diffs(self, add, retract):
        if not hasattr(self, "established_facts"):
            self.established_facts = []
        def norm(s):
            return re.sub(r"\W+", " ", str(s).lower()).strip()
        for r in retract or []:
            if r in self.established_facts:
                self.established_facts.remove(r)
                continue
            rn = norm(r)
            hit = next((f for f in self.established_facts if norm(f) == rn), None)
            if hit is not None:
                self.established_facts.remove(hit)
            else:
                print(f"[ledger] retract had no match: {str(r)[:80]}")
        seen = {norm(f) for f in self.established_facts}
        for a in add or []:
            # extractor output sometimes arrives pre-bulleted; the renderers
            # add their own "- ", so stored facts must be bare text
            a = re.sub(r"^\s*[-•*]\s*", "", str(a)).strip()
            if a and norm(a) not in seen:
                self.established_facts.append(a)
                seen.add(norm(a))
        if len(self.established_facts) > 60:
            dropped = self.established_facts[:len(self.established_facts) - 60]
            self.established_facts = self.established_facts[-60:]
            print(f"[ledger] capacity: dropped {len(dropped)} oldest fact(s)")

    def update_event(self,
                     cur_event: str,
                     intervention:str,
                     history_text: str,
                     script: str = ""):
        prompt = self._UPDATE_EVENT_PROMPT.format(**{
            "event":cur_event,
            "intervention":intervention,
            "history":history_text,
            "facts":self.facts_text()
        })
        if getattr(self, "identity_notes", ""):
            prompt = (self.identity_notes
                      + "\nIn situation_now and in every ledger fact, refer "
                      "to each such person by ONE appellation only — the one "
                      "most commonly known to the characters present — and "
                      "never as if the appellations were different people.\n"
                      + prompt)
        if script:
            prompt = self._SCRIPT_ATTENTION.format(script = script) + prompt
        raw = self.llm.chat(prompt)
        try:
            parsed = json_parser(raw)
            prose = strip_wrapping_quotes(str(parsed.get("situation_now") or "").strip())
            if not prose:
                raise ValueError("empty situation_now")
            self._apply_fact_diffs(parsed.get("facts_add"),
                                   parsed.get("facts_retract"))
        except Exception:
            # non-JSON reply (e.g. zh prompt set): keep legacy prose behavior
            prose = strip_wrapping_quotes(raw)
        self.record(prose, prompt)
        return prose
    
    def decide_next_actor(self, 
                          history_text: str, 
                          roles_info_text: str,
                          script: str = "",
                          event:str = ""):
        prompt = self._DECIDE_NEXT_ACTOR_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
        })
        
        # adapter-level retries live inside llm.chat; an exception here is a
        # deliberate fatal (credit exhaustion, hard 4xx) and must propagate —
        # the old handler's print(response) referenced an unassigned name and
        # its UnboundLocalError MASKED the real cause in an earlier batch
        response = self.llm.chat(prompt)
        role_code = response
        self.prompts.append({"prompt":prompt,
                            "response":f"{role_code}"})

        return role_code
    
    def judge_if_ended(self,history_text):
        prompt = self._JUDGE_IF_ENDED_PROMPT.format(**{
            "history":history_text
        })
        max_tries = 3
        response = {"if_end":True, "detail":""}
        for _ in range(max_tries):
            try:
                response.update(json_parser(self.llm.chat(prompt)))
                break
            except Exception as e:
                print(f"Parsing failure! Error:", e)    
                print(response)
        
        return response["if_end"],response["detail"]
        
    def decide_scene_actors(self,roles_info_text, history_text, event, previous_role_codes):
        prompt = self._SELECT_SCREEN_ACTORS_PROMPT.format(**{
            "roles_info":roles_info_text,
            "history_text":history_text,
            "event":event,
            "previous_role_codes":previous_role_codes
            
        })
        for i in range(3):
            response = self.llm.chat(prompt)
            try:
                m = re.search(r"\[.*\]", response or "", re.S)
                # literal_eval, not eval: never execute model output
                role_codes = ast.literal_eval(m.group(0) if m else response)
                if isinstance(role_codes, list):
                    return [str(c) for c in role_codes]
            except Exception as e:
                print(f"Scene-actor parsing failure! {i+1}th tries. Error:", e)
                print(response)
        # scheduler unusable this round — cast no one new; the caller falls
        # back to the previous group / full roster logic
        return []
    
    def generate_location_prologue(self,
                                   location_code,
                                   history_text,
                                   event,
                                   location_info_text):
        prompt = self._LOCATION_PROLOGUE_PROMPT.format(**{
            "location_name":self.locations_info[location_code]["location_name"],
            "location_description":self.locations_info[location_code]["location_name"],
            "location_info":location_info_text,
            "history_text":history_text,
            "event":event,
            "world_description":self.description
        })
        response = self.llm.chat(prompt)
        self.record(detail = response,prompt = prompt)
        return "\n"+response
    
    def environment_interact(self,
                            action_maker_name: str,
                            action: str,
                            action_detail: str,
                            location_code: str):
        references = self.retrieve_references(query = action_detail)
        if getattr(self, "identity_notes", ""):
            references = self.identity_notes + "\n" + references
        if getattr(self, "established_facts", None):
            references = ("ESTABLISHED FACTS (in force verbatim; outcomes "
                          "must not contradict them):\n" + self.facts_text()
                          + "\n" + references)
        # on-demand canonical fact supplementation via the archivist
        if getattr(self, "archivist", None):
            note = self.archivist.consult(action_detail)
            if note:
                references = (references + "\n" if references else "") + note
        # an in-transit role (location_code=None) can still be cast into a
        # scene and choose an environment interaction — locations_info[None]
        # crashed the whole run (KeyError) in a pilot run; fall back to
        # a neutral description instead
        loc_info = self.locations_info.get(location_code) or {}
        loc_desc = loc_info.get("detail") or (
            "在途中（不在任何已注册地点）" if self.language == "zh"
            else "in transit (not at any registered location)")
        prompt = self._ENVIRONMENT_INTERACTION_PROMPT.format(**
            {
                "role_name":action_maker_name,
                "action":action,
                "action_detail":action_detail,
                "world_description":self.description,
                "location":location_code or ("途中" if self.language == "zh"
                                             else "in transit"),
                "location_description":loc_desc,
                "references":references,
            }
            )
        response = "无事发生。" if self.language == "zh" else "Nothing happens."
        for i in range(3):
            try:
                response = self.llm.chat(prompt) 
                if response:
                    break
            except Exception as e:
                print("Enviroment Interaction failed! {i}th tries. Error:", e)
        self.record(response, prompt)
        return response
    
    
    def resolve_npc(self, target_name: str):
        """Match a free-text NPC target (e.g. "the housekeeper or Madam
        Gerton") to a canonical NPC card by token overlap with the card's
        name and aliases. Returns (canonical_name, card | None); the
        original string is returned unchanged when nothing matches."""
        cards = getattr(self, "npc_cards", None) or {}
        if not target_name or not cards:
            return target_name, None
        def toks(s):
            return set(re.findall(r"[a-z]+", str(s).lower())) - {"the", "a", "an", "or"}
        want = toks(target_name)
        # rank: most overlapping words first, fewest unmatched words second —
        # so "Gemma" resolves to the card "Gemma", not "Gemma and Jacub's Son"
        best, best_card, best_key = None, None, (0, -999)
        for name, card in cards.items():
            for alias in [name] + list(card.get("aliases") or []):
                nt = toks(alias)
                if not nt:
                    continue
                overlap = len(nt & want)
                full = overlap > 0 and (nt <= want or want <= nt)
                if overlap < 2 and not full:
                    continue
                key = (overlap, -len(nt - want))
                if key > best_key:
                    best, best_card, best_key = name, card, key
        if best:
            return best, best_card
        return target_name, None

    def resolve_cast_target(self, target_name: str, role_agents: Dict[str, Any]):
        """A target name that is really an appellation of a CAST member must
        never be played by the world as an NPC — return that member's
        role_code so the caller reroutes to a role-to-role interaction.
        Matches against the identity registry's alias table plus the cast's
        own role names/nicknames (fallback when no identity map exists)."""
        if not target_name:
            return None
        def toks(s):
            # ASCII words + CJK runs (zh names tokenize as whole runs; the
            # en path is unchanged)
            return set(re.findall(r"[a-z]+|[一-鿿]+",
                                  str(s).lower())) - {"the", "a", "an"}
        def strip_possessive(s):
            # en: "Deacon's household manager" targets the manager;
            # zh: "廉总的女人" targets 女人, not 廉总
            s = re.sub(r"\b[\w .]+?['’]s\b", " ", str(s))
            return re.sub(r"[一-鿿]+的", " ", s)
        want = toks(strip_possessive(target_name))
        if not want:
            return None
        alias_by_cast: Dict[str, list] = {}
        for code, agent in role_agents.items():
            names = [getattr(agent, "role_name", ""), getattr(agent, "nickname", "")]
            names += (getattr(self, "cast_aliases", {}) or {}).get(
                getattr(agent, "role_name", ""), [])
            alias_by_cast[code] = [n for n in names if n]
        best, best_overlap = None, 0
        for code, names in alias_by_cast.items():
            for alias in names:
                at = toks(strip_possessive(alias))
                # long strings are descriptions, not names — never match on them
                if not at or len(at) > 4:
                    continue
                has_proper = any(w[:1].isupper() for w in str(alias).split()
                                 if w.lower() not in ("the", "a", "an")) \
                    or bool(re.search(r"[一-鿿]{2,}", str(alias)))
                overlap = len(at & want)
                if at == want:
                    matched = True          # exact reference, generic or proper
                elif has_proper and at < want:
                    matched = True          # "Nicolus Yevin" within a longer target
                elif has_proper and want < at and len(at) <= 3:
                    matched = True          # "Yevin" for "Nicolus Yevin"
                else:
                    matched = False
                if matched and overlap > best_overlap:
                    best, best_overlap = code, overlap
        return best

    def npc_interact(self,
                     action_maker_name: str,
                     action_detail: str,
                     location_name: str,
                     target_name: str):
        references = self.retrieve_references(query = action_detail)
        if getattr(self, "identity_notes", ""):
            references = self.identity_notes + "\n" + references
        if getattr(self, "established_facts", None):
            references = ("ESTABLISHED FACTS (in force verbatim; your "
                          "response must not contradict them):\n"
                          + self.facts_text() + "\n" + references)
        # ground the NPC in its canonical card, and give it the
        # same canonical-fact channel the environment adjudicator has
        target_name, card = self.resolve_npc(target_name)
        if card:
            profile = "WHO YOU ARE (canonical profile — stay consistent with it):\n" \
                + f"- identity: {card.get('identity', '')}\n" \
                + (f"- manner: {card.get('manner', '')}\n" if card.get("manner") else "") \
                + "".join(f"- {f}\n" for f in card.get("facts") or []) \
                + "".join(f"- {r}\n" for r in card.get("relationships") or [])
            references = profile + "\n" + references
        if getattr(self, "archivist", None):
            note = self.archivist.consult(
                f"(the NPC {target_name} must respond to this) " + action_detail)
            if note:
                references = references + "\n" + note
        # a carded NPC is an established character: play the persona, not a
        # generic minor figure
        npc_prompt = self._CANONICAL_NPC_INTERACTION_PROMPT if card \
            else self._NPC_INTERACTION_PROMPT
        prompt = npc_prompt.format(**
            {
                "role_name":action_maker_name,
                "action_detail":action_detail,
                "world_description":self.description,
                "target":target_name,
                "references":references,
                "location":location_name
            }
            )
        
        npc_interaction = {"if_end_interaction":True,"detail":"无事发生。"} if self.language == "zh" else {"if_end_interaction":True,"detail":"Nothing happens"}
        try:
            # shape-check BEFORE adopting: a valid-JSON reply missing
            # "detail" used to be returned as-is and crash the caller's
            # ["detail"] access mid-run
            parsed = json_parser(self.llm.chat(prompt))
            response = parsed["detail"]
            self.record(response, prompt)
            parsed.setdefault("if_end_interaction", True)
            npc_interaction = parsed
        except Exception as e:
            print("Enviroment Interaction failed!",e)

        return npc_interaction
    
    
    def get_script_instruction(self, 
                               roles_info_text: str, 
                               event: str, 
                               history_text: str, 
                               script: str, 
                               last_progress: str):
        prompt = self._SCRIPT_INSTRUCTION_PROMPT.format(**{
            "roles_info":roles_info_text,
            "event":event,
            "history_text":history_text,
            "script":script,
            "last_progress":last_progress
        })
        max_tries = 3
        instruction = {}
        for i in range(max_tries):
            response = self.llm.chat(prompt)
            try:
                instruction = json_parser(response)
                break
            except Exception as e:
                print(f"Parsing failure! {i+1}th tries. Error:", e)   
                print(response)
        self.record(response, prompt)
        return instruction
    
    def generate_event(self,roles_info_text: str, event: str, history_text: str):
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        response = self.llm.chat(prompt)
        self.record(response, prompt)
        return response
        
    def generate_script(self, roles_info_text: str, event: str, history_text: str):
        prompt = self._GENERATE_INTERVENTION_PROMPT.format(**{
            "world_description":self.description,
            "roles_info":roles_info_text,
            "history_text":history_text
        })
        response = self.llm.chat(prompt)
        self.record(response, prompt)
        return response
    
    def log2story(self,logs):
        prompt = self._LOG2STORY_PROMPT.format(**{
            "logs":logs
        })
        response = self.llm.chat(prompt)
        return response
    
    # Other
    def record(self, detail: str, prompt: str = ""):
        if prompt:
            self.prompts.append({"prompt":prompt,
                                 "response":detail})
        self.history.append(detail)
    
    def add_location_during_simulation(self, location: str, detail: str):
        self.locations.append(location)
        # display name: split the CamelCase code into words so the raw code
        # never has to appear in narration ("PublicInn" -> "Public Inn")
        display = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", location).strip() or location
        self.locations_info[location] = {
            'location_code': location,
            "location_name": display,
            'description': '',
            'detail':detail
        }
        for loc in self.locations:
            if loc != location:
                self._add_edge(loc, location, 1)
                self._add_edge(location,loc, 1)
        return
    
    def retrieve_references(self, query: str, top_k = 3, max_words = 100):
        if self.db is None:
            return ""
        references = "\n".join(self.db.search(query, top_k,self.db_name))
        references = references[:max_words]
        return references

    def find_location_name(self, code: str):
        return self.locations_info[code]["location_name"]
              
    def _add_location(self, code: str, location_info: Dict[str, Any]):
        self.locations_info[code] = location_info
        
    def _add_edge(self, code1: str, code2: str, distance: int):
        self.edges[(code1,code2)] = distance
        self.edges[(code2,code1)] = distance  
        
    def get_distance(self, code1: str, code2: str):
        if (code1,code2) in self.edges:
            return self.edges[(code1,code2)]
        else:
            return None
        
    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() 
                 if isinstance(value, (str, int, list, dict, float, bool, type(None)))
                 and (key not in ['llm','embedding','db','locations_info','edges','world_data','world_settings']
                 and "PROMPT" not in key)
                 }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./world_agent.json")
        save_json_file(filename, self.__getstate__() )

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./world_agent.json")
        state = load_json_file(filename)
        self.__setstate__(state)  

