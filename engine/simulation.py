# Modified file; the original work is under Apache-2.0, see README (License).
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:  # resim fork: tqdm is decorative only
    def tqdm(x, *a, **k):
        return x
import json
import os
import re
import warnings
import random
from typing import Any, Dict, List, Optional, Literal
from collections import defaultdict
import uuid

from utils import *
from modules.llm.BaseLLM import set_call_ctx
from modules.main_role_agent import RPAgent
from modules.world_agent import WorldAgent
from modules.history_manager import HistoryManager
from modules.embedding import get_embedding_model
from modules.timekeeper import VETO_PROMPT, SETTLE_PROMPT, _jparse, span_text

warnings.filterwarnings('ignore')

class Simulation():
    def __init__(self,
                 preset_path: str,
                 world_llm_name: str,
                 role_llm_name: str,
                 embedding_name:str = "bge-small") :
        """
        The initialization function of the system.
        
        Args:
            preset_path (str): The path to config file of this experiment.
            world_llm_name (str, optional): The base model of the world agent. Defaults to 'gpt-4o'.
            role_llm_name (str, optional): The base model of all the role agents. Defaults to 'gpt-4o'.
            mode (str, optional): If set to be 'script', the role agents will act according to the given script. 
                                  If set to be 'free', the role agents will act freely based on their backround.
                                  Defaults to 'free'.
        """
        
        self.role_llm_name: str = role_llm_name
        self.world_llm_name: str = world_llm_name
        self.embedding_name:str = embedding_name
        config = load_json_file(preset_path)
        self.preset_path = preset_path
        self.config: Dict = config
        self.experiment_name: str = os.path.basename(preset_path).replace(".json","") + "/" + config["experiment_subname"] + "_" + role_llm_name
        
        role_agent_codes: List[str] = config['role_agent_codes']
        world_file_path: str = config["world_file_path"]
        map_file_path: str = config["map_file_path"] if "map_file_path" in config else ""
        role_file_dir: str = config["role_file_dir"] if "role_file_dir" in config else "./data/roles/"
        loc_file_path: str = config["loc_file_path"]
        self.intervention: str = config["intervention"] if "intervention" in config else ""
        self.event = self.intervention
        self.script: str = config["script"] if "script" in config else ""
        self.language: str = config["language"] if "language" in config else "zh"
        self.source:str = config["source"] if "source" in config else ""
        
        self.idx: int = 0
        self.cur_round: int = 0
        # data contract: every goal change is appended here and serialized
        # into timeline_<tag>.json (C3 goal rows / B1 decision-point seeds)
        self.goal_log: List[Dict[str, Any]] = []
        self.progress: str = "剧本刚刚开始，还什么都没有发生" if self.language == 'zh' else "The story has just begun, nothing happens yet."
        self.moving_roles_info: Dict[str, Any] = {}   
        self.history_manager = HistoryManager()
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_status = {
            "location_code":"",
            "group":role_agent_codes,
        }
        self.scene_characters = {}
        self.event_history = []
        
        self.role_llm = get_models(role_llm_name)
        self.logger = get_logger(self.experiment_name)
        
        self.embedding = get_embedding_model(embedding_name, language=self.language)
        # llm=None: each agent builds its own (stateless) adapter instance so
        # ANIMASK_CALL_LOG attribution tags don't clobber each other; outputs
        # are unchanged (adapters reset messages on every chat() call)
        self.init_role_agents(role_agent_codes = role_agent_codes,
                            role_file_dir = role_file_dir,
                            world_file_path=world_file_path,
                            llm = None,
                            embedding = self.embedding)
        
        if world_llm_name == role_llm_name:
            self.world_llm = self.role_llm
        else:
            self.world_llm = get_models(world_llm_name)
        self.init_world_agent_from_file(world_file_path = world_file_path,
                                        map_file_path = map_file_path,
                                        loc_file_path = loc_file_path,
                                        llm = None,
                                        embedding = self.embedding)
        setattr(self.world_llm, "_resim_tag", "server")
    
    # Init
    def init_role_agents(self, 
                         role_agent_codes: List[str], 
                         role_file_dir:str, 
                         world_file_path:str,
                         llm=None,
                         embedding=None) -> None:
        self.role_codes: List[str] = role_agent_codes
        self.role_agents: Dict[str, RPAgent] = {}
        
        for role_code in role_agent_codes:
            if check_role_code_availability(role_code,role_file_dir):
                self.role_agents[role_code] = RPAgent(role_code=role_code, 
                                                      role_file_dir=role_file_dir,
                                                      world_file_path=world_file_path,
                                                      source = self.source, 
                                                      language=self.language,
                                                      llm_name = self.role_llm_name,
                                                      llm = llm,
                                                      embedding_name=self.embedding_name,
                                                      embedding = embedding
                                                      )
                # print(f"{role_code} Initialized.")
            else:
                print(f"Warning: The specified role `{role_code}` does not exist.")
    
    def init_world_agent_from_file(self, 
                                   world_file_path: str, 
                                   map_file_path: str,
                                   loc_file_path: str,
                                   llm=None,
                                   embedding=None) -> None:
        self.world_agent: WorldAgent = WorldAgent(world_file_path = world_file_path, 
                                                  location_file_path = loc_file_path,
                                                  map_file_path = map_file_path, 
                                                  llm_name=self.world_llm_name,
                                                  llm = llm,
                                                  embedding_name=self.embedding_name,
                                                  embedding = embedding,
                                                  language=self.language)
        for role_code in self.role_agents:
            self.role_agents[role_code].world_db = self.world_agent.db
            self.role_agents[role_code].world_db_name = self.world_agent.db_name
        
    def init_role_locations(self, random_allocate: bool = True):
        """
        Set initial positions of the roles.
        
        Args:
            random_allocate (bool, optional): if set to be True, the initial positions of the roles are randomly assigned. Defaults to True.
        """
        init_locations_code = random.choices(self.world_agent.locations, k = len(self.role_codes))
        for i,role_code in enumerate(self.role_codes):
            self.role_agents[role_code].set_location(init_locations_code[i], self.world_agent.find_location_name(init_locations_code[i]))
            info_text = f"{self.role_agents[role_code].nickname} 现在位于 {self.world_agent.find_location_name(init_locations_code[i])}" \
                if self.language == "zh" else f"{self.role_agents[role_code].nickname} is now located at {self.world_agent.find_location_name(init_locations_code[i])}"
            self.log(info_text)
    
    def reset_llm(self, role_llm_name, world_llm_name):
        self.role_llm = get_models(role_llm_name)
        for role_code in self.role_codes:
            self.role_agents[role_code].llm = self.role_llm
            self.role_agents[role_code].llm_name = role_llm_name
        if world_llm_name == role_llm_name:
            self.world_llm = self.role_llm
        else:
            self.world_llm = get_models(world_llm_name)
        self.world_agent.llm = self.world_llm
        self.role_llm_name = role_llm_name
        self.world_llm_name = world_llm_name
        
    # Simulation        
    def simulate_generator(self, 
                 rounds: int = 10, 
                 save_dir: str = "", 
                 if_save: Literal[0,1] = 0,
                 mode: Literal["free", "script"] = "free",
                 scene_mode: Literal[0,1] = 1,):
        """
        The main function of the simulation.

        Args:
            rounds (int, optional): The max rounds of simulation. Defaults to 10.
            save_dir (str, optional): _description_. Defaults to "".
            if_save (Literal[0,1], optional): _description_. Defaults to 0.
        """
        self.mode = mode 
        meta_info: Dict[str, Any] = self.continue_simulation_from_file(save_dir)            
        self.if_save: int = if_save
        start_round: int = meta_info["round"]
        sub_start_round:int = meta_info["sub_round"] if "sub_round" in meta_info else 0
        if start_round == rounds: return 
        
        # Setting Locations
        if not meta_info["location_setted"]:
            self.log("========== Start Location Setting ==========")
            self.init_role_locations()
            self._save_current_simulation("location")
            
        # Setting Goals
        if not meta_info["goal_setted"]:
            yield ("system","","-- Setting Goals --",None) 
            self.log("========== Start Goal Setting ==========")
            
            if self.mode == "free":
                self.get_event()
                self.log(f"--------- Free Mode: Current Event ---------\n{self.event}\n")
                yield ("system","",f"--------- Current Event ---------\n{self.event}\n", None) 
                self.event_history.append(self.event)
            elif self.mode == "script":
                self.get_script()
                self.log(f"--------- Script Mode: Setted Script ---------\n{self.script}\n")
                yield ("system","",f"--------- Setted Script ---------\n{self.script}\n", None) 
                self.event_history.append(self.event)
            if self.mode == "free":
                for role_code in self.role_codes:
                    motivation = self.role_agents[role_code].set_motivation(
                        world_description = self.world_agent.description,
                        other_roles_info = self._get_group_members_info_dict(self.role_agents),
                        intervention = self.event,
                        script = self.script
                        )
                    # two-tier goals: seed the current goal (goal_now from
                    # extraction wins; otherwise derived from the motivation
                    # via upstream's ROLE_SET_GOAL_PROMPT)
                    goal = self.role_agents[role_code].set_goal(
                        world_description = self.world_agent.description,
                        other_roles_info = self._get_group_members_info_dict(self.role_agents),
                        )
                    self.goal_log.append({"round": self.cur_round,
                                          "role": role_code, "old": "",
                                          "new": goal, "via": "set_goal"})
                    if self.language == "zh":
                        info_text = (f"{self.role_agents[role_code].nickname} 设立了动机: {motivation}"
                                     + (f"\n当前目标: {goal}" if goal and goal != motivation else ""))
                    else:
                        info_text = (f"{self.role_agents[role_code].nickname} has set the motivation: {motivation}"
                                     + (f"\nCurrent goal: {goal}" if goal and goal != motivation else ""))

                    record_id = str(uuid.uuid4())
                    self.log(info_text)
                    self.record(role_code=role_code,
                                detail=info_text,
                                actor = role_code,
                                group = [role_code],
                                actor_type = 'role',
                                act_type="goal setting",
                                record_id = record_id)
                    # resim fork: motivation text is a second-person system
                    # directive (pack format), not an in-fiction utterance —
                    # surface it on the system channel, not the character's
                    yield ("system","",info_text,record_id)
                    
            self._save_current_simulation("goal")
            
        yield ("system","","-- Simulation Started --",None)
        selected_role_codes = []
        # Simulating
        for current_round in range(start_round, rounds):
            self.cur_round = current_round
            set_call_ctx(round=current_round)  # stamps every call log line
            self.log(f"========== Round {current_round+1} Started ==========")
            # settle an all-in-transit round BEFORE announcing the event —
            # otherwise the unchanged event is yielded twice back to back
            # (once here, once after the `continue` re-enters the loop)
            if len(self.moving_roles_info) == len(self.role_codes):
                self.settle_movement()
                continue

            if self.event and current_round >= 1:
                self.log(f"--------- Current Event ---------\n{self.event}\n")
                yield ("world","","-- Current Event --\n"+self.event, None)
                self.event_history.append(self.event)
            
            # Characters in next scene
            if scene_mode:
                group = self._name2code(
                    self.world_agent.decide_scene_actors(
                        self._get_locations_info(False),
                        self.history_manager.get_recent_history(5),
                        self.event,
                        list(set(selected_role_codes + list(self.moving_roles_info.keys())))))
                # the world may cast an NPC it plays itself (e.g. a canonical
                # figure entering the scene) — only castable role agents may
                # enter the group, or every role_agents[code] lookup downstream
                # KeyErrors on the NPC's pass-through name
                dropped = [c for c in group if c not in self.role_agents]
                if dropped:
                    self.log(f"[scene] non-agent cast dropped from group: {dropped}")
                group = [c for c in group if c in self.role_agents]
                selected_role_codes += group
                if len(selected_role_codes) >= len(self.role_codes):
                    selected_role_codes = []
            else:
                group = self.role_codes
            # resim fork: characters unable to act (asleep/unconscious at freeze)
            # cannot be cast until something wakes them
            awake = [c for c in group if not getattr(self.role_agents[c], "asleep", False)]
            if not awake:
                awake = [c for c in self.role_codes
                         if not getattr(self.role_agents[c], "asleep", False)]
            group = awake
            self.current_status['group'] = group
            self.current_status['location_code'] = self.role_agents[group[0]].location_code
            self.scene_characters[str(current_round)] = group
            # Prologue 
            # if current_round == 0 and len(group) > 0 
            #     prologue = self.world_agent.generate_location_prologue(location_code=self.role_agents[group[0]].location_code, history_text=self._get_history_text(group),event=self.event,location_info_text=self._find_roles_at_location(self.role_agents[group[0]].location_code,name=True))
            #     self.log("--Prologue--: "+prologue)
            #     self.record(role_code="None",detail=prologue,act_type="prologue")
            start_idx = len(self.history_manager)

            sub_round = sub_start_round
            last_actor = None
            for sub_round in range(sub_start_round, getattr(self, 'sub_rounds', 3)):
                if self.mode == "script":
                    self.script_instruct(self.progress)
                else:
                    for role_code in group:
                        _old_goal = self.role_agents[role_code].goal
                        self.role_agents[role_code].update_goal(other_roles_status=self._get_status_text(self.role_codes))
                        _new_goal = self.role_agents[role_code].goal
                        if _new_goal != _old_goal:
                            self.goal_log.append(
                                {"round": self.cur_round, "role": role_code,
                                 "old": _old_goal, "new": _new_goal,
                                 "via": "update_goal"})

                for slot_code in group:
                    role_code = slot_code
                    if scene_mode:
                        cand = self._name2code(self.world_agent.decide_next_actor("\n".join(self.history_manager.get_recent_history(3)),self._get_group_members_info_text(group,status=True),self.script))
                        # resim fork: the scheduler must return a castable
                        # member of this scene, and never the same character
                        # twice in a row — repeats yield verbatim echo turns
                        if cand not in self.role_agents or cand not in group:
                            cand = slot_code
                        if cand == last_actor and len(group) > 1:
                            cand = next(c for c in group if c != last_actor)
                        role_code = cand
                    last_actor = role_code
                    yield from self.implement_next_plan(role_code = role_code,
                                            group = group)
                    self._save_current_simulation("action", current_round, sub_round)

                if_end,epilogue = self.world_agent.judge_if_ended("\n".join(self.history_manager.get_recent_history(len(self.history_manager)-start_idx)))
                if if_end:
                    record_id = str(uuid.uuid4())
                    self.log("--Epilogue--: "+epilogue)
                    self.record(role_code = "None",
                                detail = epilogue, 
                                actor_type="world",
                                act_type="epilogue", 
                                actor = "world",
                                group = [],
                                record_id = record_id)
                    yield ("world","","--Epilogue--: "+epilogue, record_id)

                    break

                # resim fork: a solo scene has no interlocutor — one beat per
                # round; extra sub-rounds only re-generate near-identical turns
                if len(group) == 1:
                    break
            
                
            for role_code in group:
                yield from self.decide_whether_to_move(role_code = role_code,
                                            group = self._find_group(role_code))
                self.role_agents[role_code].update_status()
                
            self.settle_movement()
            self.update_event(group)

            # resim fork: judge-side termination check (tension resolution +
            # stagnation); verdicts never enter agent contexts
            if getattr(self, "terminator", None):
                tk = getattr(self, "timekeeper", None)
                latest = "\n".join(self.history_manager.get_subsequent_history(start_idx))
                verdict = self.terminator.check(
                    latest, getattr(self, "_last_round_text", ""),
                    pending_commitments=tk.pending_text() if tk else "")
                self._last_round_text = latest
                self.log(f"--Referee-- {verdict}")
                yield ("world", "",
                       f"-- Referee: outcome={verdict.get('outcome')} "
                       f"material_change={verdict.get('material_change')} | "
                       f"{verdict.get('reason', '')}", None)

                # commitment ledger: time-bound facts the fiction just created
                # (an appointment, a promised reply, a deadline) become one-shot
                # queue entries, so skips clip to them instead of leaping over.
                # Judge-side bookkeeping — logged, never yielded to agents.
                if tk is not None:
                    for c in verdict.get("new_commitments") or []:
                        nm = tk.add_event(c["description"], c["due_in_hours"],
                                          c["kind"])
                        if nm:
                            self.log(f"[ledger] + {nm} ({c['kind']}, due "
                                     f"~{c['due_in_hours']:.0f}h): "
                                     f"{c['description'][:100]}")
                    for nm in verdict.get("resolved_commitments") or []:
                        if tk.remove_event(nm):
                            self.log(f"[ledger] - {nm}: resolved/moot")
                        else:
                            # a bounced removal used to be a silent no-op:
                            # the settled appointment stayed in {pending}
                            # and later fired as a phantom event
                            self.log(f"[ledger] removal bounced: '{nm}' "
                                     f"matched no open commitment")

                # v3: plot-driven termination outranks every time mechanism —
                # a settled resolution ledger must never be swallowed by a
                # same-round skip proposal
                if verdict.get("terminate") and str(verdict.get("outcome", ""))\
                        .startswith(("resolved", "motivations")):
                    yield ("world", "",
                           f"-- SIMULATION TERMINATED: {verdict['outcome']} --",
                           None)
                    return

                # live-beat guard: a landed appointment is a scene owed to the
                # record — no gap or skip may leap it. Released once a round
                # engages it (material change) or after 4 quiet checks (the
                # beat never ignited — logged, so the miss stays visible
                # instead of being silently erased by a three-week skip).
                lb = getattr(self, "_live_beat", None)
                if lb is not None:
                    if verdict.get("material_change"):
                        self.log(f"[beat] '{lb['name']}' engaged; skip guard off")
                        self._live_beat = lb = None
                    else:
                        lb["age"] += 1
                        if lb["age"] >= 4:
                            self.log(f"[beat] '{lb['name']}' never ignited "
                                     f"after {lb['age']} checks; skip guard off")
                            self._live_beat = lb = None

                # v2 time design: mirror fine-mode hours into the exogenous
                # event queue; events crossed during fine play enter the world
                if tk is not None:
                    # v2 middle gear: scene-level ellipsis ("later that day",
                    # "next morning") — the referee's natural_gap_hours.
                    # Overnight-scale gaps (>=6h) are classic covert-action
                    # windows, so they run the same private veto poll as big
                    # skips: a character may quietly claim the night.
                    gap = verdict.get("natural_gap_hours")
                    gap_applied = 0.0
                    if isinstance(gap, (int, float)) and gap > 0 and lb is None:
                        if gap >= 6:
                            prop_gap = {"hours": float(gap), "event": {
                                "name": "scene_break",
                                "description": (
                                    "一段安静的时光（过夜/直到下一个自然节点）"
                                    if self.language == "zh" else
                                    "a quiet stretch (overnight / "
                                    "until the next natural beat)"),
                                "due_hours": float(gap), "period_hours": None}}
                            yield from self.attempt_time_skip(prop_gap)
                            # advance happened inside on success; veto means
                            # the night stays live and play continues
                        else:
                            rid = str(uuid.uuid4())
                            gap_txt = (f"About {gap:g} hour passes"
                                       if abs(gap - 1) < 1e-9
                                       else f"About {gap:g} hours pass")
                            note = (f"[SCENE BREAK] {gap_txt}; "
                                    f"the characters disperse before the next "
                                    f"beat.")
                            self.record(role_code="None", detail=note,
                                        actor_type="world", act_type="scene_break",
                                        actor="world", group=list(self.role_agents),
                                        record_id=rid)
                            yield ("world", "", note, rid)
                            gap_applied = float(gap)
                    h = (verdict.get("elapsed_hours_this_round") or 0) + gap_applied
                    if isinstance(h, (int, float)) and h > 0:
                        for ev in tk.advance(float(h)):
                            rid = str(uuid.uuid4())
                            times = ev.get("count", 1)
                            note = f"[WORLD EVENT] {ev['description'] or ev['name']}" \
                                + (f" (occurred {times} times during this stretch)"
                                   if times > 1 else "")
                            self.record(role_code="None", detail=note,
                                        actor_type="world", act_type="world_event",
                                        actor="world", group=list(self.role_agents),
                                        record_id=rid)
                            yield ("world", "", note, rid)
                            if ev.get("kind") == "appointment" \
                                    and getattr(self, "_live_beat", None) is None:
                                # a scheduled beat just landed in fine play:
                                # protect it until it is actually played
                                self._live_beat = {"name": ev["name"], "age": 0}
                                self.log(f"[beat] appointment '{ev['name']}' "
                                         f"landed; skip guard on")
                    if (yield from self._anchor_checkpoint(tk)):
                        return

                # v2 gearshift: "quiet" = no material change OR negligible
                # story time. Two consecutive quiet rounds propose a skip —
                # long before stagnation escalates to a deadlock verdict.
                h = verdict.get("elapsed_hours_this_round")
                quiet = (verdict.get("material_change") is False) or \
                        (isinstance(h, (int, float)) and h < 0.15)
                # a skip may only launch from a natural break: when the
                # referee says the action continues immediately (gap 0), the
                # scene is still live — skipping now freezes characters
                # mid-conversation and swallows the scene's outcome. A truly
                # stuck scene still escapes via the deadlock verdict below.
                g = verdict.get("natural_gap_hours")
                if quiet and isinstance(g, (int, float)) and g <= 0:
                    quiet = False
                self._quiet_streak = getattr(self, "_quiet_streak", 0) + 1 if quiet else 0

                is_deadlock = verdict.get("terminate") and \
                    verdict.get("outcome") == "deadlock"
                prop = tk.propose_skip(getattr(self, "canon_anchor", None)) \
                    if tk is not None else None
                # pacing pressure (timekeeper-level, invisible to agents):
                # far from the anchor -> favor passage of time (skip after 1
                # quiet round); near it -> favor fine-grained play (2 rounds)
                needed_quiet = 2
                if tk is not None:
                    remaining = tk.anchor_target_hours(getattr(self, "canon_anchor", None))
                    if remaining is not None and remaining > tk.max_skip_hours:
                        needed_quiet = 1
                # skips need a real span (a sub-hour hop is pure overhead: a
                # veto poll + settlement that advance the clock by minutes)
                # and must never leap a landed-but-unplayed appointment
                lb_live = getattr(self, "_live_beat", None) is not None
                skippable = prop is not None and prop["hours"] >= 0.5 \
                    and not lb_live
                terminate_now = verdict.get("terminate")
                if is_deadlock and (lb_live or
                                    (prop is not None and prop["hours"] < 0.5)):
                    # standing at (or one step from) a scheduled beat: the
                    # story is not dead — release the streak and let fine play
                    # ignite the beat (the live-beat guard keeps this bounded)
                    self.terminator.stagnant_streak = 0
                    is_deadlock = terminate_now = False
                if (is_deadlock or self._quiet_streak >= needed_quiet) and skippable:
                    # a veto is honored once; if its grace round was quiet
                    # again, the claimed action was routine — force the skip
                    force = getattr(self, "_veto_grace", 0) >= 1
                    skipped = yield from self.attempt_time_skip(prop, force=force)
                    if skipped:
                        self._quiet_streak = 0
                        self._veto_grace = 0
                        self.terminator.stagnant_streak = 0
                        if (yield from self._anchor_checkpoint(tk)):
                            return
                    else:
                        self._veto_grace = getattr(self, "_veto_grace", 0) + 1
                        self._quiet_streak = 0
                        self.terminator.stagnant_streak = max(
                            0, self.terminator.stagnant_streak - 1)
                elif terminate_now:
                    # deadlock with no possible skip = nothing will ever change
                    yield ("world", "", f"-- SIMULATION TERMINATED: {verdict['outcome']} --", None)
                    return

            sub_start_round = 0
            self._save_current_simulation("action", current_round + 1,sub_round + 1)

        # resim fork: mark round-cap exhaustion explicitly — otherwise the
        # transcript ends mid-scene with no stated reason
        tk = getattr(self, "timekeeper", None)
        now_txt = f", now={tk.now_hours:.0f}h" if tk is not None else ""
        yield ("world", "",
               f"-- SIMULATION TERMINATED: rounds_exhausted "
               f"(cap={rounds}{now_txt}) --", None)

    # resim fork (v3): the anchor is a measurement coordinate, not a guillotine
    def _anchor_checkpoint(self, tk):
        """Handle reaching the canonical ending's time coordinate.
        anchor_mode="stop": terminate there (legacy same-instant slice).
        anchor_mode="snapshot": record the time-aligned state once — the
        evaluation still gets its same-instant snapshot — and let play
        continue; the plot (resolution ledger, deadlock, round cap) decides
        when the story is over. Returns True if the caller should stop."""
        if not tk.anchor_reached(getattr(self, "canon_anchor", None)):
            return False
        if getattr(self, "anchor_mode", "stop") != "snapshot":
            yield ("world", "",
                   f"-- SIMULATION TERMINATED: time_reached "
                   f"(event-coordinate anchor, now={tk.now_hours:.0f}h) --", None)
            return True
        if not getattr(self, "_anchor_snapshotted", False):
            self._anchor_snapshotted = True
            self.anchor_snapshot = {
                "round": self.cur_round,
                "now_hours": tk.now_hours,
                "occurrences": dict(tk.occurrences),
                "event": self.event,
                "roles": {code: {"status": a.status,
                                 "location": a.location_name,
                                 "goal": a.goal}
                          for code, a in self.role_agents.items()}}
            self.log(f"[anchor] time-aligned snapshot taken at "
                     f"now={tk.now_hours:.0f}h")
            yield ("world", "",
                   f"-- CANON TIME COORDINATE REACHED (now={tk.now_hours:.0f}h):"
                   f" time-aligned snapshot recorded; play continues until the"
                   f" plot settles --", None)
        return False

    # resim fork (v2 time design): the time-skip gearshift
    def attempt_time_skip(self, prop, force=False):
        """Private veto poll -> neutral settlement -> land on the skip target.
        Returns True if the skip happened. The poll never enters the public
        record (information asymmetry preserved).

        force=True: a previous veto's grace round produced no material change,
        so vetoes no longer block — the vetoer's intent becomes an ongoing
        pursuit folded into the settlement (advanced neutrally, no events)."""
        tk = self.timekeeper
        hours, event = prop["hours"], prop["event"]
        event_desc = event["description"] or event["name"]
        routines, veto, overridden = {}, None, []
        for code, agent in self.role_agents.items():
            if getattr(agent, "asleep", False):
                continue
            r = _jparse(agent.llm.chat(VETO_PROMPT.format(
                name=agent.role_name, profile_head=agent.role_profile[:600],
                goal=agent.goal, status=agent.status, situation=self.event,
                span=span_text(hours), event_desc=event_desc)))
            if r.get("let_time_pass") is False:
                if not force:
                    veto = (code, r.get("intended_action", ""))
                    break
                overridden.append(code)
                routines[code] = ("(pursuing, without resolution during this "
                                  "period) " + (r.get("intended_action") or ""))
            else:
                routines[code] = r.get("routine", "")
        if veto:
            tk.log_skip({"type": "veto", "proposed_hours": hours,
                         "vetoed_by": veto[0], "intent": veto[1],
                         "now_hours": tk.now_hours})
            self.log(f"[timeskip] vetoed by {veto[0]}: {veto[1][:100]}")
            return False
        if overridden:
            self.log(f"[timeskip] vetoes overridden (no material change in "
                     f"grace round): {overridden}")

        passed = tk.advance(hours)
        advanced = "\n".join(
            f"- {e['name']}: occurred {e.get('count', 1)} time(s) ({e['description']})"
            for e in passed) or "- (no scheduled occurrences)"
        for e in passed:
            # an appointment crossed INSIDE the skip (not the landing target)
            # is a beat owed just the same — guard it until it is played
            if e.get("kind") == "appointment" \
                    and getattr(self, "_live_beat", None) is None:
                self._live_beat = {"name": e["name"], "age": 0}
                self.log(f"[beat] appointment '{e['name']}' crossed in skip; "
                         f"guard on")
        # a landed ledger commitment settles by its kind: an external response
        # arrives as a concrete fact; an appointment is about to begin
        kind = event.get("kind")
        if kind == "external_response":
            event_desc += (" [this is an awaited external response: state "
                           "concretely what arrives]")
        elif kind == "appointment":
            event_desc += (" [this is a scheduled appointment: the skip ends "
                           "just as it begins; do not narrate it]")
        characters = "\n".join(
            f"- {c}: at {self.role_agents[c].location_name or 'unknown'}; "
            f"status: {self.role_agents[c].status or '(none)'}; "
            f"routine: {routines.get(c, '(none stated)')}"
            for c in self.role_agents) or "- (none)"
        loc_map = getattr(self.world_agent, "locations_info", {}) or {}
        places = "\n".join(
            f"- {code}: {self.world_agent.find_location_name(code)}"
            for code in loc_map) or "- (none known)"
        # world-agent-as-admin: ground the settlement in canonical standing
        # facts (wages, costs, institutions, norms) via the archivist's
        # leakage-gated channel, so entailed pressures stay canon-consistent
        canon_notes = ""
        if getattr(self.world_agent, "archivist", None):
            try:
                canon_notes = self.world_agent.archivist.consult(
                    f"(settling a quiet period of {hours:.0f} in-world hours; "
                    f"the world must apply the ordinary costs of living — "
                    f"wages, lodging, provisions, obligations) Characters:\n"
                    f"{characters}")
            except Exception as e:
                self.log(f"[timeskip] archivist consult failed: {e}")
        if getattr(self.world_agent, "identity_notes", ""):
            canon_notes = (self.world_agent.identity_notes + "\n"
                           + (canon_notes or ""))
        settle = _jparse(self.world_llm.chat(SETTLE_PROMPT.format(
            span=span_text(hours), situation=self.event, advanced=advanced,
            event_desc=event_desc, characters=characters, places=places,
            canon_notes=canon_notes or "(none)",
            facts=self.world_agent.facts_text(),
            pending=tk.pending_text() or "(none)")))
        # settle locations FIRST so codes coined by this settlement have
        # display names before any narrative text is rendered
        for code, lc in (settle.get("per_character_location") or {}).items():
            if code not in self.role_agents or not isinstance(lc, str) or not lc:
                continue
            if lc not in loc_map:
                # the fiction established a place the registry lacks (oracle
                # knowledge or newly invented) — register it instead of
                # snapping the character to a wrong known code
                self.world_agent.add_location_during_simulation(
                    lc, f"introduced during play around {tk.now_hours:.0f}h")
                self.log(f"[location] registered new location: {lc}")
            self.role_agents[code].set_location(
                lc, self.world_agent.find_location_name(lc))

        def _decode(txt):
            # location codes must never leak into narrative text — render
            # every registered code as its display name (longest-first, or
            # nested codes half-substitute: 廉氏集团DaSha)
            return decode_location_codes(txt, self.world_agent.locations_info)

        narration = _decode(settle.get("narration")) or f"{hours:.0f} quiet hours pass."
        if settle.get("situation_now"):
            self.event_prose = strip_wrapping_quotes(_decode(settle["situation_now"]))
            self.event = self.world_agent.compose_event(self.event_prose)
        for code, st in (settle.get("per_character_status") or {}).items():
            if code in self.role_agents and isinstance(st, str) and st:
                self.role_agents[code].status = _decode(st)
        # experience layer: a skip is an episode boundary — each agent
        # compresses what it just lived through into a first-person note
        # before the quiet period begins
        for code, agent in self.role_agents.items():
            if getattr(agent, "asleep", False):
                continue
            try:
                agent.consolidate_episode(
                    f"随后是约 {hours:.0f} 小时的平静时段" if self.language == "zh"
                    else f"a quiet stretch of about {hours:.0f} hours follows")
            except Exception as e:
                self.log(f"[episode] consolidation failed for {code}: {e}")
        rid = str(uuid.uuid4())
        self.record(role_code="None", detail=narration, actor_type="world",
                    act_type="timeskip", actor="world",
                    group=list(self.role_agents), record_id=rid)
        tk.log_skip({"type": "skip", "hours": hours, "landed": event["name"],
                     "cap": prop.get("cap"),
                     "now_hours": tk.now_hours, "routines": routines,
                     "overridden_vetoes": overridden, "settlement": settle})
        self.log(f"[timeskip] +{hours:.0f}h -> {event['name']} "
                 f"(now {tk.now_hours:.0f}h)")
        if event.get("kind") == "appointment":
            # the skip delivered the cast to the appointment's threshold; the
            # beat itself is still owed — guard it from being skipped over
            # before it is played (otherwise the meeting can vanish unplayed)
            self._live_beat = {"name": event["name"], "age": 0}
            self.log(f"[beat] appointment '{event['name']}' at threshold; "
                     f"skip guard on")
        yield ("world", "", f"[TIME SKIP +{hours:.0f}h] {narration}", rid)
        return True

    # Main functions using llm
    def implement_next_plan(self,role_code: str, group: List[str]):
        other_roles_info = self._get_group_members_info_dict(group)
        plan = self.role_agents[role_code].plan(
            other_roles_info = other_roles_info,
            available_locations = self.world_agent.locations,
            world_description = self.world_agent.description,
            intervention = self.event,
        )
        
        info_text = plan["detail"]
        # resim fork: drop actions that repeat the same character's previous
        # action verbatim (sampler echoes add nothing and pollute the record)
        norm = " ".join(info_text.split())
        if not hasattr(self, "_last_detail_by_role"):
            self._last_detail_by_role = {}
        if norm and self._last_detail_by_role.get(role_code) == norm:
            self.log(f"[dedup] {role_code}: identical consecutive action skipped")
            return ""
        self._last_detail_by_role[role_code] = norm
        if plan["target_role_codes"]:
            plan["target_role_codes"] = self._name2code(plan["target_role_codes"])

        # resim fork: normalize interact_type to the dispatch vocabulary
        # ('role' fans out to single/multi; 'enviroment' tolerates the
        # pre-rename engine token still present in old saves/model echoes)
        itype = str(plan.get("interact_type") or "no").strip().lower()
        targets = plan.get("target_role_codes") or []
        cast_targets = [t for t in targets if t in self.role_agents]
        offcast_targets = [t for t in targets if t not in self.role_agents]
        if itype == "role":
            itype = ("single" if len(cast_targets) == 1
                     else "multi" if len(cast_targets) > 1 else "no")
        elif itype == "enviroment":
            itype = "environment"
        # a plan aimed only at someone outside the cast is an NPC exchange —
        # route it to the world agent instead of silently dropping the target
        if itype in ("no", "single", "multi") and offcast_targets and not cast_targets:
            itype = "npc"
            plan["target_npc_name"] = plan.get("target_npc_name") or offcast_targets[0]
        plan["interact_type"] = itype
        plan["target_role_codes"] = cast_targets

        record_id = str(uuid.uuid4())
        self.log(f"-Action-\n{self.role_agents[role_code].role_name}: "+ info_text)
        self.record(role_code = role_code,
                    detail = plan["detail"],
                    actor_type = 'role',
                    act_type = "plan",
                    actor = role_code,
                    group = plan["target_role_codes"] + [role_code],
                    plan = plan,
                    record_id = record_id
                        )
        yield ("role", role_code, info_text, record_id)

        if plan["interact_type"] == "single" and len(plan["target_role_codes"]) == 1 and plan["target_role_codes"][0] in group:
            yield from self.start_single_role_interaction(plan, record_id)
        elif plan["interact_type"] == "multi" and len(plan["target_role_codes"]) > 1 and set(plan["target_role_codes"]).issubset(set(group))  :
            yield from self.start_multi_role_interaction(plan, record_id)
        elif plan["interact_type"] == "environment":
            yield from self.start_environment_interaction(plan,role_code, record_id)
        elif plan["interact_type"] == "npc" and plan["target_npc_name"]:
            yield from self.start_npc_interaction(plan,role_code,target_name=plan["target_npc_name"], record_id = record_id)
        return info_text         
    
    def decide_whether_to_move(self, 
                          role_code: str, 
                          group: List[str]):
        if len(self.world_agent.locations) <= 1:
            return False
        if_move, move_detail, destination_code = self.role_agents[role_code].move(locations_info_text = self._get_locations_info(),
                                                                                  locations_info = self.world_agent.locations_info)
        # resim fork: resolve an unlisted destination against known names,
        # else register it — the fiction can establish places the pack lacks.
        # CJK-aware matching: the old ASCII-only normalization mapped every
        # zh name to "" (all locations shared one alias key), silently
        # teleporting zh movers to an arbitrary location
        if if_move and destination_code not in self.world_agent.locations_info:
            resolved = resolve_location_alias(destination_code,
                                              self.world_agent.locations_info)
            if resolved is not None:
                destination_code = resolved
            else:
                self.world_agent.add_location_during_simulation(
                    destination_code, "introduced during play")
                self.log(f"[location] registered new location: {destination_code}")
        if if_move:
            self.log(move_detail)
            print(f"角色选择移动。{self.role_agents[role_code].role_name}正在前往{self.world_agent.find_location_name(destination_code)}" if self.language == "zh" else f"The role decides to move. {self.role_agents[role_code].role_name} is heading to {self.world_agent.find_location_name(destination_code)}.")
            self.record(role_code = role_code,
                    detail = move_detail,
                    actor_type = 'role',
                    act_type = "move",
                    actor = role_code,
                    group = [role_code],
                    destination_code = destination_code
                        )
            yield ("role",role_code,move_detail,None)
            
            distance = self.world_agent.get_distance(self.role_agents[role_code].location_code, destination_code)
            self.role_agents[role_code].set_location(location_code=None, location_name=None)
            self.moving_roles_info[role_code] = {
                "location_code":destination_code,
                "distance":distance
            }
        return if_move
          
    def start_environment_interaction(self,
                                     plan: Dict[str, Any], 
                                     role_code: str,
                                     record_id: str):
        """
        Handles the role's interaction with the environment. 
        It gets interaction results from agents in the world, record the result and update the status of the role.

        Args:
            plan (Dict[str, Any]): The details of the action.
            role_code (str): The action maker.

        Returns:
            (str): The environment response.
        """
        if "action" not in plan:
            plan["action"] = ""
        self.current_status['group'] = [role_code]
        location_code = self.role_agents[role_code].location_code
        result = self.world_agent.environment_interact(action_maker_name = self.role_agents[role_code].role_name,
                                                            action  = plan["action"],
                                                            action_detail = conceal_thoughts(self.history_manager.search_record_detail(record_id)),
                                                            location_code = location_code)
        env_record_id = str(uuid.uuid4())
        self.log(f"(Environment): {result}")
        self.record(role_code = role_code,
                    detail = result,
                    actor_type = 'world',
                    act_type = "environment",
                    initiator = role_code,
                    actor = "world",
                    group = [role_code],
                    record_id = env_record_id)
        yield ("world","","(Environment): " + result, env_record_id)
        
        return conceal_thoughts(self.history_manager.search_record_detail(record_id)) + self.history_manager.search_record_detail(env_record_id)
    
    def start_npc_interaction(self,
                              plan: Dict[str, Any], 
                              role_code: str, 
                              target_name: str,
                              record_id: str,
                              max_rounds: int = 3):
        """
        Handles the role's interaction with the environment. 
        It gets interaction results from agents in the world, record the result and update the status of the role.

        Args:
            plan (Dict[str, Any]): The details of the action.
            role_code (str): The action maker.
            target_name (str): The target npc.

        Returns:
            (str): The environment response.
        """
        interaction = plan
        start_idx = len(self.history_manager)

        # resim fork: identity routing — an "npc" target that is actually an
        # appellation of a CAST member is answered by that member's own
        # agent; the world never voices cast characters
        cast_code = self.world_agent.resolve_cast_target(target_name, self.role_agents)
        if cast_code and cast_code != role_code:
            self.log(f"[identity] npc target '{target_name}' is cast member "
                     f"{cast_code}; rerouting to role interaction")
            plan2 = dict(plan)
            plan2["role_code"] = role_code
            plan2["target_role_codes"] = [cast_code]
            yield from self.start_single_role_interaction(plan=plan2,
                                                          record_id=record_id)
            return "\n".join(self.history_manager.get_subsequent_history(start_idx=start_idx))

        # resim fork: resolve the free-text target ("housekeeper or Madam
        # Gerton") to its canonical NPC card name so the speaker label and
        # the played identity stay clean and consistent
        target_name, _ = self.world_agent.resolve_npc(target_name)

        self.log(f"----------NPC Interaction----------\n")
        self.current_status['group'] = [role_code,target_name]
        for round in range(max_rounds):
            # conceal: the NPC-voicing world LLM must not see the acting
            # role's 【inner monologue】 — every other path (env/single/multi)
            # already conceals here
            npc_interaction = self.world_agent.npc_interact(action_maker_name=self.role_agents[role_code].role_name,
                                                    action_detail=conceal_thoughts(self.history_manager.search_record_detail(record_id)),
                                                    location_name=self.role_agents[role_code].location_name,
                                                    target_name=target_name)
            npc_detail = npc_interaction["detail"]
            
            npc_record_id = str(uuid.uuid4())
            self.log(f"{target_name}: " + npc_detail)
            self.record(role_code = role_code,
                        detail = npc_detail,
                        actor_type = 'world',
                        act_type = "npc",
                        actor = "world",
                        group = [role_code],
                        npc_name = target_name,
                        record_id = npc_record_id
                        )
            yield ("world","",f"({target_name}): " + npc_detail, npc_record_id)
            
            if npc_interaction["if_end_interaction"]:
                break
            
            interaction = self.role_agents[role_code].npc_interact(
                npc_name = target_name,
                npc_response = self.history_manager.search_record_detail(npc_record_id),
                history = self.history_manager.get_subsequent_history(start_idx = start_idx),
                intervention = self.event
            )
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.log(f"{self.role_agents[role_code].role_name}: " + detail)
            self.record(role_code = role_code,
                        detail = detail,
                        actor_type = 'role',
                        act_type = "npc",
                        actor = role_code,
                        group = [role_code],
                        npc_name = target_name,
                        record_id = record_id)
            yield ("role",role_code,detail,record_id)
            
            if interaction["if_end_interaction"]:
                break
            if_end,epilogue = self.world_agent.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
                
        return "\n".join(self.history_manager.get_subsequent_history(start_idx = start_idx))
    
    def start_single_role_interaction(self,
                                      plan: Dict[str, Any],
                                      record_id: str,
                                      max_rounds: int = 8):
        max_rounds = getattr(self, 'cascade_cap', max_rounds)
        interaction = plan
        acted_role_code = interaction["role_code"]
        acting_role_code = interaction["target_role_codes"][0]
        if acting_role_code not in self.role_codes:
            print(f"Warning: Role {acting_role_code} does not exist.")
            return
        self.current_status['group'] = [acted_role_code,acting_role_code]
        
        start_idx = len(self.history_manager)
        for round in range(max_rounds):
            interaction = self.role_agents[acting_role_code].single_role_interact(
                action_maker_code = acted_role_code, 
                action_maker_name = self.role_agents[acted_role_code].role_name,
                action_detail = conceal_thoughts(self.history_manager.search_record_detail(record_id)), 
                action_maker_profile = self.role_agents[acted_role_code].role_profile, 
                intervention = self.event
            )
            
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.log(f"{self.role_agents[acting_role_code].role_name}: " + detail)
            self.record(role_code = acting_role_code,
                        detail = detail,
                        actor_type = 'role',
                        act_type = "single",
                        group = [acted_role_code,acting_role_code],
                        target_role_code = acting_role_code,
                        planning_role_code = plan["role_code"],
                        round = round,
                        record_id = record_id
                        )
            yield ("role",acting_role_code,detail,record_id)
            
            if interaction["if_end_interaction"]:
                return
            # the follow-up NPC/env interaction belongs to the responder who
            # chose it (acting_role_code) — same as the multi-role path;
            # executing it as acted_role_code attributed every follow-up to
            # the wrong character
            if interaction["extra_interact_type"] == "npc":
                print("---Extra NPC Interact---")
                result = yield from self.start_npc_interaction(plan=interaction,
                                                               role_code=acting_role_code,
                                                               target_name=interaction["target_npc_name"],
                                                               record_id=record_id)
                interaction["detail"] = result

            elif interaction["extra_interact_type"] == "environment":
                print("---Extra Env Interact---")
                result = yield from self.start_environment_interaction(plan=interaction,role_code=acting_role_code,record_id=record_id)
                interaction["detail"] = result
                
            if_end,epilogue = self.world_agent.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
            acted_role_code,acting_role_code = acting_role_code,acted_role_code
        return
    
    def start_multi_role_interaction(self, 
                                     plan: Dict[str, Any], 
                                     record_id: str,
                                     max_rounds: int = 8):
        max_rounds = getattr(self, 'cascade_cap', max_rounds)

        interaction = plan
        acted_role_code = interaction["role_code"]
        group = interaction["target_role_codes"]
        group.append(acted_role_code)
        
        for code in group:
            if code not in self.role_codes:
                print(f"Warning: Role {code} does not exist.")
                return
        self.current_status['group'] = group
        
        start_idx = len(self.history_manager)
        other_roles_info = self._get_group_members_info_dict(group)
        
        for round in range(max_rounds):
            candidates = remove_list_elements(group, acted_role_code) or group
            acting_role_code = self._name2code(self.world_agent.decide_next_actor(history_text = "\n".join(self.history_manager.get_recent_history(3)),
                                                                  roles_info_text = self._get_group_members_info_text(candidates,status=True)))
            # resim fork: the respondent must be a valid group member other
            # than the actor being responded to — otherwise a character
            # "responds" to itself with a near-duplicate turn
            if acting_role_code not in candidates:
                acting_role_code = candidates[round % len(candidates)]

            interaction = self.role_agents[acting_role_code].multi_role_interact(
                action_maker_code = acted_role_code, 
                action_maker_name = self.role_agents[acted_role_code].role_name,
                action_detail = conceal_thoughts(self.history_manager.search_record_detail(record_id)), 
                action_maker_profile = self.role_agents[acted_role_code].role_profile, 
                other_roles_info = other_roles_info,
                intervention = self.event
            )
            
            detail = interaction["detail"]
            
            record_id = str(uuid.uuid4())
            self.log(f"{self.role_agents[acting_role_code].role_name}: "+ detail)
            self.record(role_code = acting_role_code,
                        detail = detail,
                        actor_type = 'role',
                        act_type = "multi",
                        group = group,
                        actor = acting_role_code,
                        planning_role_code = plan["role_code"],
                        round = round,
                        record_id = record_id
                        )
            yield ("role",acting_role_code,detail,record_id)
            
                
            if interaction["if_end_interaction"]:
                break
            result = ""
            if interaction["extra_interact_type"] == "npc":
                print("---Extra NPC Interact---")
                result = yield from self.start_npc_interaction(plan=interaction,role_code=acting_role_code,target_name=interaction["target_npc_name"],record_id = record_id)
            elif interaction["extra_interact_type"] == "environment":
                print("---Extra Env Interact---")
                result = yield from self.start_environment_interaction(plan=interaction,role_code=acting_role_code,record_id = record_id)
            interaction["detail"] = self.history_manager.search_record_detail(record_id) + result
            acted_role_code = acting_role_code
            if_end,epilogue = self.world_agent.judge_if_ended("\n".join(self.history_manager.get_subsequent_history(start_idx)))
            if if_end:
                break
            # resim fork: removed a stray `return` inherited from upstream that
            # ended every multi-role interaction after a single response

        return

    # Sub functions using llm
    def script_instruct(self, 
                        last_progress: str, 
                        top_k: int = 5):
        """
        Under the script mode, generate instruction for the roles at the beginning of each round.

        Args:
            last_progress (str): Where the script went in the last round.
            top_k (int, optional): The number of action history of each role to refer. Defaults to 1.

        Returns:
            Dict[str, Any]: Instruction for each role.
        """
        roles_info_text = self._get_group_members_info_text(self.role_codes,status=True)
        history_text = self.history_manager.get_recent_history(top_k)
    
        instruction = self.world_agent.get_script_instruction(
                                roles_info_text=roles_info_text, 
                                event = self.event, 
                                history_text=history_text,
                                script=self.script,
                                last_progress = last_progress)
        
        for code in instruction:
            if code == "progress":
                self.log("剧本进度："+ instruction["progress"]) if self.language == "zh" else self.log("Current Stage:"+ instruction["progress"])
            elif code in self.role_codes:
                # self.role_agents[code].update_goal(instruction = instruction[code])
                self.goal_log.append({"round": self.cur_round, "role": code,
                                      "old": self.role_agents[code].goal,
                                      "new": instruction[code],
                                      "via": "intervention"})
                self.role_agents[code].goal = instruction[code]
            else:
                print("Instruction failed, role code:",code)
        return instruction
       
    def get_event(self,):
        if self.intervention == "" and not self.script:
            roles_info_text = self._get_group_members_info_text(self.role_codes,profile=True)
            status_text = self._get_status_text(self.role_codes)
            event = self.world_agent.generate_event(roles_info_text=roles_info_text,event=self.intervention,history_text=status_text)
            self.intervention = event
        elif self.intervention == "" and self.script:
            self.intervention = self.script
        self.event = self.intervention
        return self.intervention
    
    def get_script(self,):
        if self.script == "":
            roles_info_text = self._get_group_members_info_text(self.role_codes,profile=True)
            status = "\n".join([self.role_agents[role_code].status for role_code in self.role_codes])
            script = self.world_agent.generate_script(roles_info_text=roles_info_text,event=self.intervention,history_text=status)
            self.script = script
        return self.script
    
    def update_event(self, group: List[str], top_k: int = 12):
        if self.intervention == "":
            self.event = ""
        else:
            # resim fork: feed the tracker actual recent history (named
            # records), not bare status strings — status-only input made the
            # summary regress to the frozen initial premise every round.
            # The tracker maintains prose + the verbatim fact ledger; agents
            # see the composed view, the tracker is fed prose only.
            history_text = "\n".join(self.history_manager.get_recent_history(top_k))
            prose = self.world_agent.update_event(
                getattr(self, "event_prose", None) or self.event,
                self.intervention, history_text, script = self.script)
            self.event_prose = prose
            self.event = self.world_agent.compose_event(prose)
    
    # other
    def record(self,
               role_code: str, 
               detail: str, 
               actor_type: str,
               act_type: str,
               group: List[str] = [],
               actor: str = "",
               record_id = None,
               **kwargs):
        if act_type == "plan" and "plan" in kwargs:
            detail = f"{self.role_agents[role_code].nickname}: {detail}"
            interact_type = kwargs["plan"]["interact_type"]
            target = ", ".join(kwargs["plan"]["target_role_codes"])
            other_info = f"Interact type: {interact_type}, Target: {target}"
        elif act_type == "move" and "destination_code" in kwargs:
            destination = kwargs["destination_code"]
            other_info = f"Destination: {destination}"
        elif act_type == "single":
            detail = f"{self.role_agents[role_code].nickname}: {detail}"
            target, planning_role, round = kwargs["target_role_code"],kwargs["planning_role_code"],kwargs["round"]
            other_info = f"Target: {target}, Planning Role: {planning_role}, Round: {round}"
        elif act_type == "multi": 
            detail = f"{self.role_agents[role_code].nickname}: {detail}"
            planning_role, round = kwargs["planning_role_code"],kwargs["round"]
            other_info = f"Group member:{group}, Planning Role: {planning_role}, Round:{round},"
        elif act_type == "npc":
            name = kwargs["npc_name"]
            other_info = f"Target: {name}"
        elif act_type == "environment":
            other_info = ""
        else:
            other_info = ""
        # data contract: link the record to the actor call that produced it.
        # _last_call_id is set by the adapter on each SUCCESSFUL call, so for
        # npc/environment records this is the role's decision call (the world
        # narration in between uses a different llm instance). None for world
        # records and after failed calls.
        call_id = None
        if actor_type == "role" and role_code in getattr(self, "role_agents", {}):
            call_id = getattr(self.role_agents[role_code].llm,
                              "_last_call_id", None)
        record = {
            "cur_round":self.cur_round,
            "role_code":role_code,
            "detail":detail,
            "actor":actor,
            "group":group,      # visible group
            "actor_type":actor_type,
            "act_type":act_type,
            "other_info":other_info,
            "record_id":record_id,
            "call_id":call_id
        }
        self.history_manager.add_record(record)
        # resim fork: inner monologue ([...]/【...】) exists only in the
        # author's own memory and the human-facing transcript; interaction
        # partners receive the concealed version, same as bystanders below.
        # Only character-action records are concealed — world records
        # ([WORLD EVENT]/[SCENE BREAK]/[TIME SKIP] tags) carry no monologue
        # and their bracketed tags must survive
        if act_type in ("plan", "single", "multi", "npc", "environment"):
            concealed_record = dict(record,
                                    detail=conceal_thoughts(record["detail"]))
        else:
            concealed_record = record
        for code in group:
            self.role_agents[code].record(record if code == role_code else concealed_record)
        # resim fork: a visible action or outcome is perceived by co-located
        # awake characters even when they are not interaction targets
        # (thoughts concealed) — otherwise bystanders re-execute actions they
        # never saw happen and contradict each other about observable facts
        if act_type in ("plan", "single", "multi", "npc", "environment") \
                and role_code in getattr(self, "role_agents", {}):
            loc = self.role_agents[role_code].location_code
            if loc:
                observers = [c for c in self.role_codes
                             if c not in group and c != role_code
                             and self.role_agents[c].location_code == loc
                             and not getattr(self.role_agents[c], "asleep", False)]
                if observers:
                    obs_record = dict(record, detail=conceal_thoughts(record["detail"]))
                    for c in observers:
                        self.role_agents[c].record(obs_record)
    
    def settle_movement(self,):
        for role_code in self.moving_roles_info.copy():
            if not self.moving_roles_info[role_code]["distance"]:
                location_code = self.moving_roles_info[role_code]["location_code"]
                self.role_agents[role_code].set_location(location_code, self.world_agent.find_location_name(location_code))
                self.log(f"{self.role_agents[role_code].role_name} 已到达 【{self.world_agent.find_location_name(location_code)}】" if self.language == "zh" else
                         f"{self.role_agents[role_code].role_name} has reached [{self.world_agent.find_location_name(location_code)}]")
                del self.moving_roles_info[role_code]
            else:
                self.moving_roles_info[role_code]["distance"] -= 1
    
    def _find_group(self,role_code):
        return [code for code in self.role_codes if self.role_agents[code].location_code==self.role_agents[role_code].location_code]
    
    def _find_roles_at_location(self,location_code,name = False):
        if name:
            return [self.role_agents[code].nickname for code in self.role_codes if self.role_agents[code].location_code==location_code]
        else:
            return [code for code in self.role_codes if self.role_agents[code].location_code==location_code]

    def _get_status_text(self,group):
        # resim fork: status lines must carry the character's name — nameless
        # concatenation made downstream summarizers guess attribution (and
        # guess wrong: pod states migrating between characters, etc.)
        return "\n".join(f"{self.role_agents[role_code].role_name}: {self.role_agents[role_code].status}"
                         for role_code in group)
    
    def _get_group_members_info_text(self,group, profile = False,status = False):
        roles_info_text = ""
        for i, role_code in enumerate(group):
            name = self.role_agents[role_code].role_name
            roles_info_text += f"{i+1}. {name}\n(role_code:{role_code})\n"
            if profile:
                profile =  self.role_agents[role_code].role_profile
                roles_info_text += f"{profile}\n"
            if status:
                status =  self.role_agents[role_code].status
                roles_info_text += f"{status}\n"
        return roles_info_text
    
    def _get_group_members_info_dict(self,group: List[str]):
        info = {
                role_code: {
                    "nickname": self.role_agents[role_code].nickname,
                    "profile": self.role_agents[role_code].role_profile
                }
                for role_code in group
            }
        return info
    
    def _get_locations_info(self,detailed = True):
        location_info_text = "---当前各角色位置---\n" if self.language == "zh" else "---Current Location of Roles---\n"
        if detailed:
            for i,location_code in enumerate(self.world_agent.locations_info):
                location_name = self.world_agent.find_location_name(location_code)
                description = self.world_agent.locations_info[location_code]["description"]
                location_info_text += f"\n{i+1}. {location_name}\nlocation_code:{location_code}\n{description}\n\n"
                role_names = [f"{self.role_agents[code].role_name}({code})" for code in self.role_codes if self.role_agents[code].location_code == location_code]
                role_names = ", ".join(role_names)
                location_info_text += "目前在这里的角色有：" + role_names if self.language == "zh" else "Roles located here: " + role_names
        else:
            for i,location_code in enumerate(self.world_agent.locations_info):
                location_name = self.world_agent.find_location_name(location_code)
                role_names = [f"{self.role_agents[code].role_name}({code})" for code in self.role_codes if self.role_agents[code].location_code == location_code]
                if len(role_names) == 0:continue
                role_names = ", ".join(role_names)
                location_info_text += f"【{location_name}】：" + role_names +"；"
        return location_info_text
    
    def _clean_role_token(self, role):
        # resim fork: models sometimes echo the "role_code:<code>" label from
        # prompts, or wrap the code in quotes/whitespace
        if isinstance(role, str):
            role = role.strip().strip('"\'').replace("role_code:", "").strip()
        return role

    def _name2code(self,roles):
        name_dic = {self.role_agents[code].role_name:code for code in self.role_codes}
        name_dic.update({self.role_agents[code].nickname:code for code in self.role_codes})
        if isinstance(roles, list):
            processed_roles = []
            for role in roles:
                role = self._clean_role_token(role)
                if role in self.role_codes:
                    processed_roles.append(role)
                elif role in name_dic:
                    processed_roles.append(name_dic[role])
                elif "-" in role and role.split("-")[0] in name_dic:
                    processed_roles.append(name_dic[role.split("-")[0]])
                elif role.replace("_","·") in self.role_codes:
                    processed_roles.append(role.replace("_","·"))
                else:
                    processed_roles.append(role)
            return processed_roles
        elif isinstance(roles, str) :
            roles = self._clean_role_token(roles.replace("\n",""))
            if roles in self.role_codes:
                return roles
            elif roles in name_dic:
                return name_dic[roles]
            elif f"{roles}-{self.language}" in self.role_codes:
                return f"{roles}-{self.language}"
            elif "-" in roles and roles.split("-")[0] in name_dic:
                return name_dic[roles.split("-")[0]]
            elif roles.replace("_","·") in self.role_codes:
                return roles.replace("_","·")
        return roles
    
    def log(self,text):
        self.logger.info(text)
        print(text)
    
    def _save_current_simulation(self, 
                                 stage: Literal["location", "goal", "action"], 
                                 current_round: int = 0,
                                 sub_round:int = 0):
        """
        Save the current simulation progress. 

        Args:
            stage (Literal["location", "goal", "action"]): The stage in which the simulation has been carried out
            current_round (int, optional): If the stage is "action", specify the number of rounds that have been completed. Defaults to 0.
        """
        if not self.if_save:
            return
        save_dir = f"./experiment_saves/{self.experiment_name}/{self.role_llm_name}_{self.start_time}"
        create_dir(save_dir)
        location_setted, goal_setted = False,False
        if stage in ["location","goal","action"]:
            location_setted = True
        if stage in ["goal","action"]:
            goal_setted = True
        meta_info = {
                "location_setted":location_setted,
                "goal_setted": goal_setted,
                "round": current_round,   
                "sub_round": sub_round,    
            }

        save_json_file(os.path.join(save_dir, "meta_info.json"), meta_info)
        name = self.experiment_name.split("/")[0]
        save_json_file(os.path.join(save_dir, f"{name}.json"), self.config)
        
        filename = os.path.join(save_dir, f"./server_info.json")
        save_json_file(filename, self.__getstate__() )
        
        self.history_manager.save_to_file(save_dir)
        if hasattr(self, 'role_agents'):
            for role_code in self.role_codes:
                self.role_agents[role_code].save_to_file(save_dir)
            self.world_agent.save_to_file(save_dir)
        
    def continue_simulation_from_file(self, save_dir: str):
        """
        Restore the record of the last simulation.
        
        Args:
            save_dir (str): The path where the last simulation record was saved.

        Returns:
            Dict[str, Any]: The meta information recording the progress of the simulation
        """
        if os.path.exists(save_dir):
            meta_info = load_json_file(os.path.join(save_dir, "./meta_info.json"))
            filename = os.path.join(save_dir, f"./server_info.json")
            states = load_json_file(filename)
            self.__setstate__(states)   
            for role_code in self.role_codes:
                self.role_agents[role_code].load_from_file(save_dir) 
            self.world_agent.load_from_file(save_dir)
            self.history_manager.load_from_file(save_dir)
            
            for record in self.history_manager.detailed_history:
                # "enviroment": archives saved under the old misspelled act_type
                if record.get("act_type") in ("plan", "single", "multi",
                                              "npc", "environment",
                                              "enviroment"):
                    concealed_record = dict(
                        record, detail=conceal_thoughts(record["detail"]))
                else:
                    concealed_record = record
                for code in record["group"]:
                    if code in self.role_codes:
                        self.role_agents[code].record(
                            record if code == record.get("role_code") else concealed_record)
        else:
            meta_info = {
                "location_setted":False,
                "goal_setted": False,
                "round": 0,
                "sub_round": 0,
            }
        return meta_info
    
    def __getstate__(self):
        states = {key: value for key, value in self.__dict__.items() \
            if isinstance(value, (str, int, list, dict, bool, type(None))) \
                and key not in ['role_agents','world_agent','logger']}
        
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)
