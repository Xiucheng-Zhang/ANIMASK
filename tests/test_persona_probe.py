"""Synthetic-fixture tests for the persona_probe plugin (no real API calls).

Run either way:
    python3 tests/test_persona_probe.py
    python3 -m pytest tests/test_persona_probe.py -q

Covers: score pipeline (L1/L2/L3), resume /
re-entry, language routing, verdict rules, call-log degradation, dry-run
planning, and the pure-Python statistics.
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp                                  # noqa: E402
from animask.persona_probe import (atomize, l1_stream, l2_checkpoint,  # noqa: E402
                           l3_ablation, stats)
from animask.persona_probe import run_persona_probe as runner       # noqa: E402

PACK = "testpack"
TAG = "t1"
HERO, VILLAIN = "Hero-en", "Villain-en"
HERO_PROFILE = "HERO_PROFILE_BLOCK: Hero is brave and blunt."
VILLAIN_PROFILE = "VILLAIN_PROFILE_BLOCK: Villain is scheming and polite."

ATOMS_HERO = [
    {"type": "trait", "statement": "Hero is brave.", "probe_question": None},
    {"type": "conditional",
     "statement": "When challenged, Hero answers bluntly.",
     "probe_question": "Someone mocks you to your face. What do you do?"},
    {"type": "relation", "statement": "Hero distrusts Villain.",
     "probe_question": "What do you make of Villain?"},
    {"type": "knowledge", "statement": "Hero knows the secret passage.",
     "probe_question": "Where is the secret passage?"},
]
ATOMS_VILLAIN = [
    {"type": "trait", "statement": "Villain is polite.",
     "probe_question": None},
    {"type": "conditional",
     "statement": "When cornered, Villain bargains.",
     "probe_question": "You are cornered. What do you do?"},
    {"type": "relation", "statement": "Villain envies Hero.",
     "probe_question": "What do you make of Hero?"},
    {"type": "knowledge", "statement": "Villain knows the poison recipe.",
     "probe_question": "What is in the poison?"},
]


def build_fixture(root: Path, lang="en", with_call_log=True):
    run = root / "results" / "runs" / PACK
    run.mkdir(parents=True)
    transcript = [{"actor_type": "system", "code": "", "text": "-- start --"}]
    for i in range(6):
        text = f"Hero line {i}." + (" BREAK" if i == 4 else "")
        transcript.append({"actor_type": "role", "code": HERO, "text": text})
    for i in range(3):
        transcript.append({"actor_type": "role", "code": VILLAIN,
                           "text": f"Villain line {i}."})
    transcript.append({"actor_type": "world", "code": "", "text": "epilogue"})
    (run / f"transcript_{TAG}.json").write_text(
        json.dumps(transcript, ensure_ascii=False))

    if with_call_log:
        lines = []

        def rec(tag, content, response):
            lines.append(json.dumps(
                {"ts": 0, "model": "gpt-5.6-sol", "tag": tag,
                 "temperature": 0.7, "attempt": 0, "latency_s": 1.0,
                 "messages": [{"role": "user", "content": content}],
                 "response": response}, ensure_ascii=False))

        rec(f"role:{HERO}", "goal prompt",
            json.dumps({"if_change_goal": False}))  # non-acting, filtered
        for i in range(5):
            rec(f"role:{HERO}",
                f"Scene context {i}\n## Your profile\n{HERO_PROFILE}\n## Go",
                json.dumps({"action": "speak",
                            "detail": f"PSTYLE hero act {i}"}))
        for i in range(4):
            rec(f"role:{VILLAIN}",
                f"Scene context {i}\n## Your profile\n{VILLAIN_PROFILE}\n## Go",
                json.dumps({"action": "speak",
                            "detail": f"PSTYLE villain act {i}"}))
        (run / f"llm_calls_{TAG}.jsonl").write_text("\n".join(lines) + "\n")

    for role, profile, other in ((HERO, HERO_PROFILE, VILLAIN),
                                 (VILLAIN, VILLAIN_PROFILE, HERO)):
        d = root / "engine" / "data" / "roles" / PACK / role
        d.mkdir(parents=True)
        (d / "role_info.json").write_text(json.dumps({
            "role_code": role, "role_name": role.split("-")[0],
            "profile": profile,
            "relation": {other: {"relation": ["rival"], "detail": "wary"}},
            "motivation": "Win."}, ensure_ascii=False))
    presets = root / "engine" / "presets"
    presets.mkdir(parents=True)
    (presets / f"{PACK}.json").write_text(json.dumps(
        {"language": lang, "role_agent_codes": [HERO, VILLAIN]}))


class FakeLLM:
    """Marker-dispatched fake backend; counts every call."""

    def __init__(self):
        self.n_query = 0
        self.n_chat = 0
        self.prompts = []

    # ---- single-prompt route (atomize / judges) ----
    def query(self, prompt, model=None, stdin=None):
        self.n_query += 1
        self.prompts.append(prompt)
        if "ATOMIC persona entries" in prompt or "「原子」" in prompt:
            atoms = ATOMS_HERO if ("name: Hero" in prompt
                                   or "姓名:Hero" in prompt) \
                else ATOMS_VILLAIN
            return json.dumps(atoms, ensure_ascii=False)
        if "NLI judge" in prompt or "NLI 裁判" in prompt:
            atom_ids = re.findall(r"^([TCRKL]\d{2}) \[", prompt, re.M)
            utts = re.findall(r"^#(\d+): (.*)$", prompt, re.M)
            out = {}
            for n, text in utts:
                if "BREAK" in text:
                    out[n] = {a: "contradict" for a in atom_ids}
                else:
                    out[n] = {a: ("entail" if i == 0 else "neutral")
                              for i, a in enumerate(atom_ids)}
            return json.dumps(out)
        if "Rewrite each numbered question" in prompt or "改写" in prompt:
            qs = re.findall(r"^(\S+): (.*)$", prompt, re.M)
            return json.dumps({qid: [text + " v2"] for qid, text in qs},
                              ensure_ascii=False)
        if "penalty rubric" in prompt or "扣分制" in prompt:
            if "VIOLATE" in prompt:
                return json.dumps({"violations": [
                    {"atom_id": "T01", "severity": 4, "evidence": "x"}]})
            return json.dumps({"violations": [], "comment": "clean"})
        if "Baseline answer" in prompt or "基线回答" in prompt:
            return json.dumps({"consistent": True, "evidence": "same"})
        if "REFERENCE" in prompt or "参照" in prompt:
            ia = prompt.find("Candidate A")
            ib = prompt.find("Candidate B")
            if ia < 0:
                ia, ib = prompt.find("候选 A"), prompt.find("候选 B")
            a_seg = prompt[ia:ib]
            return json.dumps(
                {"choice": "A" if "PSTYLE" in a_seg else "B", "reason": "r"})
        raise AssertionError("unrecognized prompt: " + prompt[:120])

    def query_many(self, jobs, max_workers=10):
        return [self.query(**j) for j in jobs]

    # ---- messages route (context-fork probes / replays) ----
    def chat(self, messages, model=None, temperature=0.0, **kw):
        self.n_chat += 1
        last = messages[-1]["content"]
        if "questionnaire" in last or "问卷" in last:
            return json.dumps({str(i): 3 for i in range(1, 11)})
        if "consistency check" in last or "一致性检查" in last:
            return "In character, briefly: my answer."
        return json.dumps({"action": "speak", "detail": "DSTYLE default act"})

    def chat_many(self, jobs, max_workers=10):
        return [self.chat(**j) for j in jobs]

    @property
    def n_calls(self):
        return self.n_query + self.n_chat


class RaisingLLM:
    def _boom(self, *a, **kw):
        raise AssertionError("dry-run must not call the LLM backend")
    query = query_many = chat = chat_many = _boom


class Base(unittest.TestCase):
    lang = "en"
    with_call_log = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_fixture(self.root, lang=self.lang,
                      with_call_log=self.with_call_log)
        os.environ["ANIMASK_ROOT"] = str(self.root)
        self._orig = (pp.llm_query, pp.llm_query_many, pp.llm_chat,
                      pp.llm_chat_many)
        self.fake = FakeLLM()
        self.patch(self.fake)

    def patch(self, fake):
        pp.llm_query = fake.query
        pp.llm_query_many = fake.query_many
        pp.llm_chat = fake.chat
        pp.llm_chat_many = fake.chat_many

    def tearDown(self):
        (pp.llm_query, pp.llm_query_many, pp.llm_chat,
         pp.llm_chat_many) = self._orig
        os.environ.pop("ANIMASK_ROOT", None)
        self._tmp.cleanup()

    def atoms(self):
        return atomize.atomize_pack(PACK, [HERO, VILLAIN], self.lang)


class TestStats(unittest.TestCase):
    def test_theil_sen(self):
        self.assertAlmostEqual(
            stats.theil_sen([0, 1, 2, 3], [0, 2, 4, 6]), 2.0)
        self.assertIsNone(stats.theil_sen([1, 1], [0, 5]))

    def test_mann_kendall(self):
        up = stats.mann_kendall(list(range(10)))
        self.assertLess(up["p"], 0.01)
        self.assertGreater(up["s"], 0)
        flat = stats.mann_kendall([1, 1, 1, 1, 1])
        self.assertEqual(flat["p"], 1.0)
        self.assertEqual(stats.mann_kendall([1, 2])["p"], 1.0)  # n<3

    def test_auc(self):
        self.assertAlmostEqual(stats.trajectory_auc([0, 1], [1, 1]), 1.0)
        self.assertAlmostEqual(stats.trajectory_auc([0, 1], [0, 1]), 0.5)
        self.assertEqual(stats.trajectory_auc([3], [7]), 7)

    def test_kappa(self):
        self.assertEqual(stats.cohens_kappa(list("aabb"), list("aabb")), 1.0)
        k = stats.cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
        self.assertAlmostEqual(k, 0.0)
        self.assertEqual(stats.cohens_kappa([True] * 3, [True] * 3), 1.0)

    def test_holm(self):
        adj = stats.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertLessEqual(adj[0], adj[2])
        self.assertLessEqual(adj[2], adj[1])

    def test_cohens_d_and_bootstrap(self):
        self.assertGreater(stats.cohens_d([5, 6, 7], [1, 2, 3]), 2)
        ci = stats.bootstrap_ci_clustered([[1, 2], [3, 4]], seed=1)
        self.assertLessEqual(ci["lo"], ci["stat"])
        self.assertLessEqual(ci["stat"], ci["hi"])

    def test_paired_permutation(self):
        res = stats.paired_permutation([5, 6, 7, 8, 9, 10],
                                       [1, 1, 2, 2, 3, 3], seed=1)
        self.assertLess(res["p"], 0.1)
        self.assertGreater(res["mean_diff"], 0)


class TestAtomize(Base):
    def test_cache_and_ids(self):
        atoms = self.atoms()
        self.assertEqual(self.fake.n_query, 2)
        self.assertTrue(atomize.atom_cache_path(PACK, HERO).exists())
        ids = [a["id"] for a in atoms[HERO]]
        self.assertEqual(ids, ["T01", "C01", "R01", "K01"])
        trait = next(a for a in atoms[HERO] if a["type"] == "trait")
        self.assertIsNone(trait["probe_question"])
        # cache hit -> no further calls
        self.atoms()
        self.assertEqual(self.fake.n_query, 2)


class TestL1(Base):
    def test_scores_and_calibration(self):
        atoms = self.atoms()
        s = l1_stream.run_l1(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        hero = s["per_agent"][HERO]
        self.assertEqual(hero["n_utterances"], 6)
        scores = [row["score"] for row in hero["series"]]
        self.assertEqual(scores, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        broke = hero["series"][4]
        self.assertEqual(broke["n_contradict"], 4)
        self.assertEqual(broke["n_applicable"], 4)
        self.assertGreater(hero["series"][0]["features"]["n_chars"], 0)
        self.assertEqual(s["calibration"]["percent_agreement"], 1.0)
        self.assertIsNotNone(hero["stats"]["auc"])

    def test_resume_no_new_calls(self):
        atoms = self.atoms()
        s1 = l1_stream.run_l1(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        fresh = FakeLLM()
        self.patch(fresh)
        s2 = l1_stream.run_l1(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        self.assertEqual(fresh.n_calls, 0)
        self.assertEqual(
            [r["score"] for r in s1["per_agent"][HERO]["series"]],
            [r["score"] for r in s2["per_agent"][HERO]["series"]])


class TestL2(Base):
    def test_scores_columns_and_agreement(self):
        atoms = self.atoms()
        s = l2_checkpoint.run_l2(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        hero = s["per_agent"][HERO]
        self.assertEqual(len(hero["checkpoints"]), 5)  # 5 acting calls
        for cp in hero["checkpoints"]:
            self.assertEqual(cp["normative_score"], 100.0)
            if cp["is_t0"]:
                self.assertEqual(cp["self_score"], 100.0)
            else:
                self.assertEqual(cp["self_score"], 100.0)
        self.assertEqual(len(hero["probe_set"]), 6)  # I1 I2 R K F C
        fq = next(q for q in hero["probe_set"] if q["qid"].startswith("F"))
        self.assertEqual(fq["expectation"], "should_not_know")
        self.assertEqual(fq["source_role"], VILLAIN)
        for q in hero["probe_set"]:
            self.assertEqual(len(q["variants"]), 2)  # original + paraphrase
        self.assertEqual(s["judge_agreement"]["binary_agreement"], 1.0)
        self.assertEqual(len(hero["bfi"]), 3)
        for row in hero["bfi"]:
            self.assertEqual(row["dims"]["E"], 3.0)

    def test_resume_and_partial_reentry(self):
        atoms = self.atoms()
        l2_checkpoint.run_l2(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        # full re-entry: zero new calls, identical summary
        fresh = FakeLLM()
        self.patch(fresh)
        l2_checkpoint.run_l2(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        self.assertEqual(fresh.n_calls, 0)
        # partial re-entry: delete 3 stored scores -> exactly 3 judge calls
        layer = pp.layer_path(PACK, TAG, "l2")
        data = json.loads(layer.read_text())
        dropped = [k for k in list(data["items"])
                   if k.startswith("score|")][:3]
        for k in dropped:
            del data["items"][k]
        layer.write_text(json.dumps(data))
        fresh2 = FakeLLM()
        self.patch(fresh2)
        l2_checkpoint.run_l2(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        self.assertEqual(fresh2.n_chat, 0)
        self.assertEqual(fresh2.n_query, 3)

    def test_call_log_missing_degrades(self):
        pp.call_log_path(PACK, TAG).unlink()
        atoms = self.atoms()
        s = l2_checkpoint.run_l2(PACK, TAG, [HERO], atoms, "en")
        self.assertEqual(s, {"skipped": "call_log_missing"})
        p = l2_checkpoint.plan_l2(PACK, TAG, [HERO], atoms)
        self.assertEqual(p, {"skipped": "call_log_missing"})


class TestL3(Base):
    def test_two_arms_and_2afc(self):
        atoms = self.atoms()
        l2_checkpoint.run_l2(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        s = l3_ablation.run_l3(PACK, TAG, [HERO, VILLAIN], atoms, "en")
        hero = s["per_agent"][HERO]
        self.assertEqual([c["call_index"] for c in hero["checkpoints"]],
                         [2, 3])
        self.assertEqual(hero["persona_rate"], 1.0)   # 2AFC under both orders
        self.assertEqual(hero["votes"], 4)
        for cp in hero["checkpoints"]:
            self.assertEqual(cp["ablated_probe_score"], 100.0)
            self.assertEqual(cp["persona_probe_score"], 100.0)
            self.assertEqual(cp["drift_gap"], 0.0)
            self.assertTrue(cp["order_consistent"])
        self.assertTrue(s["l2_available_for_gap"])
        # profile block was really ablated in the replayed context
        layer = json.loads(pp.layer_path(PACK, TAG, "l3").read_text())
        self.assertTrue(any(k.startswith("act|") for k in layer["items"]))

    def test_ablation_replaces_profile(self):
        msgs = [{"role": "user", "content": f"x {HERO_PROFILE} y"}]
        out, found = l3_ablation._ablate(msgs, HERO_PROFILE, "en")
        self.assertTrue(found)
        self.assertNotIn("HERO_PROFILE_BLOCK", out[0]["content"])
        self.assertIn("You are a character in this story.",
                      out[0]["content"])
        out2, found2 = l3_ablation._ablate(msgs, "NOT_PRESENT", "en")
        self.assertFalse(found2)


class TestLanguageRouting(Base):
    lang = "zh"

    def test_zh_prompts_selected(self):
        self.assertEqual(pp.resolve_lang(PACK, "auto"), "zh")
        self.assertEqual(pp.resolve_lang(PACK, "en"), "en")  # CLI override
        atoms = self.atoms()
        self.assertTrue(any("「原子」" in p for p in self.fake.prompts))
        l1_stream.run_l1(PACK, TAG, [HERO], atoms, "zh")
        self.assertTrue(any("NLI 裁判" in p for p in self.fake.prompts))
        s = l2_checkpoint.run_l2(PACK, TAG, [HERO], atoms, "zh",
                                 n_variants=1, bfi=False)
        self.assertTrue(any("扣分制" in p for p in self.fake.prompts))
        self.assertEqual(
            s["per_agent"][HERO]["checkpoints"][0]["normative_score"], 100.0)


class TestVerdictRules(unittest.TestCase):
    @staticmethod
    def _l1(mean=0.95, slope=0.0, total=0.0, p=0.9, n=10, role="X"):
        return {"per_agent": {role: {"stats": {
            "mean": mean, "theil_sen_slope": slope, "total_change": total,
            "mk": {"p": p}, "n": n}}}}

    @staticmethod
    def _l3(rate, votes, role="X", gap=20.0):
        return {"per_agent": {role: {"persona_rate": rate, "votes": votes,
                                     "mean_drift_gap": gap}}}

    def test_stable_plus_distinguishable(self):
        v = runner.build_verdict(self._l1(), {"skipped": "x"},
                                 self._l3(1.0, 6), [])
        self.assertEqual(v["per_agent"]["X"]["verdict"],
                         "persona_stable_model_trend")
        self.assertEqual(v["run_verdict"], "persona_stable_model_trend")

    def test_declining_is_drift(self):
        v = runner.build_verdict(
            self._l1(slope=-0.05, total=-0.4, p=0.001),
            {"skipped": "x"}, self._l3(1.0, 8), [])
        self.assertEqual(v["run_verdict"], "drift_associated")

    def test_indistinguishable_is_inert(self):
        v = runner.build_verdict(self._l1(), {"skipped": "x"},
                                 self._l3(0.5, 8, gap=3.0), [])
        self.assertEqual(v["per_agent"]["X"]["verdict"], "persona_inert")
        self.assertIn("X", v["inert_agents"])

    def test_no_l3_is_pending(self):
        v = runner.build_verdict(self._l1(), {"skipped": "x"},
                                 {"skipped": "call_log_missing"},
                                 ["call_log_missing"])
        self.assertEqual(v["per_agent"]["X"]["verdict"],
                         "persona_stable_attribution_pending")
        self.assertIn("call_log_missing", v["degradations"])

    def test_no_data(self):
        v = runner.build_verdict({"skipped": "a"}, {"skipped": "b"},
                                 {"skipped": "c"}, [])
        self.assertEqual(v["run_verdict"], "insufficient_evidence")


class TestRunnerEndToEnd(Base):
    def test_full_pipeline(self):
        res = runner.main([PACK, TAG, "--layers", "l1,l2,l3",
                           "--l3-checkpoints", "4"])
        out = pp.probe_out_path(PACK, TAG)
        self.assertTrue(out.exists())
        on_disk = json.loads(out.read_text())
        self.assertEqual(on_disk["verdict"]["run_verdict"],
                         res["verdict"]["run_verdict"])
        pa = res["verdict"]["per_agent"]
        self.assertEqual(pa[HERO]["verdict"], "persona_stable_model_trend")
        # Villain has only 2 L3 votes (< min) -> attribution pending
        self.assertEqual(pa[VILLAIN]["verdict"],
                         "persona_stable_attribution_pending")
        self.assertIn("l1_mean_score_ci", res["stats_summary"])
        self.assertIn("l3_arm_gap_permutation", res["stats_summary"])
        self.assertEqual(res["atoms"][HERO]["n_atoms"], 4)

    def test_degraded_pipeline_without_call_log(self):
        pp.call_log_path(PACK, TAG).unlink()
        res = runner.main([PACK, TAG, "--layers", "l1,l2,l3"])
        self.assertEqual(res["l2"], {"skipped": "call_log_missing"})
        self.assertEqual(res["l3"], {"skipped": "call_log_missing"})
        self.assertIn("call_log_missing", res["verdict"]["degradations"])
        self.assertEqual(res["verdict"]["per_agent"][HERO]["verdict"],
                         "persona_stable_attribution_pending")

    def test_dry_run_makes_no_calls(self):
        self.patch(RaisingLLM())
        plan = runner.main([PACK, TAG, "--dry-run"])
        self.assertEqual(plan["atomize"]["n_calls"], 2)
        self.assertEqual(plan["l1"]["judge_batches"], 2)  # 6+3 utts, batch 8
        self.assertEqual(plan["l1"]["per_agent_utterances"][HERO], 6)
        self.assertEqual(plan["l2"]["per_agent"][HERO]["acting_calls"], 5)
        self.assertGreater(plan["total_planned_calls"], 0)
        self.assertFalse(pp.probe_out_path(PACK, TAG).exists())

    def test_aggregate(self):
        from animask.persona_probe import aggregate
        runner.main([PACK, TAG, "--layers", "l1,l2,l3",
                     "--l3-checkpoints", "4"])
        agg = aggregate.main(["batchA", "--runs", f"{PACK}:{TAG}"])
        node = agg["packs"][PACK]["per_agent"][HERO]
        self.assertEqual(node["n_runs"], 1)
        self.assertEqual(node["l3_persona_rate"], 1.0)
        self.assertEqual(node["aggregate_verdict"],
                         "persona_stable_model_trend")
        md = pp.aggregate_dir() / "aggregate_batchA.md"
        self.assertTrue(md.exists())
        self.assertIn(HERO, md.read_text())


class TestStoreAndHelpers(unittest.TestCase):
    def test_resumable_store_survives_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            path.write_text("{not json")
            st = pp.ResumableStore(path, flush_every=2)
            st.put("a", 1)
            # below flush threshold: nothing written yet
            self.assertEqual(path.read_text(), "{not json")
            st.put("b", 2)  # second put crosses flush_every -> on disk
            data = json.loads(path.read_text())
            self.assertEqual(data["items"], {"a": 1, "b": 2})
            st2 = pp.ResumableStore(path)
            self.assertTrue(st2.has("a"))

    def test_quantile_indices_and_subsample(self):
        self.assertEqual(pp.quantile_indices(5, 5), [0, 1, 2, 3, 4])
        self.assertEqual(pp.quantile_indices(40, 5), [0, 10, 20, 29, 39])
        self.assertEqual(pp.quantile_indices(1, 5), [0])
        utts = [{"n": i} for i in range(10)]
        sub = l1_stream.subsample(utts, 4)
        self.assertEqual(len(sub), 4)
        self.assertEqual(sub[0]["n"], 0)
        self.assertEqual(sub[-1]["n"], 9)

    def test_seeded_sample_deterministic(self):
        a = pp.seeded_sample(list(range(100)), 0.1, 7)
        b = pp.seeded_sample(list(range(100)), 0.1, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)

    def test_parse_json_reply(self):
        self.assertEqual(pp.parse_json_reply('{"a": 1}'), {"a": 1})
        self.assertEqual(pp.parse_json_reply('```json\n{"a": 1}\n```'),
                         {"a": 1})
        self.assertEqual(pp.parse_json_reply('noise {"a": 1} noise'),
                         {"a": 1})
        with self.assertRaises(ValueError):
            pp.parse_json_reply("<<ERROR: boom>>")

    def test_acting_call_detection(self):
        act = {"tag": "role:X", "response":
               json.dumps({"action": "speak", "detail": "hi"})}
        goal = {"tag": "role:X", "response":
                json.dumps({"if_change_goal": False})}
        world = {"tag": "world", "response":
                 json.dumps({"detail": "scene"})}
        self.assertEqual(pp.acting_calls([act, goal, world], "X"), [act])
        self.assertEqual(pp.extract_detail(act["response"]), "hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
