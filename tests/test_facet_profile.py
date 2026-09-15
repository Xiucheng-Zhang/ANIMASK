"""Synthetic-fixture tests for persona_probe.facet_profile (A1 facet
presence profile + gate) and persona_probe.leak_audit (t0 possession
audit). No real API calls.

    python3 tests/test_facet_profile.py
    python3 -m pytest tests/test_facet_profile.py -q
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp                                  # noqa: E402
from animask.persona_probe import facet_profile as fp               # noqa: E402
from animask.persona_probe import leak_audit as la                  # noqa: E402

PACK, PACK2 = "fpack", "fpack2"
TAG = "fbatch_r1"
GOOD, BAD, OTHER = "Good-en", "Bad-en", "Other-en"

PSET = [
    {"qid": "I1", "expectation": "none", "atom_id": None,
     "source_role": None, "variants": ["who are you"]},
    {"qid": "QC01", "expectation": "none", "atom_id": "C01",
     "source_role": None, "variants": ["if X then?"]},
    {"qid": "QR01", "expectation": "none", "atom_id": "R01",
     "source_role": None, "variants": ["about other?"]},
    {"qid": "QK01", "expectation": "should_know", "atom_id": "K01",
     "source_role": None, "variants": ["your secret?"]},
    {"qid": "FK01", "expectation": "should_not_know", "atom_id": None,
     "source_role": OTHER, "variants": ["the vault code?"]},
]
CPS = [{"call_index": i, "frac": f}
       for i, f in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])]
FOREIGN_STMT = "Other knows the vault code is 4417."


def score_rec(score, violated=False):
    return {"violations": [{"atom_id": "C01", "severity": 3}]
            if violated else [], "penalty": 0, "score": score}


def build_pack(root: Path, pack: str, roles_scores: dict,
               foreign_stmt: str = FOREIGN_STMT):
    """roles_scores: {role: fn(qid, ci) -> score_rec or None}"""
    rd = root / "results" / "runs" / pack
    rd.mkdir(parents=True, exist_ok=True)
    roles = list(roles_scores)
    per_agent, items = {}, {}
    for role, fn in roles_scores.items():
        per_agent[role] = {"checkpoints": CPS, "probe_set": PSET,
                           "stats": {}, "bfi": []}
        for cp in CPS:
            for q in PSET:
                rec = fn(q["qid"], cp["call_index"])
                if rec is not None:
                    ci = cp["call_index"]
                    items[f"score|{role}|{ci}|{q['qid']}|0"] = rec
                    items[f"ans|{role}|{ci}|{q['qid']}|0"] = "some answer"
    probe = {"pack": pack, "tag": TAG, "roles": roles, "complete": True,
             "l1": {"per_agent": {r: {"stats": {"mean": 0.9},
                                      "feature_means": {"ttr": 0.5}}
                                  for r in roles}},
             "l2": {"per_agent": per_agent}, "l3": {}, "verdict": {}}
    (rd / f"persona_probe_{TAG}.json").write_text(json.dumps(probe))
    (rd / f"persona_probe_{TAG}.l2.json").write_text(
        json.dumps({"items": items, "meta": {}}))
    (rd / f"run_meta_{TAG}.json").write_text(json.dumps(
        {"models": {"actor": "gpt-5.5"}, "args": {"persona_level": "P3"}}))
    cache = root / "data" / "cache" / "persona_probe" / pack
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{OTHER}.atoms.json").write_text(json.dumps(
        {"atoms": [{"id": "K01", "type": "knowledge",
                    "statement": foreign_stmt, "probe_question": "q"}]}))


def good(qid, ci):
    if qid == "FK01":
        return score_rec(85, violated=(ci == 0))
    return score_rec(95)


def bad(qid, ci):
    # traits_values (I1+QC01 mean) below 80 at 3 of 5 checkpoints
    if qid in ("I1", "QC01") and ci in (2, 3, 4):
        return score_rec(60, violated=True)
    if qid == "FK01":
        return score_rec(40, violated=(ci in (1, 2, 3)))
    return score_rec(90)


class TestFacetMapping(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(fp.facet_of(PSET[0]), "traits_values")   # identity
        self.assertEqual(fp.facet_of(PSET[1]), "traits_values")   # C atom
        self.assertEqual(fp.facet_of(PSET[2]), "relations")       # R atom
        self.assertEqual(fp.facet_of(PSET[3]), "knowledge")       # should_know
        self.assertEqual(fp.facet_of(PSET[4]), "knowledge")       # foreign

    def test_nearest_slot(self):
        self.assertEqual(fp.nearest_slot(0.33), 0.25)
        self.assertEqual(fp.nearest_slot(0.9), 1.0)
        self.assertEqual(fp.nearest_slot(0.0), 0.0)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ANIMASK_ROOT"] = self.tmp.name
        self.root = Path(self.tmp.name)
        build_pack(self.root, PACK, {GOOD: good, BAD: bad})
        build_pack(self.root, PACK2, {GOOD: good})

    def tearDown(self):
        os.environ.pop("ANIMASK_ROOT", None)
        self.tmp.cleanup()


class TestFkValidity(FixtureCase):
    def test_valid_sample(self):
        v = fp.fk_validity(PACK, GOOD, PSET[4])
        self.assertTrue(v["valid"])
        self.assertEqual(v["statement"], FOREIGN_STMT)

    def test_self_referential(self):
        build_pack(self.root, "fpack3", {GOOD: good},
                   foreign_stmt="Other knows Good is secretly a robot.")
        v = fp.fk_validity("fpack3", GOOD, PSET[4])
        self.assertFalse(v["valid"])
        self.assertIn("self_referential", v["reason"])

    def test_missing_source_atom(self):
        q = dict(PSET[4], source_role="Nobody-en")
        v = fp.fk_validity(PACK, GOOD, q)
        self.assertFalse(v["valid"])
        self.assertEqual(v["reason"], "source_atom_missing")


class TestProfileAndGate(FixtureCase):
    def test_role_profile_scores(self):
        prof = fp.run_profile(PACK, TAG)
        g = prof["per_role"][GOOD]
        # knowledge facet = QK01 (95) + valid FK01 (85) -> mean 90
        for p in g["facets"]["knowledge"]["series"]:
            self.assertEqual(p["score"], 90.0)
            self.assertEqual(p["n_scored"], 2)
        self.assertEqual(g["facets"]["relations"]["stats"]["mean"], 95.0)
        self.assertEqual(g["facets"]["traits_values"]["stats"]["mean"], 95.0)
        # raw FK violations kept as diagnostic only (1 of 5)
        self.assertEqual(g["fk_violations"]["violated"], 1)
        self.assertEqual(g["fk_violations"]["n"], 5)
        # no audit file -> leak untested, vacuous pass
        self.assertFalse(g["leak"]["tested"])
        self.assertEqual(g["leak"]["reason"], "audit_pending")
        self.assertTrue(g["gate"]["pass"])

    def test_invalid_fk_excluded_from_scores(self):
        build_pack(self.root, "fpack3", {GOOD: good},
                   foreign_stmt="Other knows Good is secretly a robot.")
        prof = fp.run_profile("fpack3", TAG)
        g = prof["per_role"][GOOD]
        # knowledge facet now QK01 alone
        self.assertEqual(g["facets"]["knowledge"]["stats"]["mean"], 95.0)
        self.assertEqual(g["fk_violations"]["n"], 0)
        self.assertEqual(g["fk_violations"]["excluded_invalid"], ["FK01"])
        lk = g["leak"]
        self.assertFalse(lk["tested"])
        self.assertIn("fk_invalid", lk["reason"])
        self.assertTrue(g["gate"]["pass"])

    def test_gate_fail_traits(self):
        prof = fp.run_profile(PACK, TAG)
        b = prof["per_role"][BAD]
        comp = b["gate"]["components"]
        self.assertFalse(comp["traits_values"]["pass"])   # 2/5 < 0.8
        self.assertTrue(comp["relations"]["pass"])
        self.assertFalse(b["gate"]["pass"])

    def test_gate_uses_audit(self):
        audit = {f"{PACK}:{TAG}:{GOOD}": {"leaked_t0": True},
                 f"{PACK}:{TAG}:{BAD}": {"leaked_t0": False}}
        prof = fp.run_profile(PACK, TAG, audit=audit)
        g = prof["per_role"][GOOD]
        self.assertTrue(g["leak"]["tested"])
        self.assertTrue(g["leak"]["leaked_t0"])
        self.assertFalse(g["gate"]["pass"])       # leak flips GOOD to fail
        b = prof["per_role"][BAD]
        self.assertTrue(b["leak"]["tested"])
        self.assertTrue(b["gate"]["components"]["knowledge_leak"]["pass"])

    def test_rollup_and_markdown(self):
        audit = {f"{PACK}:{TAG}:{GOOD}": {"leaked_t0": False}}
        runs = [fp.run_profile(PACK, TAG, audit=audit),
                fp.run_profile(PACK2, TAG)]
        groups = fp.rollup(runs, seed=0)
        self.assertEqual(list(groups), ["gpt-5.5|P3"])
        g = groups["gpt-5.5|P3"]
        self.assertEqual(g["n_role_runs"], 3)
        self.assertEqual(g["gate_pass"], 2)
        self.assertEqual(g["gate_failed_roles"], [f"{PACK}/{BAD}"])
        self.assertEqual(g["leak_t0"]["audited"], 1)
        self.assertEqual(g["leak_t0"]["leaked"], 0)
        self.assertEqual(g["leak_t0"]["untested"], {"audit_pending": 2})
        self.assertEqual(len(g["drift_curve"]["knowledge"]), 5)
        fm = g["facet_means"]["traits_values"]
        self.assertLessEqual(fm["lo"], fm["stat"])
        self.assertLessEqual(fm["stat"], fm["hi"])
        md = fp.to_markdown({"batchtag": "fbatch", "generated_at": "now",
                             "runs": runs, "groups": groups})
        self.assertIn("gpt-5.5|P3", md)
        self.assertIn("**FAIL**", md)

    def test_main_end_to_end(self):
        result = fp.main(["fbatch"])
        self.assertEqual(len(result["runs"]), 2)
        self.assertTrue((pp.aggregate_dir() / "facets_fbatch.json").exists())
        self.assertTrue((pp.aggregate_dir() / "facets_fbatch.md").exists())


class TestLeakAudit(FixtureCase):
    def test_audit_end_to_end(self):
        calls = []

        def fake_many(jobs, max_workers=10):
            calls.extend(jobs)
            return [json.dumps({"possesses": False, "evidence": "no"})
                    for _ in jobs]

        orig = pp.llm_query_many
        pp.llm_query_many = fake_many
        try:
            res = la.main(["fbatch"])
        finally:
            pp.llm_query_many = orig
        # 3 valid role-runs (GOOD+BAD in fpack, GOOD in fpack2) x 1 variant
        self.assertEqual(len(calls), 3)
        for j in calls:
            self.assertIn(FOREIGN_STMT, j["prompt"])
        roles = res["roles"]
        self.assertEqual(len(roles), 3)
        for rec in roles.values():
            self.assertTrue(rec["valid"])
            self.assertFalse(rec["leaked_t0"])
        out = pp.aggregate_dir() / "leak_audit_fbatch.json"
        self.assertTrue(out.exists())
        # facet_profile picks the audit up: leak becomes tested
        prof_res = fp.main(["fbatch"])
        g = prof_res["runs"][0]["per_role"][GOOD]
        self.assertTrue(g["leak"]["tested"])

    def test_judge_legitimately_known_invalidates(self):
        def fake_many(jobs, max_workers=10):
            return [json.dumps({"possesses": True,
                                "legitimately_known": True,
                                "evidence": "fact is about them"})
                    for _ in jobs]

        orig = pp.llm_query_many
        pp.llm_query_many = fake_many
        try:
            res = la.main(["fbatch"])
        finally:
            pp.llm_query_many = orig
        rec = res["roles"][f"{PACK}:{TAG}:{GOOD}"]
        self.assertIsNone(rec["leaked_t0"])
        self.assertEqual(rec["reason"], "judge_legitimately_known")
        # facet_profile: untested, reason surfaced, vacuous pass
        prof_res = fp.main(["fbatch"])
        g = prof_res["runs"][0]["per_role"][GOOD]
        self.assertFalse(g["leak"]["tested"])
        self.assertEqual(g["leak"]["reason"], "judge_legitimately_known")
        self.assertTrue(g["gate"]["components"]["knowledge_leak"]["pass"])

    def test_audit_resume_skips_done(self):
        store = pp.ResumableStore(
            pp.aggregate_dir() / "leak_audit_fbatch.store.json")
        store.put(f"t0|{PACK}:{TAG}:{GOOD}|0",
                  {"possesses": True, "evidence": "said 4417"})
        store.flush()
        seen = []

        def fake_many(jobs, max_workers=10):
            seen.extend(jobs)
            return [json.dumps({"possesses": False, "evidence": "no"})
                    for _ in jobs]

        orig = pp.llm_query_many
        pp.llm_query_many = fake_many
        try:
            res = la.main(["fbatch"])
        finally:
            pp.llm_query_many = orig
        self.assertEqual(len(seen), 2)     # GOOD@fpack already stored
        self.assertTrue(
            res["roles"][f"{PACK}:{TAG}:{GOOD}"]["leaked_t0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
