from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import unit04_sources as d


class TestD01D02ProjectKB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prov = d.load_json(d.DEFAULT_PROVENANCE)

    def test_uploaded_kb_exists_and_checksum_locked(self):
        self.assertTrue(d.DEFAULT_SOURCE.is_file())
        self.assertEqual(d.sha256_file(d.DEFAULT_SOURCE), d.EXPECTED_SHA256)
        self.assertEqual(d.DEFAULT_SOURCE.stat().st_size, d.EXPECTED_BYTES)

    def test_d01_builds_registry_from_actual_pdf(self):
        e = d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, self.prov)
        d.validate_registry_entry(e)
        self.assertEqual(e["source_id"], "SRC_MOH_VN_361_QD_BYT_GOUT")
        self.assertEqual(e["version"]["decision_number"], "361/QĐ-BYT")
        self.assertEqual(e["version"]["document_year"], 2014)
        self.assertEqual(e["local_asset"]["path"], "data/kb/raw_docs/gout_guideline_1.pdf")
        self.assertEqual(e["local_asset"]["sha256"], d.EXPECTED_SHA256)
        self.assertEqual(e["local_asset"]["page_count"], 6)
        self.assertTrue(e["local_asset"]["file_present_in_this_build"])

    def test_d01_marks_moh_identity_verified(self):
        e = d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, self.prov)
        self.assertEqual(e["publisher"]["name"], "Bộ Y tế Việt Nam")
        self.assertTrue(e["publisher"]["identity_verified"])
        self.assertEqual(e["review"]["source_scope_status"], "approved_for_project_kb")
        self.assertTrue(e["review"]["rag_source_allowed"])

    def test_d02_accepts_as_project_kb(self):
        e = d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, self.prov)
        r = d.review_project_kb_for_scope(e, self.prov)
        self.assertEqual(r["decision"], "accept_as_project_kb_versioned_official_source")
        self.assertEqual(r["scope"]["project_knowledge_base"], "approved")
        self.assertEqual(r["scope"]["rag_retrieval_source"], "approved")
        self.assertTrue(r["source_allowed_for_rag"])

    def test_d02_does_not_claim_2014_is_latest_2026(self):
        e = d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, self.prov)
        r = d.review_project_kb_for_scope(e, self.prov)
        self.assertIs(r["currentness_claimed"], False)
        self.assertEqual(r["clinical_currentness_review_status"], "pending")
        self.assertIn("claiming_every_2014_recommendation_is_latest_in_2026", r["scope"]["not_approved_for"])

    def test_bad_source_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "gout_guideline_1.pdf"
            fake.write_bytes(b"not the knowledge base")
            with self.assertRaises(ValueError):
                d.build_project_kb_registry_entry(fake, self.prov)

    def test_absolute_registry_path_rejected(self):
        e = d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, self.prov)
        e["local_asset"]["path"] = "/tmp/source.pdf"
        with self.assertRaises(ValueError):
            d.validate_registry_entry(e)

    def test_wrong_decision_number_rejected(self):
        prov = dict(self.prov)
        prov["decision_number"] = "WRONG"
        with self.assertRaises(ValueError):
            d.build_project_kb_registry_entry(d.DEFAULT_SOURCE, prov)

    def test_run_outputs_registry_review_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            m = d.run_d01_d02(d.DEFAULT_SOURCE, d.DEFAULT_PROVENANCE, out)
            self.assertTrue(m["d01_completed"])
            self.assertTrue(m["d02_completed_source_scope"])
            self.assertTrue(m["source_allowed_for_rag"])
            self.assertFalse(m["currentness_claimed"])
            self.assertFalse(m["rag_pipeline_enabled"])
            rows = [json.loads(x) for x in (out/"source_registry.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["local_asset"]["sha256"], d.EXPECTED_SHA256)

    def test_run_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            d.run_d01_d02(d.DEFAULT_SOURCE, d.DEFAULT_PROVENANCE, out)
            with self.assertRaises(FileExistsError):
                d.run_d01_d02(d.DEFAULT_SOURCE, d.DEFAULT_PROVENANCE, out)


if __name__ == "__main__":
    unittest.main()
