import unittest

from src.report_labels import extract_acl_candidate


class TestExtractAclCandidate(unittest.TestCase):
    def test_explicit_tear_is_positive(self):
        result = extract_acl_candidate("The ACL demonstrates a partial tear.")
        self.assertEqual(result.status, "positive")

    def test_intact_acl_is_negative(self):
        result = extract_acl_candidate("The ACL is intact.")
        self.assertEqual(result.status, "negative")

    def test_no_acl_tear_is_negative(self):
        result = extract_acl_candidate("No evidence of ACL tear.")
        self.assertEqual(result.status, "negative")

    def test_uncertain_language_stays_uncertain(self):
        result = extract_acl_candidate("Possible partial tear of the ACL.")
        self.assertEqual(result.status, "uncertain")

    def test_historical_mention_stays_uncertain(self):
        result = extract_acl_candidate("History of ACL tear.")
        self.assertEqual(result.status, "uncertain")

    def test_missing_or_unmentioned_is_not_mentioned(self):
        self.assertEqual(extract_acl_candidate(None).status, "not_mentioned")
        self.assertEqual(extract_acl_candidate("Menisci are intact.").status, "not_mentioned")

    def test_conflicting_mentions_stay_uncertain(self):
        result = extract_acl_candidate("The ACL is intact. Elsewhere, ACL tear is described.")
        self.assertEqual(result.status, "uncertain")

    def test_report_text_is_preserved_as_evidence(self):
        result = extract_acl_candidate("The ACL demonstrates a partial tear.")
        self.assertIn("partial tear", result.evidence)


if __name__ == "__main__":
    unittest.main()
