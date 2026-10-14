"""Regression tests for the experimental EXP-007 parser v2 guards."""
import math
import unittest

from scripts.derive_report_weak_labels_v2 import classify_report_v2


class ParserV2GuardTests(unittest.TestCase):
    def check_result(self, report, target, expected_label, expected_status):
        label, status, positive, negative = classify_report_v2(report, target)
        if math.isnan(expected_label):
            self.assertTrue(math.isnan(label), (label, status, positive, negative))
        else:
            self.assertEqual(label, expected_label, (status, positive, negative))
        self.assertEqual(status, expected_status, (label, positive, negative))

    def test_meniscus_without_surfacing_tear_is_negative(self):
        self.check_result(
            "Findings: Lateral meniscus without surfacing tear.",
            "Lateral Meniscus", 0.0, "negative",
        )

    def test_separate_explicit_meniscus_tear_is_not_overridden(self):
        self.check_result(
            "Lateral meniscus without surfacing tear. "
            "Impression: tear of the lateral meniscus.",
            "Lateral Meniscus", 1.0, "positive",
        )

    def test_low_grade_acl_injury_is_below_threshold(self):
        self.check_result(
            "ACL demonstrates grade 1 injury without discontinuity.",
            "ACL", 0.0, "negative",
        )

    def test_high_grade_acl_tear_remains_positive(self):
        self.check_result(
            "High-grade partial tear of the ACL with fiber discontinuity.",
            "ACL", 1.0, "positive",
        )

    def test_grade_one_mcl_sprain_is_negative(self):
        self.check_result(
            "Grade 1 MCL sprain, likely chronic.",
            "MCL", 0.0, "negative",
        )

    def test_ungraded_mcl_sprain_is_not_assumed_positive(self):
        self.check_result(
            "Mild MCL sprain.",
            "MCL", float("nan"), "unknown",
        )

    def test_trace_effusion_is_negative(self):
        self.check_result(
            "There is a trace joint effusion.",
            "Effusion", 0.0, "negative",
        )

    def test_moderate_effusion_remains_positive(self):
        self.check_result(
            "There is a moderate joint effusion.",
            "Effusion", 1.0, "positive",
        )

    def test_trace_bakers_cyst_is_negative(self):
        self.check_result(
            "Small Baker's cyst.",
            "Baker's", 0.0, "negative",
        )


if __name__ == "__main__":
    unittest.main()
