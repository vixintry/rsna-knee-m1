"""Regression tests for the EXP-007 report weak-label parser."""
import math
import unittest

from scripts.derive_report_weak_labels import classify_report


class WeakLabelParserRegressionTests(unittest.TestCase):
    def check_result(self, report, target, expected_label, expected_status):
        label, status, positive, negative = classify_report(report, target)
        if math.isnan(expected_label):
            self.assertTrue(math.isnan(label), (label, status, positive, negative))
        else:
            self.assertEqual(label, expected_label, (status, positive, negative))
        self.assertEqual(status, expected_status, (label, positive, negative))

    def test_indication_tear_but_findings_deny_tear(self):
        self.check_result(
            "Indication: suspected medial meniscal tear. "
            "Findings: Medial meniscus normal. "
            "Impression: No evidence of medial meniscal tear.",
            "Medial Meniscus", 0.0, "negative",
        )

    def test_remainder_intact_does_not_negate_lateral_tear(self):
        self.check_result(
            "Findings: Tiny tear of the lateral meniscus. "
            "The remainder of the lateral meniscus is intact. "
            "Conclusion: Lateral meniscal tear.",
            "Lateral Meniscus", 1.0, "positive",
        )

    def test_acl_loss_of_normal_signal_with_high_grade_tear(self):
        self.check_result(
            "Findings: ACL: Loss of normal fiber signal and alignment "
            "with suspicious residual thin fiber, high grade tear.",
            "ACL", 1.0, "positive",
        )

    def test_lateral_patellar_facet_oa_is_not_lateral_compartment_oa(self):
        self.check_result(
            "Osteoarthritis of the lateral patellar facet.",
            "Lateral OA", float("nan"), "unknown",
        )

    def test_explicit_lateral_femorotibial_oa_is_positive(self):
        self.check_result(
            "There is osteoarthritis of the lateral femorotibial compartment.",
            "Lateral OA", 1.0, "positive",
        )

    def test_contradictory_synovitis_statements_remain_conflict(self):
        self.check_result(
            "Findings: No hay signos de sinovitis. "
            "Impression: sinovitis difusa.",
            "Synovitis", float("nan"), "conflict",
        )

    def test_rule_out_without_confirmation_is_uncertain(self):
        self.check_result(
            "Impression: Rule out medial meniscal tear.",
            "Medial Meniscus", float("nan"), "uncertain",
        )


if __name__ == "__main__":
    unittest.main()
