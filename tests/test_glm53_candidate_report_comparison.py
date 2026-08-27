import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_glm53_candidate_reports import _canonical_sha256, compare_reports


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_report(root: Path, label: str, kld: np.ndarray, top1: np.ndarray) -> Path:
    root.mkdir()
    tokenwise = root / "tokenwise.safetensors"
    save_file(
        {
            "kld_nats": kld.astype(np.float64),
            "kld_bits": (kld / np.log(2)).astype(np.float64),
            "reference_top1": np.array([1, 2, 3, 4], dtype=np.int64),
            "candidate_top1": top1.astype(np.int64),
        },
        tokenwise,
    )
    summary = {
        "candidate_capture": {"run_label": label},
        "reference_capture": {"manifest_sha256": "a" * 64},
        "tokenwise_file": tokenwise.name,
        "tokenwise_file_sha256": _sha256(tokenwise),
        "windows": [
            {"index": 0, "positions": 2, "input_ids_sha256": "b" * 64},
            {"index": 1, "positions": 2, "input_ids_sha256": "c" * 64},
        ],
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


class Glm53CandidateReportComparisonTests(unittest.TestCase):
    def test_direct_paired_deltas_and_top1_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_report(
                root / "baseline",
                "baseline",
                np.array([0.2, 0.4, 0.6, 0.8]),
                np.array([1, 9, 3, 9]),
            )
            candidate = _make_report(
                root / "candidate",
                "candidate",
                np.array([0.1, 0.3, 0.7, 0.9]),
                np.array([9, 2, 3, 9]),
            )

            result = compare_reports(baseline, candidate)

            self.assertAlmostEqual(result["paired"]["mean_kld_delta_nats"], 0.0)
            self.assertEqual(result["paired"]["token_positions_candidate_better"], 2)
            self.assertEqual(result["paired"]["token_positions_candidate_worse"], 2)
            self.assertEqual(result["paired"]["windows_candidate_better"], 1)
            self.assertEqual(result["paired"]["windows_candidate_worse"], 1)
            self.assertEqual(result["paired"]["baseline_only_top1_correct"], 1)
            self.assertEqual(result["paired"]["candidate_only_top1_correct"], 1)
            self.assertEqual(len(result["comparison_sha256"]), 64)

    def test_reference_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_report(
                root / "baseline",
                "baseline",
                np.ones(4),
                np.array([1, 2, 3, 4]),
            )
            candidate = _make_report(
                root / "candidate",
                "candidate",
                np.ones(4),
                np.array([1, 2, 3, 4]),
            )
            path = candidate / "summary.json"
            summary = json.loads(path.read_text())
            summary["reference_capture"]["manifest_sha256"] = "d" * 64
            summary.pop("summary_sha256")
            summary["summary_sha256"] = _canonical_sha256(summary)
            path.write_text(json.dumps(summary, sort_keys=True) + "\n")

            with self.assertRaisesRegex(RuntimeError, "different reference"):
                compare_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
