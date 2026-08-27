import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from token_panel import build_token_panel_batches, load_token_panel_records


class TokenPanelTests(unittest.TestCase):
    def _write_panel(self, root: Path) -> Path:
        arrays = root / "arrays"
        arrays.mkdir()
        windows = []
        for role, index in (("fit", 0), ("fit", 1), ("selection", 0)):
            window_id = f"{role}-{index:04d}"
            token_path = arrays / f"{window_id}.tokens.npy"
            np.save(token_path, np.arange(8, dtype=np.int32) + index, allow_pickle=False)
            windows.append(
                {
                    "window_id": window_id,
                    "role": role,
                    "prediction_positions": 7,
                    "token_ids_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
                }
            )
        panel_path = root / "panel.json"
        panel_path.write_text(
            json.dumps(
                {
                    "schema": "quant-pipeline.glm53-token-panel.v1",
                    "sealed_corpus_sha256": "c" * 64,
                    "windows": windows,
                }
            )
        )
        return panel_path

    def test_role_separation_and_exact_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            panel_path = self._write_panel(Path(temporary))
            records = load_token_panel_records(panel_path, role="fit")
            self.assertEqual(len(records), 2)
            batches = build_token_panel_batches(panel_path, role="fit", batch_size=2)
            self.assertEqual(len(batches), 1)
            self.assertEqual(tuple(batches[0]["input_ids"].shape), (2, 8))
            torch.testing.assert_close(
                batches[0]["position_ids"],
                torch.arange(8).unsqueeze(0).expand(2, -1),
            )
            self.assertTrue(bool(batches[0]["attention_mask"].all()))

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_path = self._write_panel(root)
            with (root / "arrays" / "fit-0000.tokens.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_token_panel_records(panel_path, role="fit")


if __name__ == "__main__":
    unittest.main()
