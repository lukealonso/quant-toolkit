"""Load sealed pre-tokenized calibration panels without re-tokenization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_panel_records(
    panel_path: str | Path,
    *,
    role: str,
    limit: int | None = None,
    max_len: int | None = None,
) -> list[tuple[dict, torch.Tensor]]:
    """Return hash-verified token rows for one sealed panel role."""
    panel_path = Path(panel_path).resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise RuntimeError(f"unexpected token-panel schema: {panel.get('schema')}")
    selected = [record for record in panel.get("windows", []) if record.get("role") == role]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError(f"token panel contains no {role!r} windows")

    records = []
    for record in selected:
        token_path = panel_path.parent / "arrays" / f"{record['window_id']}.tokens.npy"
        if sha256_file(token_path) != record["token_ids_sha256"]:
            raise RuntimeError(f"token file hash mismatch: {token_path}")
        array = np.load(token_path, allow_pickle=False)
        if array.ndim != 1 or array.dtype.kind not in "iu":
            raise RuntimeError(
                f"token IDs must be a one-dimensional integer array: {token_path}"
            )
        expected_tokens = int(record["prediction_positions"]) + 1
        if array.shape[0] != expected_tokens:
            raise RuntimeError(f"token count mismatch: {token_path}")
        if max_len is not None and array.shape[0] > max_len:
            raise RuntimeError(
                f"sealed panel row {record['window_id']} has {array.shape[0]} tokens, "
                f"exceeding max_len={max_len}; truncation is forbidden"
            )
        records.append(
            (record, torch.from_numpy(array.astype(np.int64, copy=False)).contiguous())
        )
    return records


def build_token_panel_batches(
    panel_path: str | Path,
    *,
    role: str,
    batch_size: int,
    limit: int | None = None,
    max_len: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Build exact, unpadded model inputs from equally sized sealed windows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = load_token_panel_records(
        panel_path,
        role=role,
        limit=limit,
        max_len=max_len,
    )
    batches = []
    for offset in range(0, len(records), batch_size):
        chunk = records[offset : offset + batch_size]
        lengths = {tokens.numel() for _, tokens in chunk}
        if len(lengths) != 1:
            raise RuntimeError(
                "sealed token-panel batching requires equal-length windows; "
                f"got lengths={sorted(lengths)}"
            )
        input_ids = torch.stack([tokens for _, tokens in chunk])
        seq_len = input_ids.shape[1]
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
                "position_ids": torch.arange(seq_len, dtype=torch.long)
                .unsqueeze(0)
                .expand_as(input_ids),
            }
        )
    return batches
