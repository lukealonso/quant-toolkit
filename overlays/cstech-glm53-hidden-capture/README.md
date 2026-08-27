# cstech GLM-5.3 hidden-capture overlay

This derivative preserves the exact cstech GLM-5.3 runtime and adds a bounded,
environment-gated capture of the BF16 transformer output immediately after the
final RMSNorm and before the language-model head. Normal serving is unchanged
when `VLLM_KLD_HIDDEN_CAPTURE_DIR` is unset.

The Docker build is pinned to base digest
`sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5`
and refuses to patch unless both touched upstream files match their recorded
SHA-256 values. A CPU-only synthetic capture smoke test runs during the build.

Enable capture by mounting an empty writable directory and setting:

```bash
-v /data/kld/raw-hidden:/capture-hidden:rw \
-e VLLM_KLD_HIDDEN_CAPTURE_DIR=/capture-hidden
```

Routed-fit capture is independently enabled with
`VLLM_GLM53_ROUTED_CAPTURE_DIR`. It requires eager execution, one request per
step, and a full 2,048-token prefill in one scheduler step. Rank zero records
the complete replicated BF16 MoE input before sequence-parallel chunking, FP32
pre-sigmoid router logits, exact top-k IDs, and exact FP32 route weights for
each routed layer. These factorized inputs retain cross-coordinate terms for
offline float64 moment replay without storing dense per-expert covariance
matrices.

Capture runs must use one request at a time, no prefix caching, no speculative
MTP, and exact pre-tokenized prompt IDs. The pinned launcher enforces this when
`CAPTURE_DIR` is set (`MAX_NUM_SEQS=1 PREFIX_CACHING=0 MTP_TOKENS=0`). Rank zero
writes BF16 safetensors chunks named by request ID and prompt-row interval. The
finalizer must verify contiguous rows, discard the last prompt row (which
predicts an uncaptured continuation), bind the result to the canonical token-ID
hash, and seal every artifact.

GLM-5.3-specific identity:

- hidden width: `4096`
- vocabulary size: `154880`
- captured semantic point: `after_final_rmsnorm_before_lm_head`
- the GLM model already applies final RMSNorm before returning hidden states,
  so the exposed capture transform is deliberately the identity function.
