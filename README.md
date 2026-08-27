Ask your friendly neighborhood AI agent how to use this.

## GLM-5.3-Flash routed-expert NVFP4

The first GLM-5.3 profile is deliberately conservative: only routed expert
gate/up/down linears are NVFP4. KDA linear attention, NoPE sparse attention and
its indexer, vision, hyper-connections, routers, shared experts, dense MLPs,
embeddings, the LM head, and MTP remain in source precision.

```bash
MODEL_ID=/path/to/GLM-5.3-Flash-BF16 \
EXPORT_DIR=/data/models/GLM-5.3-Flash-NVFP4-routed \
scripts/quantize_glm5_3_flash.sh
```

This covers about 304.4B of 321.3B parameters and is expected to produce a
checkpoint near 195-205 GB after NVFP4 block scales and preserved tensors. Do
not widen coverage based only on local reconstruction error; use BF16-teacher
KLD to qualify each candidate.

Before loading an exported candidate, verify the exact fail-closed coverage,
packed tensor shapes/dtypes, positive finite scales, tied gate/up global
scales, protected BF16 MTP layer, and absence of any unexpected quantized
tensors. The full shard-hash pass is the release receipt:

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with safetensors \
  python tools/verify_glm53_nvfp4_checkpoint.py \
  --source-model /data/models/GLM-5.3-Flash-BF16-b1967181 \
  --candidate-model /data/models/GLM-5.3-Flash-NVFP4-routed \
  --source-revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --candidate-name glm53-flash-nvfp4-routed-v1 \
  --hash-candidate-shards \
  --output /data/kld/runtimes/glm53-flash-nvfp4-routed-v1.coverage.json
```

For the pinned BF16 source the verifier must report exactly 36,288 quantized
weight tensors and 304,405,807,104 quantized parameters (94.7351% of the
321,323,031,390-parameter checkpoint). The resulting tensor payload is
205,063,596,408 bytes before safetensors headers.

Before replacing ModelOpt's reconstruction, prove that the local codec emits
the serving format exactly. The audit re-encodes gate, up, and down from BF16,
including the required shared gate/up secondary scale, and fails unless every
packed nibble and scale is bit-identical to the stock checkpoint:

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with safetensors \
  python tools/audit_glm53_nvfp4_codec.py \
  --source-model /data/models/GLM-5.3-Flash-BF16-b1967181 \
  --candidate-model /data/models/GLM-5.3-Flash-NVFP4-routed \
  --layer 3 --expert 0 \
  --output /data/kld/runtimes/glm53-nvfp4-codec-audit.json
```

Once codec parity passes, audit the first exact reconstruction lever on a real
BF16 expert. This searches nearby representable FP8 E4M3 block scales while
keeping the packed E2M1 runtime format and the tied gate/up secondary scale.
The unweighted result is only a reconstruction baseline; routed activation
moments and held-out end-to-end KLD decide whether a candidate is accepted.

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with safetensors \
  python tools/audit_glm53_nvfp4_scale_search.py \
  --source-model /data/models/GLM-5.3-Flash-BF16-b1967181 \
  --layer 3 --expert 0 --device cpu \
  --artifact /data/kld/runtimes/glm53-scale-search-l3-e0.safetensors \
  --output /data/kld/runtimes/glm53-scale-search-l3-e0.json
```

## Dense-prefill KLD

The campaign workflow is capture once, compare many. Every weight/topology/KV
combination gets one sealed full-vocabulary logit capture. Comparisons are then
computed offline, so canonical, total-deployment, and within-weight KV-only KLD
do not require another model load. Publishable captures use float32 storage;
do not downcast them when qualifying a checkpoint.

For the two-workstation deployment, keep CPU-heavy work off the GPU hosts. Use
the operator Mac and `uv` for corpus preparation, hashing, offline comparison,
independent replay, and reporting. Workstation-1 and workstation-2 are reserved
for distributed model inference and logit capture; copy sealed captures away
before running the offline tools. Do not overlap a workstation CPU analysis job
with a GPU model run.

### Pinned GLM-5.3 campaign path

The pinned cstech OpenAI server does not expose a practical dense-logit HTTP
response. `prompt_logprobs=-1` would materialize the full vocabulary through
Python objects and JSON. For this campaign, capture the final-normalized BF16
hidden rows with the derivative overlay, then replay the common protected BF16
LM head into float32 safetensors. Normal serving is unchanged when capture is
disabled.

The external teacher dataset is pinned at
`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits@17ffa06d2ee899b6043c3307bc572bc944ed5ea5`.
Its 25 F32 windows contain 51,175 prediction positions. The source BF16 model
revision is `a6c167b62691b2bac901344b65cb651a70f53e43`; all 120 weight shards,
`config.json`, and the tensor index are byte-identical to the lab's pinned
`b1967181a3917ae70a437f4884748f6b8e3a1f4d` checkpoint. The pinned dataset
now includes the sealed panel, the 25 held-out
held-out `final-*.tokens.npy` payloads, all fit/selection/conditional-fit/
confirmation token payloads, and their causal mask. The mirror verifies all
665 token files against the hashes in the sealed panel before import.

Import the teacher without copying its 31.7 GB logit payload:

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with 'numpy>=2.0' --with safetensors \
  python tools/import_glm53_teacher_logits.py \
  --dataset-dir /data/GLM-5.3-Flash-BF16-Teacher-Logits \
  --token-panel-dir /data/GLM-5.3-Flash-BF16-Teacher-Logits/calibration/panel-v1 \
  --output-dir /data/kld/captures/bf16-transformers-teacher
```

Build the capture image from the exact base digest:

```bash
docker build \
  -t local/glm53-cstech-hidden-capture:base-0bd709e-capture-v1 \
  overlays/cstech-glm53-hidden-capture
```

The first verified build produced image ID
`sha256:9e123d5f1677bade647356dd07e48cdacd9f4537ede1a4921e4ef1996417fd06`.
After launching that image with an empty writable capture mount and
`VLLM_KLD_HIDDEN_CAPTURE_DIR`, capture the exact teacher token suite:

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with 'numpy>=2.0' --with safetensors --with requests \
  python tools/capture_glm53_hidden_suite.py \
  --suite-manifest /data/GLM-5.3-Flash-BF16-Teacher-Logits/dataset-manifest.json \
  --token-panel-dir /data/GLM-5.3-Flash-BF16-Teacher-Logits/calibration/panel-v1 \
  --capture-dir /data/kld/raw-hidden/fp8-tp4-fp8-ds-mla \
  --output-dir /data/kld/hidden/fp8-tp4-fp8-ds-mla \
  --runtime-manifest /data/kld/runtimes/fp8-tp4-fp8-ds-mla.json \
  --run-name fp8-tp4-fp8-ds-mla
```

For calibration-panel roles, bind the capture directly to the sealed panel
instead of synthesizing an intermediate suite manifest. For example, candidate
ranking uses only the 64 selection windows:

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with 'numpy>=2.0' --with safetensors --with requests \
  python tools/capture_glm53_hidden_suite.py \
  --panel-dir /data/GLM-5.3-Flash-BF16-Teacher-Logits/calibration/panel-v1 \
  --panel-role selection \
  --capture-dir /data/kld/raw-hidden/nvfp4-selection \
  --output-dir /data/kld/hidden/nvfp4-selection \
  --runtime-manifest /data/kld/runtimes/nvfp4-selection.json \
  --run-name nvfp4-selection
```

Export the shared head once from the pinned BF16 checkpoint, then reconstruct
every candidate capture. Run the projection on a GPU only after unloading the
serving model; the output remains float32.

```bash
uv run --no-config --no-project --python 3.12 \
  --with torch --with safetensors \
  python tools/export_glm53_lm_head.py \
  --model-dir /data/models/GLM-5.3-Flash-BF16-b1967181 \
  --model-revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --output-dir /data/kld/lm-head-b1967181

uv run --no-config --no-project --python 3.12 \
  --with torch --with safetensors \
  python tools/replay_glm53_hidden_logits.py \
  --hidden-dir /data/kld/hidden/fp8-tp4-fp8-ds-mla \
  --lm-head-dir /data/kld/lm-head-b1967181 \
  --output-dir /data/kld/captures/fp8-tp4-fp8-ds-mla \
  --device cuda:0
```

The external no-cache/eager BF16 teacher is a clean weight-and-attention
reference, but it is not evidence for a BF16 KV-cache policy. Keep that
distinction explicit in the final comparison.

The direct in-process path below remains available for a runtime that supports
both BF16 KV and dense prompt-logit return. It is not runnable against the
current pinned sparse-MLA SM120 server.

Capture the canonical BF16-weights/BF16-KV run:

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash-BF16 \
  --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --output-dir /data/kld/captures/bf16-tp8-bf16-kv \
  --role canonical --run-label bf16-tp8-bf16-kv \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --kv-cache-dtype bfloat16 \
  --storage-dtype float32 \
  --context-length 2048 --stride 512 --max-windows 25
```

Capture every other run with `--role candidate`, the same pinned corpus,
context length, stride, and window count. For example:

```bash
python tools/collect_prefill_logits.py \
  --model /path/to/GLM-5.3-Flash \
  --output-dir /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --role candidate --run-label fp8-tp4-fp8-ds-mla-kv \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --storage-dtype float32 \
  --context-length 2048 --stride 512 --max-windows 25
```

Compare any two captures without loading either model:

```bash
uv run --no-project --with torch --with 'numpy>=2.0' --with safetensors \
  python tools/compare_captured_prefill_kld.py \
  --reference-logits /data/kld/captures/bf16-tp8-bf16-kv \
  --candidate-logits /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --output-dir /data/kld/reports/bf16-bf16__fp8-fp8-ds-mla
```

Independently replay that report. The verifier uses NumPy and an explicit
log-sum-exp implementation rather than the producer's PyTorch `kl_div` path,
checks both manifests, every artifact hash, and exact input-token equality, and
rejects a tokenwise discrepancy above `1e-12` by default.

```bash
uv run --no-project --with 'numpy>=2.0' --with safetensors \
  python tools/replay_captured_prefill_kld.py \
  --reference-logits /data/kld/captures/bf16-tp8-bf16-kv \
  --candidate-logits /data/kld/captures/fp8-tp4-fp8-ds-mla-kv \
  --report /data/kld/reports/bf16-bf16__fp8-fp8-ds-mla \
  --output-dir /data/kld/verifications/bf16-bf16__fp8-fp8-ds-mla
```

Reports contain tokenwise KLD plus mean, median, p95, p99, p99.9, maximum, and
top-1 agreement. The direction is always `KL(reference || candidate)`, in both nats
and bits. The vLLM build must expose `return_prompt_logits` or the lab's flat
full-logprob fallback. `score_prefill_kld.py` and `replay_prefill_kld.py` remain
available for one-off live comparisons, but they are not the preferred campaign
path.

### GLM-5.3 NoPE KV policy gate

Do not import DeepSeek's mixed-RoPE cache names into GLM-5.3-Flash. The pinned
model has `qk_rope_head_dim=0`, `qk_nope_head_dim=256`, and
`kv_lora_rank=512`; there is no model RoPE tensor to preserve or quantize.

The pinned cstech SM120 image currently resolves the intended policies as
follows:

| Intended tier | Requested dtype | Effective GLM layout | Current status |
|---|---|---|---|
| BF16 throughout | `bfloat16` | 512-wide BF16 latent cache | No compatible sparse-MLA SM120 backend in the pinned image |
| BF16 RoPE + FP8 NoPE | `fp8` | `fp8_ds_mla`: 512 FP8 NoPE values, four FP32 scales, and a padded/reserved 64-wide BF16 RoPE region | Supported; this is the current production layout |
| FP8 RoPE + NVFP4 NoPE | no valid flag yet | Must be a genuine NVFP4 latent-cache kernel; GLM has no actual RoPE part | Unsupported by the pinned sparse-MLA SM120 backends |

`nvfp4_4over6` is not the third mixed layout. It selects max/6 versus max/4
scaling per 16 NVFP4 values by reconstruction error. Do not record an NVFP4 KV
result until backend selection and the startup log prove a distinct supported
layout.

The machine-readable support record, including exact image/source hashes,
parser choices, backend aliases, and the 656-byte tensor layout, is
`manifests/glm5_3_flash_cstech_kv_support_0bd709e.json`.

The exact two-node BF16 control launcher is
`scripts/run_glm53_cstech_node.sh`. Start the worker first from
workstation-1's private-rail SSH path, then start the head locally:

```bash
ssh jack@10.42.20.2 \
  'NODE_RANK=1 DCP_SIZE=1 /path/to/run_glm53_cstech_node.sh'
NODE_RANK=0 DCP_SIZE=1 scripts/run_glm53_cstech_node.sh
```

Use `DCP_SIZE=4` for the requested capacity/control rerun. The launcher refuses
unsupported KV dtype names and preserves the two-hour Gloo/NCCL startup
timeouts needed for a cold 598.52-GiB checkpoint on workstation-2's ZFS pool.
The cold DCP1 control took 2143.02 seconds to load its main weights because
automatic prefetch was disabled and the demand-fault path only drove the
three-drive ZFS stripe at roughly 130--170 MiB/s. The pinned implementation's
explicit prefetch path is rank-sharded (`sorted_files[rank::world_size]`), not
a whole-checkpoint read by every rank. Future launches therefore default to
`SAFETENSORS_LOAD_STRATEGY=prefetch`, eight threads per rank, and 16-MiB
blocks. Set `SAFETENSORS_LOAD_STRATEGY=auto` only when reproducing the cold
control. Record the host ARC cap and load strategy in each run receipt.

For a hidden-state capture run, build the identical derivative image on both
nodes, override the image on both ranks, and set the capture invariants
explicitly. The launcher refuses unsafe mixes. Start the worker first:

```bash
ssh jack@10.42.20.2 \
  'IMAGE=local/glm53-cstech-hidden-capture:base-0bd709e-capture-v1 \
  EXPECTED_IMAGE_ID=sha256:9e123d5f1677bade647356dd07e48cdacd9f4537ede1a4921e4ef1996417fd06 \
  CAPTURE_DIR=/data/kld/raw-hidden/bf16-tp8-dcp1-fp8-ds-mla \
  MAX_NUM_SEQS=1 PREFIX_CACHING=0 MTP_TOKENS=0 \
  NODE_RANK=1 DCP_SIZE=1 /path/to/run_glm53_cstech_node.sh'

IMAGE=local/glm53-cstech-hidden-capture:base-0bd709e-capture-v1 \
EXPECTED_IMAGE_ID=sha256:9e123d5f1677bade647356dd07e48cdacd9f4537ede1a4921e4ef1996417fd06 \
CAPTURE_DIR=/data/kld/raw-hidden/bf16-tp8-dcp1-fp8-ds-mla \
MAX_NUM_SEQS=1 PREFIX_CACHING=0 MTP_TOKENS=0 \
NODE_RANK=0 DCP_SIZE=1 scripts/run_glm53_cstech_node.sh
```

For the TP4 campaign, use this order:

1. Once supported, capture BF16 weights on TP8 under BF16, production FP8, and
   genuine NVFP4 KV policies.
2. Unload BF16, then capture official FP8 weights on TP4 under the same three
   policies.
3. Compare all five non-canonical captures with BF16-weights/BF16-KV. Also
   compare BF16 policy pairs and official-FP8 policy pairs for within-weight
   KV-only deltas.
4. Use official FP8 TP4 KLD, cache capacity, prefill, and decode measurements
   to eliminate dominated KV policies.
5. Quantize routed experts, then capture NVFP4 weights on TP4 under the same
   policies. Compare against the canonical capture and within the NVFP4 weight
   family before running expensive concurrency and Estonia gates.

Keep corpus, tokenizer, context windows, attention backend, and arithmetic
settings fixed. BF16 requires TP8 while the deployment target is TP4, so its
canonical comparison intentionally includes the small topology/runtime numeric
delta. Official FP8 and NVFP4 candidates remain TP4 throughout; a matched-TP8
FP8 control is optional and should only be run if the topology delta needs to be
separated.
