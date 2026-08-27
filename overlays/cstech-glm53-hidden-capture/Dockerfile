FROM cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5

LABEL org.opencontainers.image.description="Pinned cstech GLM-5.3 runtime with environment-gated BF16 pre-LM-head capture"
LABEL org.opencontainers.image.base.digest="sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5"

WORKDIR /usr/local/lib/python3.12/dist-packages

COPY patches/cstech-glm53-hidden-capture.patch /tmp/cstech-glm53-hidden-capture.patch
COPY vllm/v1/worker/gpu/distribution_capture.py vllm/v1/worker/gpu/distribution_capture.py
COPY tests/smoke_distribution_capture.py /tmp/smoke_distribution_capture.py

RUN set -eu; \
    test "$(sha256sum vllm/v1/worker/gpu/model_runner.py | awk '{print $1}')" = "f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415"; \
    test "$(sha256sum vllm/models/glm5next/nvidia/model.py | awk '{print $1}')" = "6f45ef524e1918d3fd8248eb5e06b9b993885bce1ed33b26299494b5defea17f"; \
    patch --batch --forward -p1 </tmp/cstech-glm53-hidden-capture.patch; \
    python3 -m py_compile \
      vllm/v1/worker/gpu/distribution_capture.py \
      vllm/v1/worker/gpu/model_runner.py \
      vllm/models/glm5next/nvidia/model.py; \
    python3 /tmp/smoke_distribution_capture.py
