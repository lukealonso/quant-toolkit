ARG BASE_IMAGE=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
FROM ${BASE_IMAGE}

COPY glm5next-nvidia-canonical-loader.patch /tmp/glm5next-nvidia-canonical-loader.patch
RUN cd /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia \
    && patch --batch --forward -p0 < /tmp/glm5next-nvidia-canonical-loader.patch \
    && rm /tmp/glm5next-nvidia-canonical-loader.patch

LABEL org.opencontainers.image.base.digest="sha256:0bd709e80b8fb876ee50f7a2a3696ed1372ca3a44438754f80c386620c956df9"
LABEL org.opencontainers.image.description="Pinned GLM-5.3 cstech runtime with canonical Transformers checkpoint-key compatibility"
