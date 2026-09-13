FROM ubuntu:24.04 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    g++-13 \
    gcc-13 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# P2 fix: CMake sets no RUNTIME_OUTPUT_DIRECTORY, so binaries land at the
# build-root and build/tests/ — NOT build/tools/ or build/bench/. Also gated
# -DQUANT_AVX2=ON on amd64 only (breaks ARM image builds otherwise).
ARG TARGETARCH=amd64
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_C_COMPILER=gcc-13 \
    $([ "$TARGETARCH" = "amd64" ] && echo -DQUANT_AVX2=ON || echo -DQUANT_AVX2=OFF) \
    -DQUANT_BUILD_TESTS=ON \
    -DQUANT_BUILD_BENCHMARKS=ON \
    -DQUANT_BUILD_TOOLS=ON \
    -DCMAKE_CXX_STANDARD=20 \
    && cmake --build build --parallel $(nproc)

# D7 W5: exclusion kept intentionally for docker build speed (PR-speed set, aligned with ci_full.yml Quick/ASAN/Coverage).
# - test_protected: heavy integrity/monolith test, too slow for image build.
# - test_gpu: requires GPU/CUDA hardware; not present in docker build.
# - test_training: long training loop (600s TIMEOUT), nightly-full-asan only.
# - test_native_quant: long training-linked quant test, ASAN-heavy/slow.
# - test_moe_training: distributed MoE training test, heavy/flaky, nightly only.
# - test_paged_kv_4m: large-memory 256MiB cache test, slow in image build.
# - test_fuzz_codec: 1.05M-roundtrip fuzz (900s TIMEOUT in Debug), nightly only.
# NOTE: the union of the nightly-full-asan shards in .github/workflows/ci_full.yml
# covers the full suite with NO excludes (shard-union gate fails loudly on orphans).
RUN ctest --test-dir build --output-on-failure --timeout 300 \
    --exclude-regex "test_protected|test_gpu|test_training|test_native_quant|test_moe_training|test_paged_kv_4m|test_fuzz_codec"

FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/false appuser

WORKDIR /app

COPY --from=builder /app/build/quant_infer .
COPY --from=builder /app/build/quant_convert .
COPY --from=builder /app/build/quant_bench .
COPY --from=builder /app/build/bench_kernels .
COPY --from=builder /app/build/bench_inference .
COPY --from=builder /app/build/bench_quality .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Binary-presence probe: no-arg run prints Usage to stderr and exits 1 —
# grep for it so a present+executable binary reports healthy (exit 0).
# NOTE: HEALTHCHECK takes CMD + shell form (no CMD-SHELL keyword — that is
# a parse error, 2026-09-13). Runs under /bin/sh -c so pipes work.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ./quant_infer 2>&1 | grep -q Usage

ENTRYPOINT ["./quant_infer"]
