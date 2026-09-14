cmake_minimum_required(VERSION 3.20)
# Display version: R0001.01 (owner scheme). CMake needs numeric: 1.1.0 mirrors R0001.01.
project(Transcender VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(arch)
include(compiler)

option(QUANT_BUILD_TESTS "Build unit tests" ON)
option(QUANT_BUILD_BENCHMARKS "Build benchmarks" ON)
option(QUANT_BUILD_TOOLS "Build CLI tools" ON)
option(QUANT_BUILD_ADAPTERS "Build adapter bridge tree (src/adapters, EXCLUDE_FROM_ALL, default OFF until QUANT_ names migrated)" OFF)
option(QUANT_SANITIZE "Enable address sanitizer" OFF)
option(QUANT_COVERAGE "Enable code coverage (GCC/Clang only)" OFF)

if(QUANT_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    message(STATUS "Code coverage enabled (--coverage)")
  else()
    message(WARNING "QUANT_COVERAGE only supported with GCC or Clang")
  endif()
endif()

set(QUANT_COVERAGE_LCOV_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH
  "Directory for lcov coverage reports")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/quant_config.h.in"
  "${CMAKE_BINARY_DIR}/quant_config.h"
)

add_library(quant_config INTERFACE)
target_include_directories(quant_config INTERFACE "${CMAKE_BINARY_DIR}")

add_library(quant_core
  src/core/tensor.cpp
  src/core/memory.cpp
  src/core/random.cpp
  src/core/tensor_view.cpp
  src/core/data_gen.cpp
  src/core/metrics.cpp
  src/core/parquet_streamer.cpp
  src/core/telemetry.cpp
)
target_include_directories(quant_core PUBLIC include)
target_link_libraries(quant_core PUBLIC quant_config)

add_library(quant_math
  src/math/math.cpp
  src/math/math_avx2.cpp
  src/math/math_avx2_tiled.cpp
  src/math/math_avx2_vec.cpp
  src/math/simd_math.cpp
)
# NOTE (P18): src/math/math_avx2_tensor.cpp was a byte-identical duplicate of
# math_avx2.cpp (gemm/gemv/dot/axpy/norm/asum/relu → LNK2005) — DELETED
# (owner order 2026-09-13), do not re-add. See exp_avx2.h for the shared
# AVX2 exp approximation.
if(QUANT_AVX512)
  target_sources(quant_math PRIVATE src/math/math_avx512.cpp)
endif()
target_include_directories(quant_math PUBLIC include)
target_link_libraries(quant_math PUBLIC quant_core)

add_library(quant_format
  src/codec/quant_format.cpp
  src/codec/codebook.cpp
  src/codec/format_planner.cpp
  src/codec/format_registry.cpp
  src/codec/block_codec.cpp
  src/codec/vector_quantizer.cpp
  src/codec/quant_engines_core.cpp
  src/codec/quant_engines_fp.cpp
  src/codec/quant_engines_quant1.cpp
  src/codec/quant_engines_quant2.cpp
  src/codec/quant_engines_quant4.cpp
  src/codec/quant_engines_quant8.cpp
  src/codec/quant_engines_quant.cpp
)
target_include_directories(quant_format PUBLIC include)
target_link_libraries(quant_format PUBLIC quant_math)

add_library(quant_kernel
  src/kernel/kernel_tl.cpp
  src/kernel/kernel_quant8.cpp
  src/kernel/kernel_quant4.cpp
  src/kernel/kernel_q3.cpp
  src/kernel/kernel_q6.cpp
  src/kernel/kernel_q12.cpp
  src/kernel/kernel_q24.cpp
  src/kernel/kernel_gemm.cpp
  src/kernel/int8_quant.cpp
  src/kernel/kernel_production.cpp
)
target_include_directories(quant_kernel PUBLIC include)
target_link_libraries(quant_kernel PUBLIC quant_format)

add_library(quant_model
  src/model/transformer.cpp
  src/model/hybrid_scheduler.cpp
  src/model/hybrid_block.cpp
  src/model/hybrid_expert.cpp
  src/model/hybrid_moe_model.cpp
  src/model/model.cpp
  src/model/kv_cache.cpp
  src/model/kv_cache_quant4.cpp
  src/model/autograd_engine.cpp
  src/model/autograd_functions.cpp
  src/model/autograd_grad.cpp
  src/kernel/flash_attention.cpp
)
target_include_directories(quant_model PUBLIC include)
target_link_libraries(quant_model PUBLIC quant_kernel)

add_library(quant_inference
  src/inference/sampler.cpp
  src/inference/generator.cpp
)
target_include_directories(quant_inference PUBLIC include)
# generator.cpp includes quant/tokenizer.h (Tokenizer* members + calls):
# the dependency must be declared or static-link order breaks. Found via
# ASAN CI (2026-09-13): -fsanitize=undefined vptr checks keep a typeinfo
# reference to BPETokenizer alive that normal builds optimize out, so only
# the ASAN link died (undefined reference to `typeinfo for BPETokenizer`).
target_link_libraries(quant_inference PUBLIC quant_model quant_tokenizer)

add_library(quant_tokenizer
  src/tokenizer/bpe_tokenizer_advanced.cpp
  src/tokenizer/bpe_tokenizer_bpe.cpp
  src/tokenizer/bpe_tokenizer_unicode.cpp
  src/tokenizer/qwen35_tokenizer.cpp
  src/core/dataset.cpp
)
target_include_directories(quant_tokenizer PUBLIC include)
target_link_libraries(quant_tokenizer PUBLIC quant_core)

add_library(quant_trainer
  src/trainer/optimizer.cpp
  src/trainer/scheduler.cpp
  src/trainer/eval.cpp
  src/codec/ste_quantizer.cpp
  src/codec/qat.cpp
  src/trainer/finetune.cpp
  src/trainer/fine_tuning.cpp
  src/core/log_writer.cpp
  src/trainer/distributed.cpp
  src/trainer/native_weight.cpp
  src/trainer/native_trainer.cpp
  src/trainer/hf_streamer.cpp
  src/trainer/moe_trainer.cpp
  src/trainer/hpo_nas.cpp
  src/trainer/ddp.cpp
  src/trainer/pipeline.cpp
  src/trainer/reward.cpp
  src/trainer/continual_engine.cpp
  src/trainer/trainer_core.cpp
  src/trainer/trainer_data.cpp
  src/trainer/trainer_rl.cpp
  src/trainer/trainer_rl_ops.cpp
  src/trainer/rollout_worker.cpp
  src/trainer/training_utils.cpp
  src/trainer/zero_offload.cpp
  src/trainer/zero_optimizer_core.cpp
  src/trainer/zero_optimizer_mem.cpp
)
target_include_directories(quant_trainer PUBLIC include)
target_link_libraries(quant_trainer PUBLIC quant_model quant_inference quant_tokenizer quant_continual)

add_library(quant_engine
  engines/inference/inference.cpp
  engines/inference/stream.cpp
  engines/quant/quantize.cpp
  engines/quant/codec.cpp
  src/trainer/code_gen.cpp
)
target_include_directories(quant_engine PUBLIC include engines/inference engines/quant)
target_link_libraries(quant_engine PUBLIC quant_inference quant_tokenizer)



add_library(quant_dense
  engines/trainer/dense/trainer.cpp
  engines/trainer/dense/dataloader.cpp
  engines/trainer/dense/checkpoint.cpp
)
target_include_directories(quant_dense PUBLIC include engines/trainer/dense)
target_link_libraries(quant_dense PUBLIC quant_trainer quant_continual)

add_library(quant_moe
  engines/trainer/moe/moe.cpp
  engines/trainer/moe/vision/vision.cpp
  engines/trainer/moe/audio/audio.cpp
  engines/trainer/moe/embeddings/embeddings.cpp
  engines/trainer/moe/image/vision.cpp
  engines/trainer/moe/ocr/ocr.cpp
  engines/trainer/moe/text/multimodal_text.cpp
  engines/trainer/moe/video/video.cpp
)
target_include_directories(quant_moe PUBLIC include
  engines/trainer/moe
  engines/trainer/moe/vision
  engines/trainer/moe/audio
  engines/trainer/moe/embeddings
  engines/trainer/moe/image
  engines/trainer/moe/ocr
  engines/trainer/moe/text
  engines/trainer/moe/video
)
target_link_libraries(quant_moe PUBLIC quant_model)

# GPU compute library (Cross-platform, all 18 backends registered)
if(WIN32)
  add_library(quant_gpu src/backend/gpu_compute.cpp src/backend/gpu_compute_full.cpp src/backend/gpu_compute_vulkan.cpp src/backend/gpu_compute_cuda.cpp src/backend/igpu_zero_copy.cpp src/backend/gpu_compute_sycl.cpp src/backend/gpu_compute_cann.cpp src/backend/gpu_compute_rpc.cpp src/backend/gpu_compute_metal.cpp src/backend/gpu_compute_openvino.cpp src/backend/gpu_compute_virtgpu.cpp src/backend/gpu_compute_webgpu.cpp src/backend/gpu_compute_zendnn.cpp src/backend/gpu_compute_hexagon.cpp src/backend/gpu_compute_hip.cpp src/backend/gpu_compute_musa.cpp src/backend/gpu_compute_opencl.cpp src/backend/gpu_compute_zdnn.cpp)
  target_include_directories(quant_gpu PUBLIC include)
  target_link_libraries(quant_gpu PUBLIC quant_core)
  target_link_libraries(quant_gpu PUBLIC d3d12 dxgi d3dcompiler)
  target_compile_definitions(quant_gpu PRIVATE QUANT_USE_DIRECTX)
  target_sources(quant_gpu PRIVATE src/backend/gpu_extras.cpp)
else()
  add_library(quant_gpu src/backend/gpu_compute.cpp src/backend/gpu_compute_full.cpp src/backend/gpu_compute_vulkan.cpp src/backend/gpu_compute_cuda.cpp src/backend/igpu_zero_copy.cpp src/backend/gpu_compute_sycl.cpp src/backend/gpu_compute_cann.cpp src/backend/gpu_compute_rpc.cpp src/backend/gpu_compute_metal.cpp src/backend/gpu_compute_openvino.cpp src/backend/gpu_compute_virtgpu.cpp src/backend/gpu_compute_webgpu.cpp src/backend/gpu_compute_zendnn.cpp src/backend/gpu_compute_hexagon.cpp src/backend/gpu_compute_hip.cpp src/backend/gpu_compute_musa.cpp src/backend/gpu_compute_opencl.cpp src/backend/gpu_compute_zdnn.cpp)
  target_include_directories(quant_gpu PUBLIC include)
endif()

# Hardware backend abstraction
add_library(quant_backend
  src/backend/backend.cpp
)
target_include_directories(quant_backend PUBLIC include)
target_link_libraries(quant_backend PUBLIC quant_math quant_gpu)

# All MoE variants
add_library(quant_moe_variants
  src/model/moe_core.cpp
  src/model/moe_variants.cpp
  src/model/moe_model.cpp
  src/model/expert_parallel.cpp
  src/model/expert_prefetch.cpp
  src/model/moe_advanced_support.cpp
  src/model/moe_quantized_test_core.cpp
  src/model/moe_quantized_test_extra.cpp
  src/model/mommoe_block.cpp
  src/model/native_quant_moe.cpp
)
target_include_directories(quant_moe_variants PUBLIC include)
target_link_libraries(quant_moe_variants PUBLIC quant_model quant_backend quant_trainer)

# Distributed training (tensor parallelism, FSDP)
add_library(quant_distributed
  src/trainer/tensor_parallelism.cpp
  src/trainer/fsdp.cpp
)
target_include_directories(quant_distributed PUBLIC include)
target_link_libraries(quant_distributed PUBLIC quant_model quant_kernel)

# Cat D: Inference optimization
add_library(quant_inference_opt
  src/inference/inference_opt.cpp
  src/inference/inference_paging.cpp
  src/inference/inference_speculative.cpp
  src/inference/inference_kv.cpp
  src/inference/inference_lowprec.cpp
  src/inference/inference_serve.cpp
)
target_include_directories(quant_inference_opt PUBLIC include)
target_link_libraries(quant_inference_opt PUBLIC quant_model quant_inference)

# Cat F: MoE enhancement
add_library(quant_moe_enhance
  src/model/moe_enhance.cpp
)
target_include_directories(quant_moe_enhance PUBLIC include)
target_link_libraries(quant_moe_enhance PUBLIC quant_model quant_moe_variants)

# Cat G: AGI Pipeline
add_library(quant_agi
  src/agi/agi.cpp
  src/agi/agi_flywheel.cpp
  src/agi/agi_extended.cpp
  src/agi/meta_cognition.cpp
  src/agi/self_eval.cpp
  src/agi/world_model.cpp
  src/agi/multi_agent.cpp
  src/agi/agi_guardrails.cpp
  src/agi/sandbox.cpp
)
target_include_directories(quant_agi PUBLIC include)
target_link_libraries(quant_agi PUBLIC quant_trainer quant_model)

# Cat H: Multi-Modal
add_library(quant_multimodal
  src/multimodal/multimodal.cpp
  src/multimodal/multimodal_cross_attn.cpp
  src/multimodal/multimodal_fusion.cpp
  engines/trainer/multimodal/vision/vision.cpp
  engines/trainer/multimodal/image/image.cpp
  engines/trainer/multimodal/video/video.cpp
  engines/trainer/multimodal/text/text.cpp
  engines/trainer/multimodal/ocr/ocr.cpp
  engines/trainer/multimodal/embeddings/embeddings.cpp
  engines/trainer/multimodal/audio/audio.cpp
)
target_include_directories(quant_multimodal PUBLIC include
  engines/trainer/multimodal
  engines/trainer/multimodal/vision
  engines/trainer/multimodal/image
  engines/trainer/multimodal/video
  engines/trainer/multimodal/text
  engines/trainer/multimodal/ocr
  engines/trainer/multimodal/embeddings
  engines/trainer/multimodal/audio
)
target_link_libraries(quant_multimodal PUBLIC quant_model)
target_link_libraries(quant_multimodal PUBLIC quant_tokenizer quant_trainer)

# Cat I: Production (HTTP server, logging, config, C API, plugins)
add_library(quant_production
  src/server/production_core.cpp
  src/server/production_gen.cpp
  src/server/production_api.cpp
  src/server/production_gen_ops.cpp
  src/server/http_server.cpp
  src/server/json_parser.cpp
)
target_include_directories(quant_production PUBLIC include)
target_link_libraries(quant_production PUBLIC quant_model quant_inference quant_tokenizer)
if(WIN32)
  target_link_libraries(quant_production PUBLIC ws2_32)
endif()

# Vulkan compute backend (dynamically loaded at runtime, no SDK required)
option(QUANT_VULKAN "Enable Vulkan compute backend (dynamically loaded)" ON)
if(QUANT_VULKAN AND WIN32)
  target_compile_definitions(quant_gpu PRIVATE QUANT_USE_VULKAN)
endif()

# CUDA compute backend (dynamically loaded at runtime, no SDK required)
option(QUANT_CUDA "Enable CUDA compute backend (dynamically loaded)" ON)
if(QUANT_CUDA)
  target_compile_definitions(quant_gpu PRIVATE QUANT_USE_CUDA)
  target_compile_definitions(quant_backend PRIVATE QUANT_USE_CUDA)
endif()

# Apple ObjC runtime: gpu_compute_metal.cpp calls objc_getClass/objc_msgSend/
# sel_registerName directly (dynamic Metal dispatch, no Metal SDK needed).
# These live in libobjc (/usr/lib) — NOT auto-linked for plain C++ targets,
# so every Apple link otherwise dies with "_objc_getClass/_objc_msgSend/
# _sel_registerName not found" (macOS CI, bench_hardware link, 2026-09-13).
if(APPLE)
  target_link_libraries(quant_gpu PUBLIC "-lobjc")
endif()

if(QUANT_BUILD_TOOLS)
  add_executable(quant_infer tools/infer.cpp
    src/tokenizer/qwen35_tokenizer.cpp
    src/server/json_parser.cpp
  )
  target_include_directories(quant_infer PRIVATE engines/inference)
  target_link_libraries(quant_infer PRIVATE quant_engine quant_tokenizer)

  add_executable(quant_train tools/train.cpp)
  target_link_libraries(quant_train PRIVATE quant_dense quant_tokenizer)

  add_executable(train_64m tools/train_64m.cpp)
  target_link_libraries(train_64m PRIVATE quant_dense quant_trainer quant_tokenizer)

  add_executable(quant_finetune tools/finetune.cpp)
  target_link_libraries(quant_finetune PRIVATE quant_trainer)

  add_executable(quant_convert tools/convert.cpp)
  target_link_libraries(quant_convert PRIVATE quant_format)

  add_executable(quant_bench tools/bench.cpp)
  target_link_libraries(quant_bench PRIVATE quant_kernel quant_inference quant_tokenizer)

  add_executable(quant_info tools/info.cpp)
  target_link_libraries(quant_info PRIVATE quant_format)

  add_executable(quant_format_list tools/format_list.cpp)
  target_link_libraries(quant_format_list PRIVATE quant_format)

  add_executable(quant_serve tools/serve.cpp)
  target_link_libraries(quant_serve PRIVATE quant_production quant_model quant_tokenizer)

  add_executable(quant_quantize tools/quantize.cpp)
  target_link_libraries(quant_quantize PRIVATE quant_format quant_kernel)

  add_executable(quant_server tools/quant_server.cpp
    src/tokenizer/qwen35_tokenizer.cpp
    src/server/json_parser.cpp
  )
  target_link_libraries(quant_server PRIVATE quant_model quant_tokenizer)

  add_executable(quant_evaluate tools/evaluate.cpp
    src/tokenizer/qwen35_tokenizer.cpp
    src/server/json_parser.cpp
  )
  target_link_libraries(quant_evaluate PRIVATE quant_model quant_inference quant_tokenizer quant_trainer)

  add_executable(generate_comparison_visuals tools/generate_comparison_visuals.cpp)
  target_link_libraries(generate_comparison_visuals PRIVATE quant_format quant_core quant_math)
endif()

if(QUANT_BUILD_TESTS)
  function(add_quant_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${ARGN})
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
    set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>)
  endfunction()

  include(CTest)
  add_subdirectory(tests)

  if(QUANT_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)

    if(LCOV_PATH AND GENHTML_PATH)
      add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${QUANT_COVERAGE_LCOV_DIR}
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
        COMMAND ${CMAKE_COMMAND} -E echo "-- Running all tests for coverage..."
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j $ENV{NPROC}
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --capture
                 --output-file ${QUANT_COVERAGE_LCOV_DIR}/coverage.info
        COMMAND ${LCOV_PATH} --remove ${QUANT_COVERAGE_LCOV_DIR}/coverage.info
                 '/usr/*' '/opt/*' '${CMAKE_BINARY_DIR}/_deps/*'
                 '*/tests/*' '*/bench/*' '*/tools/*'
                 --output-file ${QUANT_COVERAGE_LCOV_DIR}/coverage_filtered.info
        COMMAND ${GENHTML_PATH} ${QUANT_COVERAGE_LCOV_DIR}/coverage_filtered.info
                 --output-directory ${QUANT_COVERAGE_LCOV_DIR}/html
                 --title "Transcender Coverage"
                 --legend
        COMMENT "Generating code coverage report in ${QUANT_COVERAGE_LCOV_DIR}/html"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      message(STATUS "Coverage target available: cmake --build . --target coverage")
    else()
      message(WARNING "lcov/genhtml not found; coverage target not available")
    endif()
  endif()
endif()

if(QUANT_BUILD_BENCHMARKS)
  macro(add_quant_bench name src deps)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE ${deps} ${ARGN})
  endmacro()
  add_quant_bench(bench_kernels   bench/bench_kernels.cpp   quant_kernel quant_math)
  add_quant_bench(bench_inference bench/bench_inference.cpp quant_inference quant_model quant_math quant_core)
  add_quant_bench(bench_gemm_tiled bench/bench_gemm_tiled.cpp quant_math quant_core)
  add_quant_bench(bench_hotswap_lat bench/bench_hotswap_lat.cpp quant_core)
  add_quant_bench(bench_throughput bench/bench_throughput.cpp quant_core)
  add_executable(probe_mi_extraction tools/probe_mi_extraction.cpp)
  target_include_directories(probe_mi_extraction PRIVATE include)
  target_link_libraries(probe_mi_extraction PRIVATE quant_agi quant_core)
  add_executable(bench_quality bench/bench_quality.cpp)
  target_include_directories(bench_quality PRIVATE include)
  target_link_libraries(bench_quality PRIVATE quant_inference quant_tokenizer quant_model quant_format quant_math quant_core)

  # J1-J15: All benchmarks
  add_quant_bench(bench_all bench/bench_all.cpp quant_model quant_math quant_format)

  # Hardware-probe / auto-select benchmark
  add_quant_bench(bench_hardware tests/test_bench.cpp quant_backend quant_math)

  # QUANT quantized GEMM benchmarks (Gap #2)
  add_quant_bench(bench_quant_quant bench/bench_quant_quant.cpp quant_kernel quant_math quant_core)

  # GPT-2 real inference quality test
  add_quant_bench(bench_gpt2_inference bench/bench_gpt2_inference.cpp quant_format quant_core)

  # Training benchmarks
  add_quant_bench(bench_training bench/bench_training.cpp quant_trainer quant_model quant_tokenizer)
  add_quant_bench(bench_multimodal bench/bench_multimodal.cpp quant_multimodal quant_model quant_tokenizer)

  # Training speed benchmarks
  add_quant_bench(bench_speed tools/bench_speed.cpp quant_trainer quant_model)

  # PoC Benchmark: all in-house QUANT/QUANT formats + QUANT_MIX routing
  add_quant_bench(bench_poc bench/bench_poc_benchmark.cpp quant_format quant_core quant_math)
  add_executable(bench_format_comparison bench/bench_format_comparison.cpp)
  target_include_directories(bench_format_comparison PRIVATE include)
  target_link_libraries(bench_format_comparison PRIVATE quant_trainer quant_continual quant_format quant_core quant_math quant_model quant_engine)
endif()

# SOPS tools + integration
add_library(quant_sops STATIC
  sops/sops_scheduler.cpp
  src/core/sops_integration.cpp
)
target_include_directories(quant_sops PUBLIC include sops)
target_link_libraries(quant_sops PUBLIC quant_core)
set_target_properties(quant_sops PROPERTIES CXX_STANDARD 20)

# Thread safety primitives
add_library(quant_thread_safety STATIC
  src/core/thread_safety.cpp
)
target_include_directories(quant_thread_safety PUBLIC include)
target_link_libraries(quant_thread_safety PUBLIC quant_core)

# GLE telemetry — Gauntlet Loop Enhanced append-only event stream (PART-J)
add_library(quant_gle STATIC
  src/gle/gle_telemetry.cpp
)
target_include_directories(quant_gle PUBLIC ${CMAKE_SOURCE_DIR}/src/gle)
set_target_properties(quant_gle PROPERTIES CXX_STANDARD 20)

# GLE report generator (markdown summary from events.jsonl)
add_executable(gle_report tools/gle_report.cpp)
target_link_libraries(gle_report PRIVATE quant_gle)

# Continual learning engine
add_library(quant_continual STATIC
  src/trainer/continual_engine.cpp
)
target_include_directories(quant_continual PUBLIC include)
target_link_libraries(quant_continual PUBLIC quant_format quant_sops)

# Speculative decoding engine
add_library(quant_speculative STATIC
  src/inference/speculative_decoder.cpp
)
target_include_directories(quant_speculative PUBLIC include)
target_link_libraries(quant_speculative PUBLIC quant_model quant_sops)

add_executable(bench_sops sops/sops_bench.cpp)
target_include_directories(bench_sops PRIVATE include)
target_link_libraries(bench_sops PRIVATE quant_core)
set_target_properties(bench_sops PROPERTIES CXX_STANDARD 20)

add_executable(sops_info sops/sops_format_info.cpp)
target_include_directories(sops_info PRIVATE include)
set_target_properties(sops_info PROPERTIES CXX_STANDARD 20)

add_executable(sops_train_sim sops/sops_training_sim.cpp)
target_include_directories(sops_train_sim PRIVATE include)
target_link_libraries(sops_train_sim PRIVATE quant_core)
set_target_properties(sops_train_sim PROPERTIES CXX_STANDARD 20)

# Adapter bridge tree (P18: wired, no file missed, no delete).
# Default OFF because adapters still reference pre-QG names (QUANT_Q1_G/TWI_MIX era)
# and need migration to QG4/QG_4_K_M/QG_4.5/Q_MX_4.5/QG_MX_4.5. EXCLUDE_FROM_ALL
# keeps default build green while `cmake --build build --target quant_adapter`
# still works when -DQUANT_BUILD_ADAPTERS=ON.
if(QUANT_BUILD_ADAPTERS)
  add_subdirectory(src/adapters EXCLUDE_FROM_ALL)
endif()

# ===========================================================================
# P2: install + packaging — `cmake --install` / CPack shipped nothing before
# (zero install() rules). Tools + public headers + static libs install so a
# user can `cmake --install build --prefix <dir>` and get a working tree.
# ===========================================================================
include(GNUInstallDirs)
if(QUANT_BUILD_TOOLS)
  install(TARGETS
    quant_infer quant_train quant_finetune quant_convert quant_info
    quant_bench quant_format_list quant_serve quant_quantize quant_server
    quant_evaluate
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/quant DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
# Generated header (not the .in template): version/feature macros live in the
# configured copy at ${CMAKE_BINARY_DIR}/quant_config.h.
install(FILES "${CMAKE_BINARY_DIR}/quant_config.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(CPACK_PACKAGE_NAME "TransCender")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "ORIGIN Labs")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Zero-dependency C++20 AI engine (QUANT format)")
set(CPACK_GENERATOR "ZIP;TGZ")
include(CPack)
