cmake_minimum_required(VERSION 3.20)
project(InNova VERSION 0.1.2 LANGUAGES CXX)

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
  src/tensor.cpp
  src/memory.cpp
  src/random.cpp
  src/tensor_view.cpp
  src/data_gen.cpp
  src/metrics.cpp
  src/parquet_streamer.cpp
  src/telemetry.cpp
)
target_include_directories(quant_core PUBLIC include)
target_link_libraries(quant_core PUBLIC quant_config)

add_library(quant_math
  src/math.cpp
  src/math_avx2.cpp
  src/math_avx2_tiled.cpp
  src/math_avx2_tensor.cpp
  src/math_avx2_vec.cpp
  src/simd_math.cpp
)
if(QUANT_AVX512)
  target_sources(quant_math PRIVATE src/math_avx512.cpp)
endif()
target_include_directories(quant_math PUBLIC include)
target_link_libraries(quant_math PUBLIC quant_core)

add_library(quant_format
  src/quant_format.cpp
  src/codebook.cpp
  src/format_planner.cpp
  src/format_registry.cpp
  src/block_codec.cpp
  src/vector_quantizer.cpp
  src/quant_engines_core.cpp
  src/quant_engines_fp.cpp
  src/quant_engines_quant1.cpp
  src/quant_engines_quant2.cpp
  src/quant_engines_quant4.cpp
  src/quant_engines_quant8.cpp
  src/quant_engines_quant.cpp
)
target_include_directories(quant_format PUBLIC include)
target_link_libraries(quant_format PUBLIC quant_math)

add_library(quant_kernel
  src/kernel_tl.cpp
  src/kernel_quant8.cpp
  src/kernel_quant4.cpp
  src/kernel_q3.cpp
  src/kernel_q6.cpp
  src/kernel_q12.cpp
  src/kernel_q24.cpp
  src/kernel_gemm.cpp
  src/int8_quant.cpp
  src/kernel_production.cpp
)
target_include_directories(quant_kernel PUBLIC include)
target_link_libraries(quant_kernel PUBLIC quant_format)

add_library(quant_model
  src/transformer.cpp
  src/kda_attention.cpp
  src/mla_attention.cpp
  src/hybrid_scheduler.cpp
  src/hybrid_block.cpp
  src/hybrid_expert.cpp
  src/hybrid_moe_model.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/kv_cache_quant4.cpp
  src/autograd_engine.cpp
  src/autograd_functions.cpp
  src/autograd_grad.cpp
  src/flash_attention.cpp
)
target_include_directories(quant_model PUBLIC include)
target_link_libraries(quant_model PUBLIC quant_kernel)

add_library(quant_inference
  src/sampler.cpp
  src/generator.cpp
)
target_include_directories(quant_inference PUBLIC include)
target_link_libraries(quant_inference PUBLIC quant_model)

add_library(quant_tokenizer
  src/bpe_tokenizer_advanced.cpp
  src/bpe_tokenizer_bpe.cpp
  src/bpe_tokenizer_unicode.cpp
  src/qwen35_tokenizer.cpp
  src/k3_tokenizer.cpp
  src/dataset.cpp
)
target_include_directories(quant_tokenizer PUBLIC include)
target_link_libraries(quant_tokenizer PUBLIC quant_core)

add_library(quant_trainer
  src/optimizer.cpp
  src/scheduler.cpp
  src/eval.cpp
  src/ste_quantizer.cpp
  src/qat.cpp
  src/finetune.cpp
  src/fine_tuning.cpp
  src/log_writer.cpp
  src/distributed.cpp
  src/native_weight.cpp
  src/native_trainer.cpp
  src/hf_streamer.cpp
  src/moe_trainer.cpp
  src/hpo_nas.cpp
  src/ddp.cpp
  src/pipeline.cpp
  src/reward.cpp
  src/continual_engine.cpp
  src/trainer_core.cpp
  src/trainer_data.cpp
  src/trainer_rl.cpp
  src/trainer_rl_ops.cpp
  src/rollout_worker.cpp
  src/training_utils.cpp
  src/zero_offload.cpp
  src/zero_optimizer_core.cpp
  src/zero_optimizer_mem.cpp
)
target_include_directories(quant_trainer PUBLIC include)
target_link_libraries(quant_trainer PUBLIC quant_model quant_inference quant_tokenizer quant_continual)

add_library(quant_engine
  engines/inference/inference.cpp
  engines/inference/stream.cpp
  engines/quant/quantize.cpp
  engines/quant/codec.cpp
  src/code_gen.cpp
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
  add_library(quant_gpu src/gpu_compute.cpp src/gpu_compute_full.cpp src/gpu_compute_vulkan.cpp src/gpu_compute_cuda.cpp src/igpu_zero_copy.cpp src/gpu_compute_sycl.cpp src/gpu_compute_cann.cpp src/gpu_compute_rpc.cpp src/gpu_compute_metal.cpp src/gpu_compute_openvino.cpp src/gpu_compute_virtgpu.cpp src/gpu_compute_webgpu.cpp src/gpu_compute_zendnn.cpp src/gpu_compute_hexagon.cpp src/gpu_compute_hip.cpp src/gpu_compute_musa.cpp src/gpu_compute_opencl.cpp src/gpu_compute_zdnn.cpp)
  target_include_directories(quant_gpu PUBLIC include)
  target_link_libraries(quant_gpu PUBLIC quant_core)
  target_link_libraries(quant_gpu PUBLIC d3d12 dxgi d3dcompiler)
  target_compile_definitions(quant_gpu PRIVATE QUANT_USE_DIRECTX)
  target_sources(quant_gpu PRIVATE src/gpu_extras.cpp)
else()
  add_library(quant_gpu src/gpu_compute.cpp src/gpu_compute_full.cpp src/gpu_compute_vulkan.cpp src/gpu_compute_cuda.cpp src/igpu_zero_copy.cpp src/gpu_compute_sycl.cpp src/gpu_compute_cann.cpp src/gpu_compute_rpc.cpp src/gpu_compute_metal.cpp src/gpu_compute_openvino.cpp src/gpu_compute_virtgpu.cpp src/gpu_compute_webgpu.cpp src/gpu_compute_zendnn.cpp src/gpu_compute_hexagon.cpp src/gpu_compute_hip.cpp src/gpu_compute_musa.cpp src/gpu_compute_opencl.cpp src/gpu_compute_zdnn.cpp)
  target_include_directories(quant_gpu PUBLIC include)
endif()

# Hardware backend abstraction
add_library(quant_backend
  src/backend.cpp
)
target_include_directories(quant_backend PUBLIC include)
target_link_libraries(quant_backend PUBLIC quant_math quant_gpu)

# All MoE variants
add_library(quant_moe_variants
  src/moe_variants.cpp
  src/moe_model.cpp
  src/expert_parallel.cpp
  src/expert_prefetch.cpp
  src/moe_advanced_support.cpp
  src/moe_quantized_test_core.cpp
  src/moe_quantized_test_extra.cpp
  src/mommoe_block.cpp
  src/native_quant_moe.cpp
)
target_include_directories(quant_moe_variants PUBLIC include)
target_link_libraries(quant_moe_variants PUBLIC quant_model quant_backend quant_trainer)

# Distributed training (tensor parallelism, FSDP)
add_library(quant_distributed
  src/tensor_parallelism.cpp
  src/fsdp.cpp
)
target_include_directories(quant_distributed PUBLIC include)
target_link_libraries(quant_distributed PUBLIC quant_model quant_kernel)

# Cat D: Inference optimization
add_library(quant_inference_opt
  src/inference_opt.cpp
)
target_include_directories(quant_inference_opt PUBLIC include)
target_link_libraries(quant_inference_opt PUBLIC quant_model quant_inference)

# Cat F: MoE enhancement
add_library(quant_moe_enhance
  src/moe_enhance.cpp
)
target_include_directories(quant_moe_enhance PUBLIC include)
target_link_libraries(quant_moe_enhance PUBLIC quant_model quant_moe_variants)

# Cat G: AGI Pipeline
add_library(quant_agi
  src/agi.cpp
  src/agi_flywheel.cpp
  src/agi_extended.cpp
  src/meta_cognition.cpp
  src/self_eval.cpp
  src/world_model.cpp
  src/multi_agent.cpp
  src/agi_guardrails.cpp
)
target_include_directories(quant_agi PUBLIC include)
target_link_libraries(quant_agi PUBLIC quant_trainer quant_model)

# Cat H: Multi-Modal
add_library(quant_multimodal
  src/multimodal.cpp
  src/multimodal_cross_attn.cpp
  src/multimodal_fusion.cpp
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
  src/production_core.cpp
  src/production_gen.cpp
  src/production_api.cpp
  src/production_gen_ops.cpp
  src/http_server.cpp
  src/json_parser.cpp
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

if(QUANT_BUILD_TOOLS)
  add_executable(quant_infer tools/infer.cpp
    src/qwen35_tokenizer.cpp
    src/json_parser.cpp
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

  add_executable(k3_convert tools/k3_convert.cpp)
  target_link_libraries(k3_convert PRIVATE quant_format quant_model quant_tokenizer)

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
    src/qwen35_tokenizer.cpp
    src/json_parser.cpp
  )
  target_link_libraries(quant_server PRIVATE quant_model quant_tokenizer)

  add_executable(quant_evaluate tools/evaluate.cpp
    src/qwen35_tokenizer.cpp
    src/json_parser.cpp
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
                 --title "InNova Coverage"
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
  src/sops_integration.cpp
)
target_include_directories(quant_sops PUBLIC include sops)
target_link_libraries(quant_sops PUBLIC quant_core)
set_target_properties(quant_sops PROPERTIES CXX_STANDARD 20)

# Thread safety primitives
add_library(quant_thread_safety STATIC
  src/thread_safety.cpp
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
  src/continual_engine.cpp
)
target_include_directories(quant_continual PUBLIC include)
target_link_libraries(quant_continual PUBLIC quant_format quant_sops)

# Speculative decoding engine
add_library(quant_speculative STATIC
  src/speculative_decoder.cpp
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
