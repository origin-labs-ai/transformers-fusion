# tests/CMakeLists.txt — CTest integration with labels and timeouts
# Requires parent CMakeLists.txt to call add_subdirectory(tests)

# ── helper ────────────────────────────────────────────────────────
function(add_quant_test_full name src)
  set(_all_args ${ARGN})
  list(LENGTH _all_args _arg_count)
  math(EXPR _timeout_idx "${_arg_count} - 1")
  list(GET _all_args ${_timeout_idx} _timeout_val)
  list(REMOVE_AT _all_args ${_timeout_idx})
  set(deps ${_all_args})

  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE ${deps})
  add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
  set_tests_properties(${name} PROPERTIES
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>
    TIMEOUT ${_timeout_val}
  )
endfunction()

set(_default_timeout 60)
set(_long_timeout    300)

# ── core ──────────────────────────────────────────────────────────
add_quant_test_full(test_tensor     test_tensor.cpp     quant_core                              ${_default_timeout})
set_tests_properties(test_tensor     PROPERTIES LABELS "core")

add_quant_test_full(test_math       test_math.cpp       quant_math                              ${_default_timeout})
set_tests_properties(test_math       PROPERTIES LABELS "core")

add_quant_test_full(test_format     test_format.cpp     quant_format                            ${_default_timeout})
set_tests_properties(test_format     PROPERTIES LABELS "core")

add_quant_test_full(test_kernel     test_kernel.cpp     quant_kernel                            ${_default_timeout})
set_tests_properties(test_kernel     PROPERTIES LABELS "core")

add_quant_test_full(test_model      test_model.cpp      quant_model                             ${_default_timeout})
add_quant_test_full(test_kda        test_kda.cpp        quant_model                             ${_default_timeout})
add_quant_test_full(test_mla        test_mla.cpp        quant_model                             ${_default_timeout})
set_tests_properties(test_model      PROPERTIES LABELS "core")
set_tests_properties(test_kda        PROPERTIES LABELS "core")
set_tests_properties(test_mla        PROPERTIES LABELS "core")
add_quant_test_full(test_hybrid_scheduler test_hybrid_scheduler.cpp quant_model     ${_default_timeout})
set_tests_properties(test_hybrid_scheduler PROPERTIES LABELS "core")
add_quant_test_full(test_hybrid_model test_hybrid_model.cpp quant_model          ${_default_timeout})
set_tests_properties(test_hybrid_model PROPERTIES LABELS "core")
add_quant_test_full(test_k3_tokenizer test_k3_tokenizer.cpp quant_tokenizer          ${_default_timeout})
set_tests_properties(test_k3_tokenizer PROPERTIES LABELS "core")

add_quant_test_full(test_tokenizer  test_tokenizer.cpp  quant_tokenizer                         ${_default_timeout})
set_tests_properties(test_tokenizer  PROPERTIES LABELS "core")

add_quant_test_full(test_debug      test_debug.cpp      quant_model quant_production                   ${_default_timeout})

# ── GLE telemetry ─────────────────────────────────────────────────
add_quant_test_full(test_gle_telemetry test_gle_telemetry.cpp quant_gle ${_default_timeout})
set_tests_properties(test_gle_telemetry PROPERTIES LABELS "gle")

# ── MIX component invariants (L013/L014) ─────────────────────────
add_quant_test_full(test_mix_components test_mix_components.cpp quant_format ${_default_timeout})
set_tests_properties(test_mix_components PROPERTIES LABELS "core")
set_tests_properties(test_debug      PROPERTIES LABELS "core")

add_quant_test_full(test_all        test_all.cpp        quant_moe_variants quant_backend quant_continual ${_long_timeout})
set_tests_properties(test_all        PROPERTIES LABELS "core;distributed")

# ── training ──────────────────────────────────────────────────────
add_quant_test_full(test_trainer    test_trainer.cpp    quant_trainer quant_inference quant_tokenizer ${_long_timeout})
set_tests_properties(test_trainer    PROPERTIES LABELS "training")

add_quant_test_full(test_training   test_training.cpp   quant_trainer quant_model quant_engine quant_math quant_core ${_long_timeout})
set_tests_properties(test_training   PROPERTIES LABELS "training" TIMEOUT 600)

add_quant_test_full(test_gradient_check test_gradient_check.cpp quant_trainer quant_model quant_math quant_core ${_long_timeout})
set_tests_properties(test_gradient_check PROPERTIES LABELS "training")

add_quant_test_full(test_expert_parallel test_expert_parallel.cpp quant_moe_variants quant_trainer quant_model quant_math quant_kernel ${_long_timeout})
set_tests_properties(test_expert_parallel PROPERTIES LABELS "training;distributed")

add_quant_test_full(test_mixed_precision_proof test_mixed_precision_proof.cpp quant_trainer quant_model quant_math quant_core ${_long_timeout})
set_tests_properties(test_mixed_precision_proof PROPERTIES LABELS "training")

add_quant_test_full(test_quant_engines   test_quant_engines.cpp   quant_format quant_kernel quant_math ${_long_timeout})
set_tests_properties(test_quant_engines   PROPERTIES LABELS "training")

add_quant_test_full(test_quant_convergence test_quant_convergence.cpp quant_format quant_kernel quant_math quant_core quant_model quant_trainer quant_inference ${_long_timeout})
set_tests_properties(test_quant_convergence PROPERTIES LABELS "training")

add_quant_test_full(test_native_quant   test_native_quant.cpp   quant_trainer quant_model quant_kernel quant_format ${_long_timeout})
set_tests_properties(test_native_quant   PROPERTIES LABELS "training")

add_quant_test_full(test_native_quant_moe test_native_quant_moe.cpp quant_trainer quant_model quant_moe_variants quant_kernel quant_format ${_long_timeout})
set_tests_properties(test_native_quant_moe PROPERTIES LABELS "training;distributed")

add_quant_test_full(test_moe_training test_moe_training.cpp quant_trainer quant_model quant_moe_variants quant_kernel ${_long_timeout})
set_tests_properties(test_moe_training PROPERTIES LABELS "training;distributed")
add_quant_test_full(test_grpo         test_grpo.cpp         quant_trainer quant_model quant_moe_variants quant_kernel ${_long_timeout})
set_tests_properties(test_grpo PROPERTIES LABELS "training;rl")
add_quant_test_full(test_rollout_worker test_rollout_worker.cpp quant_trainer quant_model quant_moe_variants ${_long_timeout})
set_tests_properties(test_rollout_worker PROPERTIES LABELS "training;rl")

add_quant_test_full(test_training_features test_training_features.cpp quant_trainer quant_model quant_math quant_core ${_long_timeout})
set_tests_properties(test_training_features PROPERTIES LABELS "training")

add_quant_test_full(test_fine_tuning test_fine_tuning.cpp quant_trainer quant_model quant_tokenizer quant_format ${_long_timeout})
set_tests_properties(test_fine_tuning PROPERTIES LABELS "training")

add_quant_test_full(test_native_weight test_native_weight.cpp quant_trainer quant_model quant_kernel quant_format ${_long_timeout})
set_tests_properties(test_native_weight PROPERTIES LABELS "training")

# ── inference ─────────────────────────────────────────────────────
add_quant_test_full(test_inference_opt test_inference_opt.cpp quant_inference_opt quant_model quant_inference ${_default_timeout})
set_tests_properties(test_inference_opt PROPERTIES LABELS "inference")

add_quant_test_full(test_inference_batch test_inference_batch.cpp quant_inference quant_inference_opt quant_model quant_math quant_core quant_moe_variants ${_default_timeout})
set_tests_properties(test_inference_batch PROPERTIES LABELS "inference")

add_quant_test_full(test_ops          test_ops.cpp          quant_kernel quant_math quant_core quant_model quant_inference quant_backend quant_format ${_default_timeout})
set_tests_properties(test_ops          PROPERTIES LABELS "inference")

# ── GPU ───────────────────────────────────────────────────────────
if(WIN32)
  target_link_libraries(test_all PRIVATE quant_gpu)
  add_executable(test_gpu test_gpu.cpp)
  target_include_directories(test_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(test_gpu PRIVATE quant_math quant_gpu quant_backend)
  add_test(NAME test_gpu COMMAND $<TARGET_FILE:test_gpu>)
  set_tests_properties(test_gpu PROPERTIES
    WORKING_DIRECTORY $<TARGET_FILE_DIR:test_gpu>
    TIMEOUT ${_long_timeout}
    LABELS "gpu"
  )
endif()

# ── distributed / MoE ────────────────────────────────────────────
add_quant_test_full(test_moe_crash_repro test_moe_crash_repro.cpp quant_moe_variants quant_model ${_long_timeout})
set_tests_properties(test_moe_crash_repro PROPERTIES LABELS "distributed")

# ── AGI / multimodal / production ─────────────────────────────────
add_quant_test_full(test_agi          test_agi.cpp          quant_agi quant_trainer quant_model quant_tokenizer ${_long_timeout})
set_tests_properties(test_agi          PROPERTIES LABELS "inference")

add_quant_test_full(test_agi_flywheel  test_agi_flywheel.cpp quant_agi quant_trainer quant_model         ${_long_timeout})
set_tests_properties(test_agi_flywheel  PROPERTIES LABELS "inference")

add_quant_test_full(test_multimodal   test_multimodal.cpp   quant_multimodal quant_model quant_tokenizer quant_trainer quant_inference ${_long_timeout})
set_tests_properties(test_multimodal   PROPERTIES LABELS "inference")

add_quant_test_full(test_multimodal_encoders test_multimodal_encoders.cpp quant_multimodal quant_model quant_tokenizer ${_long_timeout})
set_tests_properties(test_multimodal_encoders PROPERTIES LABELS "inference")

add_quant_test_full(test_production   test_production.cpp   quant_production quant_model quant_tokenizer ${_long_timeout})
set_tests_properties(test_production   PROPERTIES LABELS "inference")

# ── protection / integrity ────────────────────────────────────────
add_quant_test_full(test_protected   test_protected.cpp   quant_moe_variants quant_backend quant_trainer quant_tokenizer quant_model quant_kernel quant_format quant_math quant_inference quant_engine ${_long_timeout})
set_tests_properties(test_protected   PROPERTIES LABELS "core;training")

add_quant_test_full(test_sha256_corrupt test_sha256_corrupt.cpp quant_core quant_format               ${_default_timeout})
set_tests_properties(test_sha256_corrupt PROPERTIES LABELS "core")

add_quant_test_full(test_format_registry_complete test_format_registry_complete.cpp quant_format quant_core ${_long_timeout})
set_tests_properties(test_format_registry_complete PROPERTIES LABELS "core")

add_quant_test_full(grp_proof_test test_grp_quality_proof.cpp quant_format quant_core        ${_long_timeout})
set_tests_properties(grp_proof_test PROPERTIES LABELS "core")

add_quant_test_full(test_fp32_gather_quality test_fp32_gather_quality.cpp quant_format quant_core      ${_long_timeout})
set_tests_properties(test_fp32_gather_quality PROPERTIES LABELS "core")

add_quant_test_full(test_bpw_150_proof test_bpw_150_proof.cpp quant_format quant_core                  ${_long_timeout})
set_tests_properties(test_bpw_150_proof PROPERTIES LABELS "core")

add_quant_test_full(test_block_codec test_block_codec.cpp quant_kernel quant_format quant_core              ${_long_timeout})
set_tests_properties(test_block_codec PROPERTIES LABELS "core")

add_quant_test_full(test_quant_mix test_quant_mix.cpp quant_format quant_trainer quant_model quant_kernel quant_core ${_long_timeout})
set_tests_properties(test_quant_mix PROPERTIES LABELS "training;core")

# ── new architecture feature tests ──────────────────────────────
add_quant_test_full(test_cuda_backend test_cuda_backend.cpp quant_backend quant_gpu ${_default_timeout})
set_tests_properties(test_cuda_backend PROPERTIES LABELS "gpu;core")

add_quant_test_full(test_expert_prefetch test_expert_prefetch.cpp quant_moe_variants quant_core ${_default_timeout})
set_tests_properties(test_expert_prefetch PROPERTIES LABELS "distributed;moe")

add_quant_test_full(test_continual_anticollapse test_continual_anticollapse.cpp quant_continual quant_core ${_default_timeout})
set_tests_properties(test_continual_anticollapse PROPERTIES LABELS "training;core")

# ── paged KV cache ───────────────────────────────────────────────
add_quant_test_full(test_paged_kv_4m  paged_kv_4m_test.cpp  quant_model quant_kernel               ${_long_timeout})
set_tests_properties(test_paged_kv_4m  PROPERTIES LABELS "inference")

# ── parallel execution ────────────────────────────────────────────
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 4)
endif()
set_tests_properties(
  test_tensor test_math test_format test_kernel test_model test_tokenizer
  test_ops test_sha256_corrupt test_expert_prefetch test_continual_anticollapse
  test_gpu test_cuda_backend
  PROPERTIES COST 1.0
)
set_tests_properties(
  test_trainer test_training test_gradient_check test_expert_parallel
  test_all test_protected test_inference_opt test_agi test_multimodal
  test_production test_mixed_precision_proof test_native_quant test_moe_crash_repro
  test_quant_engines test_quant_convergence test_native_weight
  test_moe_training test_native_quant_moe test_multimodal_encoders
  test_agi_flywheel test_inference_batch test_paged_kv_4m
  test_training_features grp_proof_test test_block_codec
  test_fp32_gather_quality test_bpw_150_proof test_quant_mix test_format_registry_complete
  PROPERTIES COST 2.0
)

# ── codec fuzz roundtrip (L036) ───────────────────────────────────
add_quant_test_full(test_fuzz_codec test_fuzz_codec.cpp quant_kernel quant_format quant_core ${_long_timeout})
set_tests_properties(test_fuzz_codec PROPERTIES LABELS "codec;core")
