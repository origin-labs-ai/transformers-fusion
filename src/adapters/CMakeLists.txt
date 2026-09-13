# ============================================================================
# ADAPTER-EDITION — Subdirectory of main Transcender project
# ----------------------------------------------------------------------------
# Links against the parent project's already-built libraries (quant_core,
# quant_format, quant_kernel, quant_model, quant_math).  Does NOT recompile parent
# sources — avoids ODR violations when built as a subdirectory.
#
# Scope: format converters (GGUF, Safetensors, PTQ) + native trainer
# (scratch + continual modes) + delta-adapter host + MTP-head trainer.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(Transcender_Adapter_Edition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ────────────────────────────────────────────────────────────────
option(QUANT_ADAPTER_BUILD_TOOLS "Build adapter edition CLI tools" ON)
option(QUANT_ADAPTER_BUILD_TESTS "Build adapter edition tests" ON)

# Directory holding the parent project's already-built Release libraries
# (quant_core, quant_math, quant_format, quant_kernel, quant_model). Override with
#   -DPARENT_LIB_DIR=<absolute path to parent build Release dir>
set(PARENT_LIB_DIR "" CACHE PATH "Parent project Release lib directory")
if(NOT PARENT_LIB_DIR)
    set(PARENT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build_verify/Release")
endif()
foreach(_quant_lib quant_core quant_math quant_format quant_kernel quant_model)
    if(NOT TARGET ${_quant_lib})
        add_library(${_quant_lib} STATIC IMPORTED)
        set_target_properties(${_quant_lib} PROPERTIES
            IMPORTED_LOCATION "${PARENT_LIB_DIR}/${_quant_lib}.lib")
    endif()
endforeach()

# ── Library: quant_adapter ───────────────────────────────────────────────────
# Adapter bridge implementations + continual trainer + delta adapter + MTP heads
set(ADAPTER_SOURCES
    "src/adapter_core.cpp"
    "src/ptq_bridge.cpp"
    "src/gguf_bridge.cpp"
    "src/safetensors_bridge.cpp"
    "src/continual_trainer.cpp"
    "src/delta_adapter.cpp"
    "src/mtp_head_trainer.cpp"
)

add_library(quant_adapter STATIC ${ADAPTER_SOURCES})
target_include_directories(quant_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)
target_link_libraries(quant_adapter PUBLIC
    quant_core
    quant_math
    quant_format
    quant_kernel
    quant_model
)
if(MSVC)
    target_compile_options(quant_adapter PRIVATE /W3 /utf-8 /DNOMINMAX)
else()
    target_compile_options(quant_adapter PRIVATE -Wall -Wextra)
endif()

# ── CLI tools ──────────────────────────────────────────────────────────────
if(QUANT_ADAPTER_BUILD_TOOLS)
    function(add_adapter_tool name src)
        add_executable(${name} ${src} ${ARGN})
        target_include_directories(${name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        )
        target_link_libraries(${name} PRIVATE quant_adapter)
        if(MSVC)
            target_compile_options(${name} PRIVATE /W3 /utf-8 /DNOMINMAX)
        endif()
    endfunction()

    add_adapter_tool(quant_ptq    "tools/quant_ptq.cpp")
    add_adapter_tool(quant_adapt  "tools/quant_adapt.cpp")
    add_adapter_tool(quant_import "tools/quant_import.cpp")
    add_adapter_tool(bench_compare "tools/bench_compare.cpp")
    add_adapter_tool(bench_full "tools/bench_full.cpp")
    add_adapter_tool(bench_ollama_comparison "tools/bench_ollama_comparison.cpp")
    add_adapter_tool(bench_bitnet_comparison "tools/bench_bitnet_comparison.cpp")
    add_adapter_tool(bench_quant_loss "tools/bench_quant_loss.cpp")
    add_adapter_tool(bench_industry_real "tools/bench_industry_real.cpp")
    add_adapter_tool(probe_grp "tools/probe_grp.cpp")
    add_adapter_tool(probe_search "tools/probe_search.cpp")
    add_adapter_tool(probe_t4 "tools/probe_t4.cpp")
    add_adapter_tool(quant_verify_roundtrip "tools/quant_verify_roundtrip.cpp")
    add_adapter_tool(quant_refcheck "tools/quant_refcheck.cpp")
    add_adapter_tool(dump_quant "tools/dump_quant.cpp")
    add_adapter_tool(quant_chat "tools/quant_chat.cpp" "tools/qwen35_engine.cpp"
        "../../src/tokenizer/qwen35_tokenizer.cpp" "../../src/server/json_parser.cpp")
    if(MSVC)
        target_compile_options(quant_chat PRIVATE /O2 /arch:AVX2)
    endif()
endif()

# ── Tests ──────────────────────────────────────────────────────────────────
if(QUANT_ADAPTER_BUILD_TESTS)
    enable_testing()

    add_executable(test_adapter_bridges "tests/test_adapter_bridges.cpp")
    target_include_directories(test_adapter_bridges PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    )
    target_link_libraries(test_adapter_bridges PRIVATE quant_adapter)
    if(MSVC)
        target_compile_options(test_adapter_bridges PRIVATE /W3 /utf-8 /DNOMINMAX)
    endif()
    add_test(NAME test_adapter_bridges COMMAND test_adapter_bridges)
    set_tests_properties(test_adapter_bridges PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
