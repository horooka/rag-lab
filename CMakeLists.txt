cmake_minimum_required(VERSION 3.22)
project(rag-lab VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
find_package(Iconv REQUIRED)

if(TARGET SQLite3::SQLite3)
    set(RAG_SQLITE_TARGET SQLite3::SQLite3)
else()
    set(RAG_SQLITE_TARGET SQLite::SQLite3)
endif()

set(RAG_DEFAULT_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/bge-small-en-v1.5"
    CACHE PATH "Default embedding model directory")
set(RAG_ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "ONNX Runtime installation (fetched by scripts/fetch-deps.sh)")

set(RAG_REQUIRED_THIRD_PARTY
    third_party/tree-sitter/lib/src/lib.c
    third_party/tree-sitter-c/src/parser.c
    third_party/tree-sitter-cpp/src/parser.c
    third_party/tree-sitter-python/src/parser.c
    third_party/tree-sitter-pascal/src/parser.c
    third_party/sqlite-vec/sqlite-vec.c
    third_party/CLI11/CLI/CLI.hpp
    third_party/nlohmann/nlohmann/json.hpp
    third_party/onnxruntime/include/onnxruntime_cxx_api.h)

foreach(required IN LISTS RAG_REQUIRED_THIRD_PARTY)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${required}")
        message(FATAL_ERROR
            "Missing ${required}.\n"
            "Run scripts/fetch-deps.sh (or scripts/fetch-deps.sh --force) first.")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Vendored third-party C libraries
# ---------------------------------------------------------------------------
add_library(tree_sitter STATIC third_party/tree-sitter/lib/src/lib.c)
target_include_directories(tree_sitter
    PUBLIC third_party/tree-sitter/include
    PRIVATE third_party/tree-sitter/lib/src)

add_library(treesitter_c STATIC third_party/tree-sitter-c/src/parser.c)
target_include_directories(treesitter_c PRIVATE third_party/tree-sitter-c/src)

add_library(treesitter_cpp STATIC
    third_party/tree-sitter-cpp/src/parser.c
    third_party/tree-sitter-cpp/src/scanner.c)
target_include_directories(treesitter_cpp PRIVATE third_party/tree-sitter-cpp/src)

add_library(treesitter_python STATIC
    third_party/tree-sitter-python/src/parser.c
    third_party/tree-sitter-python/src/scanner.c)
target_include_directories(treesitter_python PRIVATE third_party/tree-sitter-python/src)

add_library(treesitter_pascal STATIC
    third_party/tree-sitter-pascal/src/parser.c)
target_include_directories(treesitter_pascal
    PRIVATE third_party/tree-sitter-pascal/src)

add_library(sqlite_vec STATIC third_party/sqlite-vec/sqlite-vec.c)
target_compile_definitions(sqlite_vec PRIVATE SQLITE_CORE SQLITE_VEC_STATIC)
target_include_directories(sqlite_vec PRIVATE third_party/sqlite-vec)
target_link_libraries(sqlite_vec PUBLIC ${RAG_SQLITE_TARGET})

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${RAG_ORT_DIR}/lib/libonnxruntime.so"
    IMPORTED_SONAME "libonnxruntime.so.1"
    INTERFACE_INCLUDE_DIRECTORIES "${RAG_ORT_DIR}/include")

# ---------------------------------------------------------------------------
# Embedding support (shared by all apps, no index or retrieval dependencies)
# ---------------------------------------------------------------------------
add_library(rag_embed STATIC
    src/onnx_embedder.cpp
    src/tokenizer.cpp
    src/util.cpp)

target_include_directories(rag_embed
    PUBLIC include
    PRIVATE third_party/nlohmann)

target_compile_options(rag_embed PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rag_embed PUBLIC onnxruntime Threads::Threads)
set_target_properties(rag_embed PROPERTIES
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "${RAG_ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# RAG core
# ---------------------------------------------------------------------------
add_library(rag_core STATIC
    src/config.cpp
    src/encoding.cpp
    src/source.cpp
    src/tree_sitter_chunker.cpp
    src/markdown_chunker.cpp
    src/chunker_registry.cpp
    src/sqlite_store.cpp
    src/indexer.cpp
    src/search.cpp
    src/eval.cpp
    src/workspace.cpp
    src/resolve.cpp)

target_include_directories(rag_core
    PUBLIC include
    PRIVATE src
            third_party/tree-sitter-c/bindings
            third_party/tree-sitter-cpp/bindings
            third_party/tree-sitter-python/bindings
            third_party/tree-sitter-pascal/bindings
            third_party/CLI11
            third_party/nlohmann
            third_party/sqlite-vec)

target_compile_definitions(rag_core
    PRIVATE RAG_DEFAULT_MODEL_DIR="${RAG_DEFAULT_MODEL_DIR}"
            SQLITE_CORE)

target_compile_options(rag_core PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(rag_core
    PUBLIC ${RAG_SQLITE_TARGET}
    PRIVATE rag_embed
            tree_sitter
            treesitter_c
            treesitter_cpp
            treesitter_python
            treesitter_pascal
            sqlite_vec
            Threads::Threads
            Iconv::Iconv)

set_target_properties(rag_core PROPERTIES
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "${RAG_ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
add_executable(rag apps/rag/main.cpp)
target_include_directories(rag PRIVATE third_party/CLI11 third_party/nlohmann)
target_link_libraries(rag PRIVATE rag_core)
set_target_properties(rag PROPERTIES
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# semantic-grouper app
# ---------------------------------------------------------------------------
add_library(grouper_core STATIC
    src/grouper_clustering.cpp
    src/grouper_input.cpp)
target_include_directories(grouper_core PUBLIC include)
target_compile_options(grouper_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(grouper apps/grouper/main.cpp)
target_include_directories(grouper PRIVATE third_party/CLI11 third_party/nlohmann)
target_compile_definitions(grouper PRIVATE
    GROUPER_DEFAULT_MODEL_DIR="${RAG_DEFAULT_MODEL_DIR}")
target_compile_options(grouper PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(grouper PRIVATE grouper_core rag_embed)
set_target_properties(grouper PROPERTIES
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# MCP stdio server
# ---------------------------------------------------------------------------
add_executable(rag_mcp apps/mcp/main.cpp)
target_include_directories(rag_mcp PRIVATE third_party/CLI11 third_party/nlohmann)
target_compile_options(rag_mcp PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rag_mcp PRIVATE rag_core)
set_target_properties(rag_mcp PROPERTIES
    OUTPUT_NAME rag-mcp
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# coverage app: how much of the index belongs to supplied semantic groups
# ---------------------------------------------------------------------------
add_library(coverage_core STATIC src/coverage.cpp)
target_include_directories(coverage_core PUBLIC include)
target_compile_options(coverage_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(coverage_core PUBLIC rag_core)

add_executable(coverage apps/coverage/main.cpp)
target_include_directories(coverage PRIVATE third_party/CLI11 third_party/nlohmann)
target_compile_options(coverage PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(coverage PRIVATE coverage_core)
set_target_properties(coverage PROPERTIES
    BUILD_RPATH "${RAG_ORT_DIR}/lib"
    INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
    add_executable(rag_tests tests/test_core.cpp)
    target_include_directories(rag_tests PRIVATE third_party/nlohmann)
    target_link_libraries(rag_tests PRIVATE rag_core coverage_core)
    add_test(NAME rag_tests COMMAND rag_tests)
    set_tests_properties(rag_tests PROPERTIES
        ENVIRONMENT "RAG_MODEL_DIR=${RAG_DEFAULT_MODEL_DIR}")
    add_test(NAME rag_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            RAG_MODEL_DIR=${RAG_DEFAULT_MODEL_DIR}
            ${CMAKE_SOURCE_DIR}/tests/smoke.sh $<TARGET_FILE:rag>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(NAME mcp_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            RAG_MODEL_DIR=${RAG_DEFAULT_MODEL_DIR}
            python3 ${CMAKE_SOURCE_DIR}/tests/mcp_smoke.py
                $<TARGET_FILE:rag_mcp> $<TARGET_FILE:rag>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

install(TARGETS rag grouper coverage rag_mcp RUNTIME DESTINATION bin)
install(DIRECTORY "${RAG_ORT_DIR}/lib/" DESTINATION lib
        USE_SOURCE_PERMISSIONS)
