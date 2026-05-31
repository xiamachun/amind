# ── amind Benchmarks ──────────────────────────────────────────────────────

# ── HNSW vector index benchmarks ─────────────────────────────────────────
add_executable(hnsw_bench hnsw_bench.cpp)
target_include_directories(hnsw_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(hnsw_bench PRIVATE
    amind_vector
    benchmark::benchmark
    benchmark::benchmark_main
)
