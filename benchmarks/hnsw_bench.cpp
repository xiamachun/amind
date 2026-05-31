
#include "vector/hnsw_index.h"

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace amind;

namespace {

constexpr size_t kDim = 128;

/// Generate a random normalized vector.
std::vector<float> randomVector(std::mt19937& rng, size_t dim) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (auto& v : vec) {
        v = dist(rng);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) v /= norm;
    return vec;
}

/// Build an index with N vectors (heap-allocated, HNSWIndex is non-movable due to mutex).
std::unique_ptr<HNSWIndex> buildIndex(size_t numVectors, std::mt19937& rng) {
    HNSWConfig config;
    config.dimension = kDim;
    config.maxConnections = 16;
    config.maxConnectionsLayer0 = 32;
    config.efConstruction = 100;
    config.efSearch = 50;
    auto index = std::make_unique<HNSWIndex>(config);

    for (size_t i = 1; i <= numVectors; ++i) {
        auto vec = randomVector(rng, kDim);
        index->insert(static_cast<uint64_t>(i), vec);
    }
    return index;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// BM1: Insert throughput
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWInsert(benchmark::State& state) {
    std::mt19937 rng(42);
    HNSWConfig config;
    config.dimension = kDim;
    config.maxConnections = 16;
    config.efConstruction = 100;
    HNSWIndex index(config);

    uint64_t nextId = 1;
    for (auto _ : state) {
        auto vec = randomVector(rng, kDim);
        index.insert(nextId++, vec);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWInsert)->Range(1, 1 << 14);

// ═══════════════════════════════════════════════════════════════════════════
// BM2: Search top-10 latency at various scales
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWSearchTop10(benchmark::State& state) {
    std::mt19937 rng(42);
    auto numVectors = static_cast<size_t>(state.range(0));
    auto index = buildIndex(numVectors, rng);

    auto query = randomVector(rng, kDim);
    for (auto _ : state) {
        benchmark::DoNotOptimize(index->search(query, 10));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWSearchTop10)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000);

// ═══════════════════════════════════════════════════════════════════════════
// BM3: Filtered search
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWFilteredSearch(benchmark::State& state) {
    std::mt19937 rng(42);
    auto index = buildIndex(10000, rng);

    auto query = randomVector(rng, kDim);
    // Filter: accept only even IDs (50% selectivity)
    auto filter = [](uint64_t id) { return id % 2 == 0; };

    for (auto _ : state) {
        benchmark::DoNotOptimize(index->searchWithFilter(query, 10, filter));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWFilteredSearch);

// ═══════════════════════════════════════════════════════════════════════════
// BM4: Remove (soft delete) throughput
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWRemove(benchmark::State& state) {
    std::mt19937 rng(42);
    auto index = buildIndex(10000, rng);

    uint64_t nextRemove = 1;
    for (auto _ : state) {
        state.PauseTiming();
        if (nextRemove > 10000) {
            index = buildIndex(10000, rng);
            nextRemove = 1;
        }
        state.ResumeTiming();
        index->remove(nextRemove++);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWRemove);

// ═══════════════════════════════════════════════════════════════════════════
// BM5: Compact (V2) — rebuild graph after deletions
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWCompact(benchmark::State& state) {
    std::mt19937 rng(42);
    auto deletionPct = static_cast<size_t>(state.range(0));
    constexpr size_t total = 5000;
    auto deleteCount = total * deletionPct / 100;

    for (auto _ : state) {
        state.PauseTiming();
        auto idx = buildIndex(total, rng);
        for (size_t i = 1; i <= deleteCount; ++i) {
            idx->remove(static_cast<uint64_t>(i));
        }
        state.ResumeTiming();
        idx->compact();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWCompact)->Arg(10)->Arg(30)->Arg(50);

// ═══════════════════════════════════════════════════════════════════════════
// BM6: Search quality degradation under deletions (before compact)
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWSearchWithDeletions(benchmark::State& state) {
    std::mt19937 rng(42);
    auto deletionPct = static_cast<size_t>(state.range(0));
    constexpr size_t total = 10000;
    auto deleteCount = total * deletionPct / 100;

    auto index = buildIndex(total, rng);
    for (size_t i = 1; i <= deleteCount; ++i) {
        index->remove(static_cast<uint64_t>(i));
    }

    auto query = randomVector(rng, kDim);
    for (auto _ : state) {
        benchmark::DoNotOptimize(index->search(query, 10));
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["active"] = static_cast<double>(index->size());
}
BENCHMARK(BM_HNSWSearchWithDeletions)->Arg(0)->Arg(10)->Arg(30)->Arg(50);

// ═══════════════════════════════════════════════════════════════════════════
// BM7: Save/Load roundtrip
// ═══════════════════════════════════════════════════════════════════════════

static void BM_HNSWSaveLoad(benchmark::State& state) {
    std::mt19937 rng(42);
    auto index = buildIndex(5000, rng);
    const std::string tmpPath = "/tmp/amind_hnsw_bench.idx";

    for (auto _ : state) {
        index->save(tmpPath);
        HNSWConfig config;
        config.dimension = kDim;
        HNSWIndex loaded(config);
        loaded.load(tmpPath);
        benchmark::DoNotOptimize(loaded.size());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWSaveLoad);
