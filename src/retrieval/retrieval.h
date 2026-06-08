#pragma once

#include "core/memory_record.h"
#include "core/result.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"
#include "provider/provider.h"
#include "staleness_filter.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace amind { class CapturePipeline; }  // forward declaration

namespace amind {

/// Query intent analyzed from user query.
struct QueryIntent {
    std::string rewritten_query;
    std::vector<std::string> entities;
    MemoryScope preferred_scope{MemoryScope::Private};
    MemoryType preferred_type{MemoryType::UserProfile};
    float urgency{0.5f};
};

/// Scored memory result.
struct ScoredMemory {
    MemoryRecord record;
    float total_score{0.0f};
    float semantic_score{0.0f};
    float keyword_score{0.0f};
    float graph_score{0.0f};
    float recency_score{0.0f};
    std::vector<float> embedding;  // retained for MMR diversity calculation
};

/// Retrieval weights configuration.
struct RetrievalWeights {
    float semantic{0.4f};
    float keyword{0.25f};
    float graph{0.15f};
    float recency{0.1f};
    float importance{0.1f};

    /// When true, recency acts as a multiplicative gate on the entire score
    /// instead of an additive component. This suppresses very old memories
    /// regardless of semantic match quality.
    bool recency_gate_enabled{false};
};

/// Intent-aware hybrid retrieval pipeline.
/// Query → IntentAnalyzer → HybridRetriever → ResultRanker
class RetrievalPipeline {
public:
    RetrievalPipeline(MemoryStore& store, GraphStore& graph,
                      std::shared_ptr<LLMProvider> llm,
                      std::shared_ptr<EmbedProvider> embedder,
                      RetrievalWeights weights = {});

    /// Full recall pipeline. Intent analysis is OFF by default (pure vector
    /// search is sufficient for most queries and saves one LLM round-trip).
    Result<std::vector<ScoredMemory>> recall(
        const std::string& query, 
        const std::string& agent_id,
        const std::string& user_id,
        size_t top_k = 10, 
        bool analyze_intent = false);

    /// Simple vector similarity search (no intent analysis).
    Result<std::vector<ScoredMemory>> simpleSearch(
        const std::string& query, size_t top_k = 10);

private:
    /// Analyze query intent via LLM.
    Result<QueryIntent> analyzeIntent(const std::string& query);

    /// Compute recency score for a memory.
    float computeRecencyScore(const MemoryRecord& record) const;

    /// Compute keyword match score between content and query terms.
    float computeKeywordScore(const std::string& content,
                              const std::string& query,
                              const std::vector<std::string>& entities) const;

    /// Rank and fuse results with MMR diversity.
    std::vector<ScoredMemory> rankAndFuse(
        std::vector<ScoredMemory>& candidates, size_t top_k);

    MemoryStore& store_;
    GraphStore& graph_;
    std::shared_ptr<LLMProvider> llm_;
    std::shared_ptr<EmbedProvider> embedder_;
    std::shared_ptr<RerankProvider> reranker_;
    RetrievalWeights weights_;
    mutable std::mutex weights_mutex_;

public:
    /// Dynamically update a retrieval weight.
    void setWeight(const std::string& name, float v);
    RetrievalWeights getWeights() const;

    /// Set an optional reranker (called before MMR in rankAndFuse).
    void setReranker(std::shared_ptr<RerankProvider> reranker) { reranker_ = std::move(reranker); }

    /// Set capture pipeline reference for freshness barrier support.
    /// When set, recall() will wait for pending refinements before returning.
    void setCapturePipeline(CapturePipeline* cp) { capture_pipeline_ = cp; }

    /// Optional aggregate-staleness filter (V2). When set, recall results are
    /// post-filtered to drop list-style aggregates rendered stale by newer
    /// atomic facts in the same result set. Both pointers may be null
    /// (filter disabled). The filter does not own the log.
    void setStalenessFilter(const AggregateStalenessFilter* filter,
                            MemoryEventLog* events) {
        staleness_filter_ = filter;
        events_log_ = events;
    }

private:
    CapturePipeline* capture_pipeline_{nullptr};
    const AggregateStalenessFilter* staleness_filter_{nullptr};
    MemoryEventLog* events_log_{nullptr};
};

}  // namespace amind
