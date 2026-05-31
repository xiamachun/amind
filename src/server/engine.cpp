#include "engine.h"
#include "provider/ollama_provider.h"
#include "provider/openai_provider.h"
#include "provider/anthropic_provider.h"
#include "provider/fallback_provider.h"

#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace amind {

Engine::Engine(AppConfig config, std::string config_path)
    : config_(std::move(config)), config_path_(std::move(config_path)),
      var_mgr_(std::make_unique<VariableManager>()) {}

Engine::~Engine() {
    shutdown();
}

Result<void, Error> Engine::init() {
    spdlog::info("Initializing amind engine...");

    // ── Step 0: Register all variables and load values from config ──
    registerVariables();
    var_mgr_->loadFrom(config_);

    // Register all provider implementations
    registerOllamaProviders();
    registerOpenAIProviders();
    registerAnthropicProviders();

    // Create HTTP connection pool
    HttpConnectionPool::Config pool_config;
    pool_config.maxConnections = static_cast<size_t>(config_.getInt("pool_max_connections", 8));
    pool_config.idleTimeoutMs = static_cast<size_t>(config_.getInt("pool_idle_timeout_ms", 30000));
    pool_config.circuitBreakerThreshold = static_cast<size_t>(config_.getInt("pool_circuit_threshold", 5));
    pool_config.circuitBreakerCooldownMs = static_cast<size_t>(config_.getInt("pool_circuit_cooldown_ms", 10000));
    conn_pool_ = std::make_shared<HttpConnectionPool>(pool_config);

    // Create providers from config
    // Helper: resolve API key — direct value takes priority, then env var
    auto resolveApiKey = [&](const std::string& direct_key, const std::string& env_var_name) -> std::string {
        if (!direct_key.empty()) return direct_key;
        if (env_var_name.empty()) return "";
        const char* val = std::getenv(env_var_name.c_str());
        return val ? std::string(val) : "";
    };

    auto llm_name = config_.get("llm_provider", "ollama");
    auto llm_model = config_.get("llm_model", "qwen3:8b");
    auto llm_host = config_.get("llm_host", "127.0.0.1");
    auto llm_port = config_.getInt("llm_port", 11434);
    auto llm_api_key = resolveApiKey(config_.get("llm_api_key", ""),
                                      config_.get("llm_api_key_env", ""));
    auto llm_base_url = config_.get("llm_base_url", "");

    if (llm_name == "ollama") {
        llm_ = std::make_shared<OllamaLLM>(llm_model, llm_host, llm_port, conn_pool_);
    } else if (llm_name == "anthropic" || llm_name == "claude") {
        // Parse extra headers from config: "Key:Value;Key:Value"
        std::vector<std::pair<std::string, std::string>> extra_headers;
        auto headers_str = config_.get("llm_extra_headers", "");
        if (!headers_str.empty()) {
            size_t pos = 0;
            while (pos < headers_str.size()) {
                auto semi = headers_str.find(';', pos);
                auto part = headers_str.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
                auto colon = part.find(':');
                if (colon != std::string::npos) {
                    extra_headers.emplace_back(part.substr(0, colon), part.substr(colon + 1));
                }
                pos = (semi == std::string::npos) ? headers_str.size() : semi + 1;
            }
        }
        llm_ = std::make_shared<AnthropicLLM>(llm_model, llm_host, llm_port,
                                               llm_api_key, llm_base_url,
                                               conn_pool_, std::move(extra_headers));
        spdlog::info("Using LLM provider: {} model={} host={}:{} base_url={} extra_headers={}",
                     llm_name, llm_model, llm_host, llm_port, llm_base_url,
                     config_.get("llm_extra_headers", "(none)"));
    } else {
        auto& registry = ProviderRegistry::instance();
        llm_ = std::shared_ptr<LLMProvider>(
            registry.createLLM(llm_name, llm_model, llm_host, llm_port,
                               llm_api_key, llm_base_url));
        spdlog::info("Using LLM provider: {} model={} host={}:{} base_url={}",
                     llm_name, llm_model, llm_host, llm_port, llm_base_url);
    }
    if (!llm_) {
        return makeError(Error::ConfigError, "failed to create LLM provider: " + llm_name);
    }

    // ── Multi-level LLM fallback ──
    // If llm_fallback_keys is configured (semicolon-separated API keys),
    // wrap the primary LLM in a FallbackLLMProvider for automatic failover.
    auto fallback_keys_str = config_.get("llm_fallback_keys", "");
    if (!fallback_keys_str.empty() && (llm_name == "anthropic" || llm_name == "claude")) {
        std::vector<FallbackLLMProvider::Level> levels;
        levels.push_back({llm_, "L1-primary"});

        // Parse extra headers (reuse from primary)
        std::vector<std::pair<std::string, std::string>> extra_headers;
        auto headers_str = config_.get("llm_extra_headers", "");
        if (!headers_str.empty()) {
            size_t pos = 0;
            while (pos < headers_str.size()) {
                auto semi = headers_str.find(';', pos);
                auto part = headers_str.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
                auto colon = part.find(':');
                if (colon != std::string::npos) {
                    extra_headers.emplace_back(part.substr(0, colon), part.substr(colon + 1));
                }
                pos = (semi == std::string::npos) ? headers_str.size() : semi + 1;
            }
        }

        // Create one provider per fallback key
        size_t level_num = 2;
        size_t kpos = 0;
        while (kpos < fallback_keys_str.size()) {
            auto sep = fallback_keys_str.find(';', kpos);
            auto key = fallback_keys_str.substr(kpos, sep == std::string::npos ? std::string::npos : sep - kpos);
            kpos = (sep == std::string::npos) ? fallback_keys_str.size() : sep + 1;
            if (key.empty()) continue;

            auto fallback_llm = std::make_shared<AnthropicLLM>(
                llm_model, llm_host, llm_port, key, llm_base_url,
                conn_pool_, extra_headers);
            levels.push_back({fallback_llm, "L" + std::to_string(level_num++)});
        }

        if (levels.size() > 1) {
            size_t num_levels = levels.size();
            auto cooldown_s = config_.getInt("llm_fallback_cooldown", 30);
            auto threshold = static_cast<size_t>(config_.getInt("llm_fallback_threshold", 3));
            llm_ = std::make_shared<FallbackLLMProvider>(
                std::move(levels), threshold, std::chrono::seconds(cooldown_s));
            spdlog::info("LLM fallback enabled: {} levels, threshold={}, cooldown={}s",
                         num_levels, threshold, cooldown_s);
        }
    }

    auto embed_name = config_.get("embed_provider", "ollama");
    auto embed_model = config_.get("embed_model", "qwen3-embedding");
    auto embed_host = config_.get("embed_host", "127.0.0.1");
    auto embed_port = config_.getInt("embed_port", 11434);
    auto embed_dim = static_cast<size_t>(config_.getInt("embedding_dimension", 4096));
    auto embed_api_key = resolveApiKey(config_.get("embed_api_key", ""),
                                       config_.get("embed_api_key_env", ""));
    auto embed_base_url = config_.get("embed_base_url", "");

    if (embed_name == "ollama") {
        embedder_ = std::make_shared<OllamaEmbed>(embed_model, embed_host, embed_port, embed_dim, conn_pool_);
    } else {
        auto& registry = ProviderRegistry::instance();
        embedder_ = std::shared_ptr<EmbedProvider>(
            registry.createEmbed(embed_name, embed_model, embed_host, embed_port,
                                 embed_api_key, embed_base_url, embed_dim));
        spdlog::info("Using Embed provider: {} model={} host={}:{} base_url={}",
                     embed_name, embed_model, embed_host, embed_port, embed_base_url);
    }
    if (!embedder_) {
        return makeError(Error::ConfigError, "failed to create Embed provider: " + embed_name);
    }

    // Memory store
    MemoryStore::Config store_config;
    store_config.data_dir = config_.get("data_dir", "./amind_data");
    store_config.embedding_dim = embed_dim;
    store_config.conflict_similarity_threshold = config_.getFloat("conflict_similarity_threshold", 0.85f);
    store_config.stale_threshold_hours = static_cast<uint32_t>(config_.getInt("stale_threshold_hours", 720));
    store_config.decay_rate = config_.getFloat("decay_rate", 0.01f);
    store_config.importance_boost = config_.getFloat("importance_boost", 0.1f);
    store_config.hnsw_max_connections = static_cast<size_t>(config_.getInt("hnsw_max_connections", 16));
    store_config.hnsw_ef_construction = static_cast<size_t>(config_.getInt("hnsw_ef_construction", 200));
    store_config.hnsw_hot_capacity = static_cast<size_t>(config_.getInt("hot_capacity", 50000));
    store_config.hnsw_cold_tier_path = config_.get("cold_tier_path", "");
    store_config.hnsw_eviction_ratio = config_.getFloat("eviction_ratio", 0.1f);
    store_config.hnsw_save_interval = static_cast<uint32_t>(config_.getInt("hnsw_save_interval", 60));
    store_config.lsm_flush_interval = static_cast<uint32_t>(config_.getInt("lsm_flush_interval", 30));

    memory_store_ = std::make_unique<MemoryStore>(store_config);
    auto init_result = memory_store_->init();
    if (!init_result.ok()) return init_result;

    // Graph store
    auto graph_data_dir = store_config.data_dir + "/graph";
    graph_store_ = std::make_unique<GraphStore>(graph_data_dir);
    graph_store_->recover();

    // Async pipeline
    auto queue_capacity = static_cast<size_t>(config_.getInt("async_queue_capacity", 10000));
    auto worker_threads = static_cast<size_t>(config_.getInt("async_worker_threads", 4));
    task_queue_ = std::make_unique<TaskQueue>(queue_capacity);
    task_executor_ = std::make_unique<TaskExecutor>(*task_queue_, worker_threads);
    task_executor_->start();

    // ── V2 subsystems (between storage and business logic) ────────────────
    // IMPORTANT: Initialization order matters! Dependencies:
    //   FeatureGate (standalone) → WriteGate (standalone) → LineageIndex (standalone)
    //   → ForgetEngine (standalone) → ConflictResolver (standalone)
    //   → RemoveCoordinator (lineage + forget) → ConsolidationWorker (standalone)
    //   → DerivedExtractor (write_gate + lineage)
    //   → CapturePipeline (feature_gate + derived_extractor)

    // Feature gate (reads [v2] config section)
    V2Config v2_cfg;
    v2_cfg.write_gate_enabled        = config_.get("v2_write_gate_enabled", "false") == "true";
    v2_cfg.lineage_propagation_enabled = config_.get("v2_lineage_propagation_enabled", "false") == "true";
    v2_cfg.forget_score_enabled      = config_.get("v2_forget_score_enabled", "false") == "true";
    v2_cfg.hnsw_compact_enabled      = config_.get("v2_hnsw_compact_enabled", "false") == "true";
    v2_cfg.conflict_resolver_enabled = config_.get("v2_conflict_resolver_enabled", "false") == "true";
    v2_cfg.consolidation_enabled     = config_.get("v2_consolidation_enabled", "false") == "true";
    v2_cfg.reconcile_enabled         = config_.get("v2_reconcile_enabled", "false") == "true";
    v2_cfg.aggregate_staleness_filter_enabled
        = config_.get("v2_aggregate_staleness_filter", "true") == "true";
    v2_cfg.global_shadow_mode        = config_.get("v2_global_shadow_mode", "false") == "true";
    feature_gate_ = std::make_unique<FeatureGate>(v2_cfg);

    // WriteGate
    WriteGateConfig gate_cfg;
    gate_cfg.duplicate_threshold  = config_.getFloat("v2_duplicate_threshold", 0.95f);
    gate_cfg.low_value_threshold  = config_.getFloat("v2_low_value_threshold", 0.15f);
    gate_cfg.deferred_threshold   = config_.getFloat("v2_deferred_threshold", 0.30f);
    gate_cfg.shadow_mode          = v2_cfg.global_shadow_mode;
    write_gate_ = std::make_unique<WriteGate>(gate_cfg);

    // GateLog (audit + resurrect store; no TTL — keeps all rotated WAL forever)
    {
        GateLogConfig glog_cfg;
        glog_cfg.max_file_bytes = static_cast<size_t>(
            config_.getInt("gate_log_max_file_bytes", 100 * 1024 * 1024));
        glog_cfg.max_memory_entries = static_cast<size_t>(
            config_.getInt("gate_log_max_memory_entries", 1000));
        gate_log_ = std::make_unique<GateLog>(
            config_.get("data_dir", "./amind_data"), glog_cfg);
        if (!gate_log_->open()) {
            spdlog::warn("GateLog: failed to open WAL; in-memory only");
        } else {
            gate_log_->replay();
        }

        // Backfill namespace_hash → string reverse map from gate_log so the
        // Forget Log can render namespaces for memories written before this
        // process started.
        if (gate_log_) {
            GateLog::Filter f;
            f.limit = 5000;  // recent ring is enough; capture path keeps it warm
            for (const auto& entry : gate_log_->query(f)) {
                if (!entry.namespace_.empty()) {
                    memory_store_->registerNamespaceString(entry.namespace_);
                }
            }
        }
    }

    // LineageIndex
    lineage_index_ = std::make_unique<LineageIndex>();

    // ForgetEngine
    ForgetConfig forget_cfg;
    forget_cfg.decay_threshold     = config_.getFloat("v2_forget_decay_threshold", 0.3f);
    forget_cfg.archive_threshold   = config_.getFloat("v2_forget_archive_threshold", 0.6f);
    forget_cfg.tombstone_threshold = config_.getFloat("v2_forget_tombstone_threshold", 0.85f);
    forget_cfg.gc_interval_seconds = static_cast<uint32_t>(config_.getInt("v2_forget_gc_interval", 3600));
    forget_cfg.sample_ratio        = config_.getFloat("v2_forget_sample_ratio", 0.10f);
    forget_cfg.shadow_mode         = v2_cfg.global_shadow_mode;
    // Use the data_dir overload so the persistent ForgetLog (forget.log JSONL,
    // 10MB rotation x5) is opened — this is what the /v1/forget/log endpoint
    // and the WebUI ForgetLog page read from.
    forget_engine_ = std::make_unique<ForgetEngine>(
        forget_cfg, store_config.data_dir);

    // ConflictResolver
    ConflictResolverConfig conflict_cfg;
    conflict_cfg.enable_llm_judge = config_.get("v2_conflict_llm_judge", "false") == "true";
    conflict_resolver_ = std::make_unique<ConflictResolver>(conflict_cfg);

    // RemoveCoordinator (depends on lineage + forget)
    remove_coordinator_ = std::make_unique<RemoveCoordinator>(*lineage_index_, *forget_engine_);

    // ConsolidationWorker
    ConsolidationConfig consol_cfg;
    consol_cfg.enabled = v2_cfg.consolidation_enabled;
    consol_cfg.interval_seconds = static_cast<uint32_t>(config_.getInt("v2_consolidation_interval", 3600));
    consol_cfg.dedup_threshold = config_.getFloat("v2_consolidation_dedup_threshold", 0.95f);
    consol_cfg.drift_threshold = config_.getFloat("v2_consolidation_drift_threshold", 0.3f);
    consolidation_worker_ = std::make_unique<ConsolidationWorker>(consol_cfg);

    // DerivedExtractor (depends on write_gate + lineage)
    derived_extractor_ = std::make_unique<DerivedExtractor>(*write_gate_, *lineage_index_);

    // Reconciler (optional; only created when feature flag is on, and only
    // when an LLM provider is available).
    if (v2_cfg.reconcile_enabled && llm_) {
        Reconciler::Config rc_cfg;
        rc_cfg.similarity_floor = config_.getFloat("v2_reconcile_similarity_floor", 0.70f);
        rc_cfg.max_neighbours   = static_cast<size_t>(
            config_.getInt("v2_reconcile_max_neighbours", 3));
        reconciler_ = std::make_unique<Reconciler>(llm_, rc_cfg);
        spdlog::info("Reconciler enabled (sim_floor={}, max_neighbours={})",
                     rc_cfg.similarity_floor, rc_cfg.max_neighbours);
    }

    spdlog::info("V2 subsystems initialized (feature_gate={} features enabled, shadow={})",
                 feature_gate_->enabledCount(), v2_cfg.global_shadow_mode ? "on" : "off");

    // Business logic layers
    capture_ = std::make_unique<CapturePipeline>(
        *memory_store_, *graph_store_, *task_queue_, llm_, embedder_,
        feature_gate_.get(), derived_extractor_.get(), write_gate_.get());
    capture_->setGateLog(gate_log_.get());
    capture_->setReconciler(reconciler_.get());

    RetrievalWeights weights;
    weights.semantic = config_.getFloat("semantic_weight", 0.4f);
    weights.keyword = config_.getFloat("keyword_weight", 0.25f);
    weights.graph = config_.getFloat("graph_weight", 0.15f);
    weights.recency = config_.getFloat("recency_weight", 0.1f);
    weights.importance = config_.getFloat("importance_weight", 0.1f);

    retrieval_ = std::make_unique<RetrievalPipeline>(
        *memory_store_, *graph_store_, llm_, embedder_, weights);
    retrieval_->setCapturePipeline(capture_.get());

    // Aggregate staleness filter (V2). Reads its own flag and persists
    // an audit log of every filter event next to gate.log / forget.log.
    if (v2_cfg.aggregate_staleness_filter_enabled) {
        AggregateStalenessFilter::Config sf_cfg;
        sf_cfg.enabled = true;
        staleness_filter_ = std::make_unique<AggregateStalenessFilter>(sf_cfg);
        stale_log_ = std::make_unique<StaleLog>(store_config.data_dir);
        stale_log_->open();
        stale_log_->replay();
        retrieval_->setStalenessFilter(staleness_filter_.get(), stale_log_.get());
        spdlog::info("Aggregate staleness filter enabled");
    }

    metacog_ = std::make_unique<MetaCognition>(*memory_store_, *graph_store_);
    session_mgr_ = std::make_unique<SessionManager>(*memory_store_, llm_, *capture_,
                                                     store_config.data_dir);
    backup_mgr_ = std::make_unique<BackupManager>(*memory_store_, *graph_store_);
    auth_mgr_ = std::make_unique<AuthManager>(*memory_store_);

    // ── Register onChange callbacks AFTER subsystems are created ──
    registerCallbacks();

    // Start background scheduled tasks (ForgetEngine GC, Consolidation)
    startScheduledTasks();

    spdlog::info("amind engine initialized (pool_size={}, variables={})",
                 pool_config.maxConnections, var_mgr_->list().size());
    return Result<void, Error>();
}

void Engine::shutdown() {
    spdlog::info("Shutting down amind engine...");
    scheduler_running_ = false;
    if (forget_thread_.joinable()) forget_thread_.join();
    if (consolidation_thread_.joinable()) consolidation_thread_.join();
    if (task_executor_) task_executor_->shutdown();
    if (graph_store_) {
        graph_store_->checkpoint();
        graph_store_->flush();
    }
    if (memory_store_) memory_store_->flush();
    spdlog::info("amind engine shutdown complete");
}

// ── Background Schedulers ─────────────────────────────────────────────────

void Engine::startScheduledTasks() {
    scheduler_running_ = true;

    // ForgetEngine GC cycle (only if feature enabled and shadow mode off)
    if (feature_gate_->isForgetScoreEnabled()) {
        forget_thread_ = std::thread(&Engine::forgetCycleLoop, this);
        spdlog::info("ForgetEngine scheduler started (interval={}s, shadow={})",
                     forget_engine_->config().gc_interval_seconds,
                     forget_engine_->isShadowMode() ? "on" : "off");
    }

    // Consolidation worker (only if feature enabled)
    if (feature_gate_->isConsolidationEnabled()) {
        consolidation_thread_ = std::thread(&Engine::consolidationCycleLoop, this);
        spdlog::info("ConsolidationWorker scheduler started (interval={}s)",
                     consolidation_worker_->config().interval_seconds);
    }
}

Engine::ForgetCycleResult Engine::runForgetCycleOnce() {
    ForgetCycleResult out{};

    // Sample a fraction of memories and compute signals
    std::vector<std::pair<uint64_t, ForgetSignals>> batch;
    uint32_t now = MemoryRecord::currentTimeSec();
    float stale_hours = forget_engine_->config().stale_hours;

    memory_store_->scanAll([&](const MemoryRecord& rec) {
        if (!rec.isAlive()) return;
        // Sample based on configured ratio
        if ((rec.memory_id % 100) >= static_cast<uint64_t>(
                forget_engine_->config().sample_ratio * 100)) return;

        ForgetSignals sig;
        float hours_since_access = (now - rec.last_accessed) / 3600.0f;
        sig.staleness = std::min(1.0f, hours_since_access / stale_hours);
        sig.low_access = std::max(0.0f, 1.0f - std::log(1.0f + rec.access_count) / 5.0f);
        sig.low_importance = 1.0f - rec.importance;
        sig.conflict_penalty = (rec.confidence == Confidence::Conflicted) ? 1.0f : 0.0f;
        sig.verified_bonus = (rec.confidence == Confidence::Verified) ? 1.0f : 0.0f;
        batch.emplace_back(rec.memory_id, sig);
    });

    if (batch.empty()) {
        spdlog::info("ForgetEngine GC cycle: no live memories sampled");
        return out;
    }

    auto results = forget_engine_->runCycle(batch);
    out.evaluated = results.size();

    const bool shadow = forget_engine_->isShadowMode();
    const float decay_thr = forget_engine_->config().decay_threshold;
    const uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // Apply GC actions (unless shadow mode), and audit-log every
    // non-trivial decision. We skip "below all thresholds" no-ops to
    // avoid flooding the log on cold startup.
    constexpr size_t kPreviewMaxBytes = 200;
    for (const auto& eval : results) {
        if (eval.forget_score < decay_thr) continue;

        MemoryPhase before = MemoryPhase::Active;
        std::string ns_str;
        std::string preview;
        if (auto rec = memory_store_->get(eval.memory_id); rec.ok()) {
            before = rec->phase;
            ns_str = memory_store_->namespaceFromHash(rec->namespace_hash);
            // UTF-8-safe truncation: never split a code-point. For amind's
            // typical Chinese content a code-point is ≤4 bytes.
            preview = rec->content;
            if (preview.size() > kPreviewMaxBytes) {
                size_t cut = kPreviewMaxBytes;
                while (cut > 0 && (preview[cut] & 0xC0) == 0x80) --cut;
                preview.resize(cut);
                preview += "…";
            }
        }

        MemoryPhase after = before;
        if (!shadow) {
            if (eval.recommended_action == ForgetLogEntry::Decision::Tombstone) {
                memory_store_->remove(eval.memory_id);
                after = MemoryPhase::Tombstone;
                ++out.actioned;
            }
            // Archive and Decay are logged but not implemented as store ops yet.
        }

        ForgetLogEntry log_entry;
        log_entry.timestamp_ms = now_ms;
        log_entry.memory_id    = eval.memory_id;
        log_entry.decision     = eval.recommended_action;
        log_entry.reason       = eval.reason +
                                 " (forget_score=" + std::to_string(eval.forget_score) +
                                 ", shadow=" + (shadow ? "yes" : "no") + ")";
        log_entry.before_state = before;
        log_entry.after_state  = after;
        log_entry.gc_worker_id = "main";
        log_entry.namespace_   = std::move(ns_str);
        log_entry.content_preview = std::move(preview);
        forget_engine_->logEntry(std::move(log_entry));
        ++out.logged;
    }

    spdlog::info("ForgetEngine GC cycle: evaluated {} memories, logged {}, actioned {}, shadow={}",
                 out.evaluated, out.logged, out.actioned, shadow ? "yes" : "no");
    return out;
}

void Engine::forgetCycleLoop() {
    const auto interval = std::chrono::seconds(
        forget_engine_->config().gc_interval_seconds);

    while (scheduler_running_) {
        std::this_thread::sleep_for(interval);
        if (!scheduler_running_) break;
        try {
            runForgetCycleOnce();
        } catch (const std::exception& e) {
            spdlog::error("ForgetEngine cycle failed: {}", e.what());
        }
    }
}

Engine::ConsolidationCycleResult Engine::runConsolidationCycleOnce() {
    ConsolidationCycleResult out{};

    const bool incremental = (config_.get("v2_consolidation_mode", "incremental") == "incremental");
    const float dedup_threshold = consolidation_worker_->config().dedup_threshold;
    const uint32_t sampled_full_every = static_cast<uint32_t>(
        config_.getInt("v2_consolidation_sampled_full_every", 24));
    const float sampled_full_ratio = config_.getFloat("v2_consolidation_sampled_full_ratio", 0.05f);

    uint32_t last_cycle_time;
    uint32_t cycle_count;
    {
        std::lock_guard<std::mutex> lk(consolidation_state_mutex_);
        if (consolidation_last_cycle_time_ == 0) {
            consolidation_last_cycle_time_ = MemoryRecord::currentTimeSec();
        }
        last_cycle_time = consolidation_last_cycle_time_;
        cycle_count = ++consolidation_cycle_count_;
    }

    uint32_t now = MemoryRecord::currentTimeSec();
    size_t dedup_count = 0;
    size_t checked_count = 0;

    if (!incremental) {
        // ── Full mode: O(n²) — only for small stores or first-time cleanup ──
        std::vector<MemoryRecord> all_alive;
        memory_store_->scanAll([&](const MemoryRecord& rec) {
            if (rec.isAlive()) all_alive.push_back(rec);
        });
        checked_count = all_alive.size();

        if (!all_alive.empty()) {
            auto cosine_sim = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
                if (a.size() != b.size() || a.empty()) return 0.0f;
                float dot = 0, na = 0, nb = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                float denom = std::sqrt(na) * std::sqrt(nb);
                return denom > 0 ? dot / denom : 0.0f;
            };

            auto dedup_results = consolidation_worker_->dedup(all_alive, cosine_sim);
            for (const auto& dr : dedup_results) {
                for (uint64_t archived_id : dr.archived_ids) {
                    if (!feature_gate_->isGlobalShadowMode()) {
                        memory_store_->remove(archived_id);
                    }
                    dedup_count++;
                }
            }
            spdlog::info("Consolidation (full): {} alive, {} deduped",
                         all_alive.size(), dedup_count);
        }
    } else {
        // ── Incremental mode with periodic sampled-full ──
        bool is_sampled_full = (sampled_full_every > 0)
                               && (cycle_count % sampled_full_every == 0);
        out.sampled_full = is_sampled_full;

        // Collect records to check
        std::vector<MemoryRecord> to_check;
        if (is_sampled_full) {
            // Sampled-full: random fraction of ALL alive memories
            uint32_t sample_mod = static_cast<uint32_t>(
                std::max(1.0f, 1.0f / sampled_full_ratio));
            memory_store_->scanAll([&](const MemoryRecord& rec) {
                if (!rec.isAlive()) return;
                if (rec.embedding.empty()) return;
                if ((rec.memory_id % sample_mod) == 0) {
                    to_check.push_back(rec);
                }
            });
            spdlog::info("Consolidation: sampled-full cycle #{} — sampling {} records (ratio={:.2f})",
                         cycle_count, to_check.size(), sampled_full_ratio);
        } else {
            // Incremental: only memories created since last cycle
            memory_store_->scanAll([&](const MemoryRecord& rec) {
                if (!rec.isAlive()) return;
                if (rec.embedding.empty()) return;
                if (rec.created_at >= last_cycle_time) {
                    to_check.push_back(rec);
                }
            });
        }
        checked_count = to_check.size();

        // Dedup each record against HNSW neighbors
        for (const auto& rec : to_check) {
            auto neighbors = memory_store_->searchSimilar(rec.embedding, 5);
            for (const auto& [nid, sim] : neighbors) {
                if (nid == rec.memory_id) continue;
                if (sim < dedup_threshold) continue;
                auto nrec = memory_store_->get(nid);
                if (!nrec.ok() || !nrec->isAlive()) continue;
                uint64_t to_archive = (rec.importance >= nrec->importance)
                                      ? nid : rec.memory_id;
                if (!feature_gate_->isGlobalShadowMode()) {
                    memory_store_->remove(to_archive);
                }
                dedup_count++;
                break;  // one dedup per record per cycle
            }
        }

        spdlog::info("Consolidation ({}): checked {}, deduped {}",
                     is_sampled_full ? "sampled-full" : "incremental",
                     to_check.size(), dedup_count);
    }

    {
        std::lock_guard<std::mutex> lk(consolidation_state_mutex_);
        consolidation_last_cycle_time_ = now;
    }

    out.checked = checked_count;
    out.deduped = dedup_count;
    return out;
}

void Engine::consolidationCycleLoop() {
    const auto interval = std::chrono::seconds(
        consolidation_worker_->config().interval_seconds);

    while (scheduler_running_) {
        std::this_thread::sleep_for(interval);
        if (!scheduler_running_) break;
        try {
            runConsolidationCycleOnce();
        } catch (const std::exception& e) {
            spdlog::error("Consolidation cycle failed: {}", e.what());
        }
    }
}

// ── Variable Registration (all 35 parameters) ────────────────────────────

void Engine::registerVariables() {
    using T = VarType;
    using M = VarMode;
    auto& vm = *var_mgr_;

    // Server (READONLY)
    vm.registerVar("host",               T::STRING, M::READONLY, "0.0.0.0",     "Bind address for REST API", "server");
    vm.registerVar("port",               T::INT,    M::READONLY, "8080",         "REST API port", "server");
    vm.registerVar("max_connections",    T::INT,    M::READONLY, "128",          "Listen backlog", "server");
    vm.registerVar("request_timeout_ms", T::INT,    M::READONLY, "30000",        "Per-request timeout (ms)", "server");

    // WebUI (READONLY)
    vm.registerVar("webui_enabled",      T::BOOL,   M::READONLY, "true",         "Enable WebUI Dashboard", "webui");
    vm.registerVar("webui_port",         T::INT,    M::READONLY, "11011",        "WebUI listen port", "webui");
    vm.registerVar("webui_root",         T::STRING, M::READONLY, "web/dist",     "WebUI static file root", "webui");

    // LLM Provider (READONLY)
    vm.registerVar("llm_provider",       T::STRING, M::READONLY, "ollama",       "LLM provider name", "provider");
    vm.registerVar("llm_model",          T::STRING, M::READONLY, "qwen3:8b",     "LLM model name", "provider");
    vm.registerVar("llm_host",           T::STRING, M::READONLY, "127.0.0.1",    "LLM provider host", "provider");
    vm.registerVar("llm_port",           T::INT,    M::READONLY, "11434",        "LLM provider port", "provider");
    vm.registerVar("llm_api_key_env",    T::STRING, M::READONLY, "OPENAI_API_KEY", "(Reserved) API key env var", "provider");
    vm.registerVar("llm_base_url",       T::STRING, M::READONLY, "",             "(Reserved) Custom base URL", "provider");

    // Embed Provider (READONLY)
    vm.registerVar("embed_provider",     T::STRING, M::READONLY, "ollama",       "Embedding provider name", "provider");
    vm.registerVar("embed_model",        T::STRING, M::READONLY, "qwen3-embedding", "Embedding model name", "provider");
    vm.registerVar("embed_host",         T::STRING, M::READONLY, "127.0.0.1",    "Embedding provider host", "provider");
    vm.registerVar("embed_port",         T::INT,    M::READONLY, "11434",        "Embedding provider port", "provider");
    vm.registerVar("embed_api_key_env",  T::STRING, M::READONLY, "OPENAI_API_KEY", "(Reserved) API key env var", "provider");
    vm.registerVar("embed_base_url",     T::STRING, M::READONLY, "",             "(Reserved) Custom base URL", "provider");
    vm.registerVar("embedding_dimension",T::INT,    M::READONLY, "4096",         "Embedding vector dimension", "provider");

    // Reranker (READONLY)
    vm.registerVar("rerank_provider",    T::STRING, M::READONLY, "none",         "(Reserved) Reranker provider", "provider");
    vm.registerVar("rerank_model",       T::STRING, M::READONLY, "",             "(Reserved) Reranker model", "provider");

    // Connection Pool (mixed)
    vm.registerVar("pool_max_connections",      T::INT, M::READONLY, "8",        "Max pooled connections", "pool");
    vm.registerVar("pool_idle_timeout_ms",      T::INT, M::DYNAMIC, "30000",     "Idle connection timeout (ms)", "pool");
    vm.registerVar("pool_circuit_threshold",    T::INT, M::DYNAMIC, "5",         "Circuit breaker failure threshold", "pool");
    vm.registerVar("pool_circuit_cooldown_ms",  T::INT, M::DYNAMIC, "10000",     "Circuit breaker cooldown (ms)", "pool");

    // HNSW (READONLY — structural)
    vm.registerVar("hnsw_max_connections",  T::INT,   M::READONLY, "16",   "Max HNSW node connections", "hnsw");
    vm.registerVar("hnsw_ef_construction",  T::INT,   M::READONLY, "200",  "Build-time search width", "hnsw");

    // Hot/Cold (DYNAMIC)
    vm.registerVar("hot_capacity",       T::INT,    M::DYNAMIC, "50000",   "Max vectors in hot HNSW layer", "hnsw");
    vm.registerVar("cold_tier_path",     T::STRING, M::READONLY, "",        "Cold tier storage path", "hnsw");
    vm.registerVar("eviction_ratio",     T::FLOAT,  M::DYNAMIC, "0.1",     "Hot layer eviction ratio", "hnsw");

    // Memory Engine (DYNAMIC)
    vm.registerVar("decay_rate",                    T::FLOAT, M::DYNAMIC, "0.01",  "Temporal importance decay rate", "memory");
    vm.registerVar("importance_boost",              T::FLOAT, M::DYNAMIC, "0.1",   "Importance boost on access", "memory");
    vm.registerVar("stale_threshold_hours",         T::INT,   M::DYNAMIC, "720",   "Hours before memory is stale", "memory");
    vm.registerVar("conflict_similarity_threshold", T::FLOAT, M::DYNAMIC, "0.85",  "Cosine similarity for conflict detection", "memory");

    // Retrieval Weights (DYNAMIC)
    vm.registerVar("semantic_weight",    T::FLOAT, M::DYNAMIC, "0.4",   "Semantic similarity weight", "retrieval");
    vm.registerVar("keyword_weight",     T::FLOAT, M::DYNAMIC, "0.25",  "Keyword match weight", "retrieval");
    vm.registerVar("graph_weight",       T::FLOAT, M::DYNAMIC, "0.15",  "Graph proximity weight", "retrieval");
    vm.registerVar("recency_weight",     T::FLOAT, M::DYNAMIC, "0.1",   "Recency freshness weight", "retrieval");
    vm.registerVar("importance_weight",  T::FLOAT, M::DYNAMIC, "0.1",   "Importance score weight", "retrieval");

    // Async (mixed)
    vm.registerVar("async_worker_threads",  T::INT, M::READONLY, "4",     "Async task worker threads", "async");
    vm.registerVar("async_queue_capacity",  T::INT, M::DYNAMIC,  "10000", "Max queued async tasks", "async");

    // Session (READONLY — reserved)
    vm.registerVar("max_recent_turns",          T::INT, M::READONLY, "10",  "(Reserved) Max recent turns per session", "session");
    vm.registerVar("initial_memory_load_count", T::INT, M::READONLY, "20",  "(Reserved) Memories pre-loaded on session start", "session");

    // Data & Logging (mixed)
    vm.registerVar("data_dir",           T::STRING, M::READONLY, "./amind_data", "Root data directory", "storage");
    vm.registerVar("log_level",          T::STRING, M::DYNAMIC,  "info",         "Log verbosity (debug/info/warn/error)", "storage");

    // LSM (mixed)
    vm.registerVar("lsm_flush_interval",     T::INT, M::DYNAMIC,  "30",         "LSM flush interval (seconds)", "storage");
    vm.registerVar("wal_max_size",           T::INT, M::READONLY, "104857600",   "(Reserved) Max WAL size bytes", "storage");
    vm.registerVar("wal_expiry_seconds",     T::INT, M::READONLY, "3600",        "(Reserved) WAL entry expiry", "storage");
    vm.registerVar("sst_target_size",        T::INT, M::READONLY, "4194304",     "(Reserved) SSTable target size", "storage");
    vm.registerVar("l0_compaction_threshold",T::INT, M::READONLY, "4",           "(Reserved) L0 compaction threshold", "storage");
    vm.registerVar("level_size_multiplier",  T::INT, M::READONLY, "10",          "(Reserved) LSM level size multiplier", "storage");

    // HNSW persistence
    vm.registerVar("hnsw_save_interval", T::INT,  M::DYNAMIC, "60",   "HNSW index save interval (seconds)", "storage");
    vm.registerVar("sst_use_mmap",       T::BOOL, M::READONLY, "true", "(Reserved) Use mmap for SSTable reads", "storage");

    // ── V2 Feature Gates (DYNAMIC — runtime switchable) ──────────────────
    vm.registerVar("v2_write_gate_enabled",          T::BOOL,  M::DYNAMIC, "false", "Enable WriteGate quality filter", "v2");
    vm.registerVar("v2_lineage_propagation_enabled", T::BOOL,  M::DYNAMIC, "false", "Enable lineage cascade invalidation", "v2");
    vm.registerVar("v2_forget_score_enabled",        T::BOOL,  M::DYNAMIC, "false", "Enable forget score GC", "v2");
    vm.registerVar("v2_hnsw_compact_enabled",        T::BOOL,  M::DYNAMIC, "false", "Enable HNSW compaction", "v2");
    vm.registerVar("v2_conflict_resolver_enabled",   T::BOOL,  M::DYNAMIC, "false", "Enable conflict auto-resolution", "v2");
    vm.registerVar("v2_consolidation_enabled",       T::BOOL,  M::DYNAMIC, "false", "Enable consolidation worker", "v2");
    vm.registerVar("v2_global_shadow_mode",          T::BOOL,  M::DYNAMIC, "false", "V2 shadow mode (compute but don't enforce)", "v2");

    // V2 WriteGate params (DYNAMIC)
    vm.registerVar("v2_duplicate_threshold",         T::FLOAT, M::DYNAMIC, "0.95",  "WriteGate duplicate cosine threshold", "v2");
    vm.registerVar("v2_low_value_threshold",         T::FLOAT, M::DYNAMIC, "0.15",  "WriteGate low-value rejection threshold", "v2");
    vm.registerVar("v2_deferred_threshold",          T::FLOAT, M::DYNAMIC, "0.30",  "WriteGate deferred threshold", "v2");

    // V2 ForgetEngine params (DYNAMIC)
    vm.registerVar("v2_forget_decay_threshold",      T::FLOAT, M::DYNAMIC, "0.3",   "Forget score → Decay threshold", "v2");
    vm.registerVar("v2_forget_archive_threshold",    T::FLOAT, M::DYNAMIC, "0.6",   "Forget score → Archive threshold", "v2");
    vm.registerVar("v2_forget_tombstone_threshold",  T::FLOAT, M::DYNAMIC, "0.85",  "Forget score → Tombstone threshold", "v2");

    // V2 ConflictResolver (READONLY)
    vm.registerVar("v2_conflict_llm_judge",          T::BOOL,  M::READONLY, "false", "Enable LLM judge for L5 conflict resolution", "v2");

    // V2 Consolidation params (DYNAMIC)
    vm.registerVar("v2_consolidation_dedup_threshold",  T::FLOAT, M::DYNAMIC, "0.95", "Consolidation dedup cosine threshold", "v2");
    vm.registerVar("v2_consolidation_drift_threshold",  T::FLOAT, M::DYNAMIC, "0.3",  "Consolidation drift invalidation threshold", "v2");
}

void Engine::registerCallbacks() {
    // Memory engine dynamic params
    var_mgr_->onChange("conflict_similarity_threshold", [this](auto&, auto& v) {
        memory_store_->setConflictThreshold(std::stof(v));
    });
    var_mgr_->onChange("decay_rate", [this](auto&, auto& v) {
        memory_store_->setDecayRate(std::stof(v));
    });
    var_mgr_->onChange("importance_boost", [this](auto&, auto& v) {
        memory_store_->setImportanceBoost(std::stof(v));
    });
    var_mgr_->onChange("stale_threshold_hours", [this](auto&, auto& v) {
        memory_store_->setStaleThresholdHours(static_cast<uint32_t>(std::stoi(v)));
    });

    // Retrieval weights
    for (const auto& name : {"semantic_weight", "keyword_weight", "graph_weight",
                              "recency_weight", "importance_weight"}) {
        std::string short_name(name);
        short_name = short_name.substr(0, short_name.find("_weight"));
        var_mgr_->onChange(name, [this, short_name](auto&, auto& v) {
            retrieval_->setWeight(short_name, std::stof(v));
        });
    }

    // Connection pool
    var_mgr_->onChange("pool_idle_timeout_ms", [this](auto&, auto& v) {
        conn_pool_->setIdleTimeout(static_cast<size_t>(std::stoi(v)));
    });
    var_mgr_->onChange("pool_circuit_threshold", [this](auto&, auto& v) {
        conn_pool_->setCircuitThreshold(static_cast<size_t>(std::stoi(v)));
    });
    var_mgr_->onChange("pool_circuit_cooldown_ms", [this](auto&, auto& v) {
        conn_pool_->setCircuitCooldown(static_cast<size_t>(std::stoi(v)));
    });

    // Log level
    var_mgr_->onChange("log_level", [](auto&, auto& v) {
        if (v == "debug") spdlog::set_level(spdlog::level::debug);
        else if (v == "info") spdlog::set_level(spdlog::level::info);
        else if (v == "warn") spdlog::set_level(spdlog::level::warn);
        else if (v == "error") spdlog::set_level(spdlog::level::err);
        spdlog::info("Log level changed to: {}", v);
    });

    // ── V2 feature gate callbacks ────────────────────────────────────────
    var_mgr_->onChange("v2_write_gate_enabled", [this](auto&, auto& v) {
        feature_gate_->setWriteGateEnabled(v == "true");
    });
    var_mgr_->onChange("v2_lineage_propagation_enabled", [this](auto&, auto& v) {
        feature_gate_->setLineagePropagationEnabled(v == "true");
    });
    var_mgr_->onChange("v2_forget_score_enabled", [this](auto&, auto& v) {
        feature_gate_->setForgetScoreEnabled(v == "true");
    });
    var_mgr_->onChange("v2_hnsw_compact_enabled", [this](auto&, auto& v) {
        feature_gate_->setHnswCompactEnabled(v == "true");
    });
    var_mgr_->onChange("v2_conflict_resolver_enabled", [this](auto&, auto& v) {
        feature_gate_->setConflictResolverEnabled(v == "true");
    });
    var_mgr_->onChange("v2_consolidation_enabled", [this](auto&, auto& v) {
        feature_gate_->setConsolidationEnabled(v == "true");
    });
    var_mgr_->onChange("v2_global_shadow_mode", [this](auto&, auto& v) {
        bool shadow = (v == "true");
        feature_gate_->setGlobalShadowMode(shadow);
        write_gate_->setShadowMode(shadow);
        forget_engine_->setShadowMode(shadow);
        spdlog::info("V2 global shadow mode: {}", shadow ? "on" : "off");
    });

    // V2 WriteGate param callbacks
    var_mgr_->onChange("v2_duplicate_threshold", [this](auto&, auto& v) {
        auto cfg = write_gate_->config();
        cfg.duplicate_threshold = std::stof(v);
        write_gate_->setConfig(cfg);
    });
    var_mgr_->onChange("v2_low_value_threshold", [this](auto&, auto& v) {
        auto cfg = write_gate_->config();
        cfg.low_value_threshold = std::stof(v);
        write_gate_->setConfig(cfg);
    });
    var_mgr_->onChange("v2_deferred_threshold", [this](auto&, auto& v) {
        auto cfg = write_gate_->config();
        cfg.deferred_threshold = std::stof(v);
        write_gate_->setConfig(cfg);
    });
}

}  // namespace amind
