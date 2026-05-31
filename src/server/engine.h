#pragma once

#include "config.h"
#include "core/result.h"
#include "retrieval/staleness_filter.h"
#include "memory/memory_store.h"
#include "graph/graph_store.h"
#include "async/task_queue.h"
#include "async/task_executor.h"
#include "provider/provider.h"
#include "provider/connection_pool.h"
#include "capture/capture_pipeline.h"
#include "retrieval/retrieval.h"
#include "metacognition/metacognition.h"
#include "session/session_manager.h"
#include "memory/backup.h"
#include "variable_manager.h"
#include "config/v2_config.h"
#include "gate/write_gate.h"
#include "gate_log/gate_log.h"
#include "lineage/lineage_index.h"
#include "forget/forget_engine.h"
#include "conflict/conflict_resolver.h"
#include "coordinator/remove_coordinator.h"
#include "consolidation/consolidation_worker.h"
#include "capture/derived_extractor.h"
#include "reconcile/reconciler.h"
#include "auth/auth_manager.h"

#include <atomic>
#include <mutex>
#include <memory>
#include <thread>

namespace amind {

/// The engine is the central orchestrator that owns all subsystems.
/// Created once at startup, passed to the REST server.
class Engine {
public:
    explicit Engine(AppConfig config, std::string config_path = "amind.conf");
    ~Engine();

    /// Initialize all subsystems.
    Result<void, Error> init();

    /// Graceful shutdown.
    void shutdown();

    // Accessors for subsystems
    MemoryStore& memoryStore() { return *memory_store_; }
    GraphStore& graphStore() { return *graph_store_; }
    CapturePipeline& capturePipeline() { return *capture_; }
    RetrievalPipeline& retrievalPipeline() { return *retrieval_; }
    MetaCognition& metaCognition() { return *metacog_; }
    SessionManager& sessionManager() { return *session_mgr_; }
    BackupManager& backupManager() { return *backup_mgr_; }
    HttpConnectionPool& connectionPool() { return *conn_pool_; }

    // V2 subsystem accessors
    FeatureGate& featureGate() { return *feature_gate_; }
    WriteGate& writeGate() { return *write_gate_; }
    GateLog& gateLog() { return *gate_log_; }
    LineageIndex& lineageIndex() { return *lineage_index_; }
    ForgetEngine& forgetEngine() { return *forget_engine_; }
    ConflictResolver& conflictResolver() { return *conflict_resolver_; }
    RemoveCoordinator& removeCoordinator() { return *remove_coordinator_; }
    ConsolidationWorker& consolidationWorker() { return *consolidation_worker_; }
    DerivedExtractor& derivedExtractor() { return *derived_extractor_; }
    Reconciler* reconciler() { return reconciler_.get(); }   // optional, may be null
    AggregateStalenessFilter* stalenessFilter() { return staleness_filter_.get(); }  // optional
    StaleLog* staleLog() { return stale_log_.get(); }       // optional
    AuthManager& authManager() { return *auth_mgr_; }

    TaskQueue& taskQueue() { return *task_queue_; }
    TaskExecutor& taskExecutor() { return *task_executor_; }

    const AppConfig& config() const { return config_; }
    VariableManager& variableManager() { return *var_mgr_; }
    const std::string& configPath() const { return config_path_; }

private:
    void registerVariables();
    void registerCallbacks();
    void startScheduledTasks();
    void forgetCycleLoop();
    void consolidationCycleLoop();

public:
    /// Run one ForgetEngine GC cycle synchronously. Returns the count of
    /// evaluations and the count of memories that were actually modified
    /// (zero in shadow mode). Safe to call regardless of feature_gate state.
    struct ForgetCycleResult { size_t evaluated; size_t logged; size_t actioned; };
    ForgetCycleResult runForgetCycleOnce();

    /// Run one Consolidation cycle synchronously. Returns checked + deduped.
    struct ConsolidationCycleResult { size_t checked; size_t deduped; bool sampled_full; };
    ConsolidationCycleResult runConsolidationCycleOnce();

private:

    AppConfig config_;
    std::string config_path_;
    std::unique_ptr<VariableManager> var_mgr_;
    std::shared_ptr<HttpConnectionPool> conn_pool_;
    std::unique_ptr<MemoryStore> memory_store_;
    std::unique_ptr<GraphStore> graph_store_;
    std::unique_ptr<TaskQueue> task_queue_;
    std::unique_ptr<TaskExecutor> task_executor_;
    std::shared_ptr<LLMProvider> llm_;
    std::shared_ptr<EmbedProvider> embedder_;
    std::unique_ptr<CapturePipeline> capture_;
    std::unique_ptr<RetrievalPipeline> retrieval_;
    std::unique_ptr<MetaCognition> metacog_;
    std::unique_ptr<SessionManager> session_mgr_;
    std::unique_ptr<BackupManager> backup_mgr_;

    // V2 subsystems
    std::unique_ptr<FeatureGate> feature_gate_;
    std::unique_ptr<WriteGate> write_gate_;
    std::unique_ptr<GateLog> gate_log_;
    std::unique_ptr<LineageIndex> lineage_index_;
    std::unique_ptr<ForgetEngine> forget_engine_;
    std::unique_ptr<ConflictResolver> conflict_resolver_;
    std::unique_ptr<RemoveCoordinator> remove_coordinator_;
    std::unique_ptr<ConsolidationWorker> consolidation_worker_;
    std::unique_ptr<DerivedExtractor> derived_extractor_;
    std::unique_ptr<Reconciler> reconciler_;       // optional (only when v2_reconcile_enabled)
    std::unique_ptr<AggregateStalenessFilter> staleness_filter_;  // optional
    std::unique_ptr<StaleLog> stale_log_;                          // optional
    std::unique_ptr<AuthManager> auth_mgr_;

    // Background scheduler threads
    std::atomic<bool> scheduler_running_{false};
    std::thread forget_thread_;
    std::thread consolidation_thread_;

    // State carried across consolidation cycles (used by both the loop and
    // the manual-trigger admin endpoint).
    std::mutex consolidation_state_mutex_;
    uint32_t consolidation_last_cycle_time_{0};
    uint32_t consolidation_cycle_count_{0};
};

}  // namespace amind
