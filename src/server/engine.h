#pragma once

#include "config.h"
#include "core/result.h"
#include "observability/memory_event_log.h"
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
// gate_log/gate_log.h removed in Phase 4 (replaced by observability/memory_event_log.h).
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
#include <unordered_map>

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

    // ========== Agent Management ==========

    /// Per-agent isolated storage container.
    struct AgentStore {
        std::string agent_id;
        std::string display_name;
        std::string domain;
        std::unique_ptr<MemoryStore> memory;
        std::unique_ptr<GraphStore> graph;
        std::unique_ptr<LineageIndex> lineage;
    };

    /// Register a new agent. Creates its AgentStore and data directories.
    Result<void, Error> registerAgent(const std::string& agent_id,
                                       const std::string& display_name = "",
                                       const std::string& domain = "");

    /// Get an existing AgentStore, or create "default" if not found.
    AgentStore& getAgentStore(const std::string& agent_id);

    /// Get AgentStore pointer (null if not found, does not auto-create).
    AgentStore* findAgentStore(const std::string& agent_id);

    /// List all registered agent IDs.
    std::vector<std::string> listAgents() const;

    /// Remove an agent and all its data.
    Result<void, Error> removeAgent(const std::string& agent_id);

    // ========== Legacy Accessors (delegate to default AgentStore) ==========

    // Accessors for subsystems
    MemoryStore& memoryStore() { return *getAgentStore("default").memory; }
    GraphStore& graphStore() { return *getAgentStore("default").graph; }
    CapturePipeline& capturePipeline() { return *capture_; }
    RetrievalPipeline& retrievalPipeline() { return *retrieval_; }
    MetaCognition& metaCognition() { return *metacog_; }
    SessionManager& sessionManager() { return *session_mgr_; }
    BackupManager& backupManager() { return *backup_mgr_; }
    HttpConnectionPool& connectionPool() { return *conn_pool_; }

    // V2 subsystem accessors
    FeatureGate& featureGate() { return *feature_gate_; }
    WriteGate& writeGate() { return *write_gate_; }
    MemoryEventLog& eventsLog() { return *events_log_; }
    LineageIndex& lineageIndex() { return *getAgentStore("default").lineage; }
    ForgetEngine& forgetEngine() { return *forget_engine_; }
    ConflictResolver& conflictResolver() { return *conflict_resolver_; }
    RemoveCoordinator& removeCoordinator() { return *remove_coordinator_; }
    ConsolidationWorker& consolidationWorker() { return *consolidation_worker_; }
    DerivedExtractor& derivedExtractor() { return *derived_extractor_; }
    Reconciler* reconciler() { return reconciler_.get(); }   // optional, may be null
    AggregateStalenessFilter* stalenessFilter() { return staleness_filter_.get(); }  // optional
    // staleLog() removed in Phase 4 — RecallStale events live in eventsLog().
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
    std::unique_ptr<MemoryEventLog> events_log_;
    std::unique_ptr<ForgetEngine> forget_engine_;
    std::unique_ptr<ConflictResolver> conflict_resolver_;
    std::unique_ptr<RemoveCoordinator> remove_coordinator_;
    std::unique_ptr<ConsolidationWorker> consolidation_worker_;
    std::unique_ptr<DerivedExtractor> derived_extractor_;
    std::unique_ptr<Reconciler> reconciler_;       // optional (only when v2_reconcile_enabled)
    std::unique_ptr<AggregateStalenessFilter> staleness_filter_;  // optional
    // stale_log_ removed in Phase 4 — RecallStale events live in events_log_.
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

    // ========== Agent Management (private) ==========
    std::unordered_map<std::string, std::unique_ptr<AgentStore>> agent_stores_;
    mutable std::mutex agent_stores_mutex_;
    std::string agents_meta_path_;  // path to agents.json

    Result<void, Error> initAgentStore(AgentStore& store);
    void loadAgentsMeta();
    void saveAgentsMeta();
};

}  // namespace amind
