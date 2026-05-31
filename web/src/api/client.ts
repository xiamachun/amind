const API_BASE = '/api'

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!res.ok) throw new Error(`API Error: ${res.status}`)
  return res.json()
}

export interface Metrics {
  total_memories: number
  graph_edges: number
  pool_connections: number
  pool_total_acquired: number
  pool_total_reused: number
  pool_circuit_open: boolean
}

export interface CoverageStats {
  total: number; active: number; stale: number; conflicted: number
  owner_distribution: Record<string, number>
  phase_distribution: Record<string, number>
  confidence_distribution: Record<string, number>
}

export interface Memory {
  memory_id: string; content: string; owner: string; phase: string
  confidence: string; importance: number; created_at: number
  last_accessed: number; access_count: number; version: number
  has_embedding: boolean
  metadata?: Record<string, string>
}

export interface MemoryListResponse {
  total: number; page: number; per_page: number; memories: Memory[]
}

export interface GraphEdge {
  from_id: string; to_id: string; type: string; weight: number; created_at: number
}

export interface EdgeListResponse {
  total: number; page: number; per_page: number; edges: GraphEdge[]
}

export interface SessionSummary {
  session_id: string; agent_id: string; turn_count: number
  current_intent: string; memory_count: number; fact_count: number
  started_at: number; last_turn_at: number; active: boolean
}

export interface ConflictInfo {
  memory_a: string; memory_b: string; conflict_type: string; explanation: string
}

export interface HealthResponse { status: string; version: string }

export interface GateLogEntry {
  entry_id: string
  timestamp_ms: number
  namespace: string
  content: string
  decision: 'Accepted' | 'Rejected' | 'Deferred'
  reason: string
  marginal_value: number
  conflict_with_id: string
  owner: string
  layer: 'Raw' | 'Derived'
  user_metadata?: Record<string, string>
  memory_id?: string
  resurrected_to: string
  resurrected_at_ms: number
  resurrect_strategy?: string
}

export interface GateLogStats {
  accepted: number; rejected: number; deferred: number
  resurrected: number; total: number
}

export interface GateLogResponse {
  entries: GateLogEntry[]
  stats: GateLogStats
  memory_size: number
}

export interface ResurrectResponse {
  memory_id: string
  strategy: string
  replaced_id?: string
  note?: string
}

export interface ForgetLogEntry {
  timestamp_ms: number
  memory_id: string
  decision: 'Decay' | 'Archive' | 'Tombstone' | 'Vacuum'
            | 'DropFromHNSW' | 'ResolveConflict' | 'LineageInvalidate'
            | 'GateReject' | 'GateDefer'
  reason: string
  before_state: string
  after_state: string
  lineage_affected?: string[]
  gc_worker_id?: string
  namespace?: string
  content?: string
}

export interface ForgetLogStats {
  decay: number; archive: number; tombstone: number; vacuum: number
  other: number; total: number
}

export interface ForgetLogConfig {
  enabled: boolean
  shadow_mode: boolean
  decay_threshold: number
  archive_threshold: number
  tombstone_threshold: number
  gc_interval_seconds: number
  sample_ratio: number
}

export interface ForgetLogResponse {
  entries: ForgetLogEntry[]
  stats: ForgetLogStats
  config: ForgetLogConfig
  memory_size: number
}

export interface RecallStaleLogEntry {
  timestamp_ms: number
  query: string
  namespace: string
  aggregate_id: string
  aggregate_preview: string
  aggregate_created_at: number
  witness_in_aggregate: string[]
  witness_in_newer: string[]
  newer_fact_ids: string[]
  action: 'Filter' | 'Downweight'
  pre_score: number
  post_score: number
}

export interface RecallStaleLogResponse {
  enabled: boolean
  entries: RecallStaleLogEntry[]
  memory_size: number
  stats: { total: number }
}

export const api = {
  health:    () => fetchApi<HealthResponse>('/v1/health'),
  metrics:   () => fetchApi<Metrics>('/v1/metrics'),
  coverage:  () => fetchApi<CoverageStats>('/v1/metacognition/coverage'),
  conflicts: () => fetchApi<ConflictInfo[]>('/v1/metacognition/conflicts'),

  listMemories: (page = 1, perPage = 50, owner = '', phase = '', q = '', namespace = '', userId = '') =>
    fetchApi<MemoryListResponse>(
      `/v1/memories/list?page=${page}&per_page=${perPage}&owner=${owner}&phase=${phase}&q=${encodeURIComponent(q)}&namespace=${encodeURIComponent(namespace)}&user_id=${encodeURIComponent(userId)}`),

  getMemory:  (id: string) => fetchApi<Memory>(`/v1/memories/${id}`),
  getHistory: (id: string) => fetchApi<Memory[]>(`/v1/memories/${id}/history`),
  deleteMemory: (id: string) => fetchApi<{deleted: boolean}>(`/v1/memories/${id}`, { method: 'DELETE' }),
  archiveMemory: (id: string) => fetchApi<{archived: boolean}>(`/v1/memories/${id}/archive`, { method: 'POST' }),

  listEdges: (page = 1, perPage = 500) =>
    fetchApi<EdgeListResponse>(`/v1/graph/edges?page=${page}&per_page=${perPage}`),
  getNeighbors: (id: string) => fetchApi<GraphEdge[]>(`/v1/graph/neighbors/${id}`),

  listSessions: () => fetchApi<SessionSummary[]>('/v1/sessions/list'),
  sessionSummary: (id: string) => fetchApi<SessionSummary>(`/v1/sessions/${id}/summary`),

  storeMemory: (content: string, owner = 'user', agent_id = 'default') =>
    fetchApi<{ memory_ids: string[]; async_refinement_scheduled: boolean }>(
      '/v1/memories', {
        method: 'POST',
        body: JSON.stringify({ content, owner, agent_id }),
      }),

  backupExport: (type = 'memories') => fetchApi<string>(`/v1/backup/export?type=${type}`),

  // --- Variables (SHOW / SET / RELOAD) ---
  listVariables: (like = '%') =>
    fetchApi<{ variables: Variable[] }>(`/v1/variables?like=${encodeURIComponent(like)}`),

  setVariable: (name: string, value: string) =>
    fetchApi<{ ok: boolean; name: string; old_value: string; new_value: string }>(
      `/v1/variables/${encodeURIComponent(name)}`, {
        method: 'PUT',
        body: JSON.stringify({ value }),
      }),

  reloadConfig: () =>
    fetchApi<{ ok: boolean; changed: number }>('/v1/config/reload', { method: 'POST' }),

  // --- Pipeline observability ---
  pipelineStats: () => fetchApi<PipelineStats>('/v1/pipeline/stats'),
  reconcileLog: (limit = 100) =>
    fetchApi<ReconcileLogResponse>(`/v1/pipeline/reconcile-log?limit=${limit}`),

  // --- Gate Log (audit + resurrect) ---
  gateLog: (params: {
    limit?: number; decision?: string; namespace?: string;
    sinceMs?: number; onlyUnresurrected?: boolean
  } = {}) => {
    const q = new URLSearchParams()
    q.set('limit', String(params.limit ?? 200))
    if (params.decision) q.set('decision', params.decision)
    if (params.namespace) q.set('namespace', params.namespace)
    if (params.sinceMs) q.set('since_ms', String(params.sinceMs))
    if (params.onlyUnresurrected) q.set('only_unresurrected', '1')
    return fetchApi<GateLogResponse>(`/v1/gate/log?${q.toString()}`)
  },

  gateResurrect: (entryId: string,
                   strategy: 'coexist' | 'replace_conflict' | 'update_existing' = 'coexist') =>
    fetchApi<ResurrectResponse>(`/v1/gate/log/${entryId}/resurrect`, {
      method: 'POST',
      body: JSON.stringify({ strategy }),
    }),

  // --- Forget Log (audit) ---
  forgetLog: (params: { limit?: number; decision?: string } = {}) => {
    const q = new URLSearchParams()
    q.set('limit', String(params.limit ?? 200))
    if (params.decision) q.set('decision', params.decision)
    return fetchApi<ForgetLogResponse>(`/v1/forget/log?${q.toString()}`)
  },

  // --- Recall stale-log (aggregate staleness filter audit) ---
  recallStaleLog: (params: { limit?: number; namespace?: string } = {}) => {
    const q = new URLSearchParams()
    q.set('limit', String(params.limit ?? 200))
    if (params.namespace) q.set('namespace', params.namespace)
    return fetchApi<RecallStaleLogResponse>(`/v1/recall/stale-log?${q.toString()}`)
  },

  // --- Auth Key Management ---
  createApiKey: (label: string) =>
    fetchApi<ApiKeyCreateResponse>('/v1/auth/keys', {
      method: 'POST',
      body: JSON.stringify({ label }),
    }),
  listApiKeys: () => fetchApi<ApiKeyListResponse>('/v1/auth/keys'),
  revokeApiKey: (keyId: string) =>
    fetchApi<{ status: string }>(`/v1/auth/keys/${keyId}`, { method: 'DELETE' }),
}

export interface Variable {
  name: string
  value: string
  default_value: string
  type: string        // STRING | INT | FLOAT | BOOL
  mode: string        // DYNAMIC | READONLY
  category: string
  description: string
}

export interface PipelineStats {
  queue_depth: number
  queue_capacity: number
  executor_threads: number
  tasks_completed: number
  tasks_failed: number
  reconciler: {
    total_calls: number
    llm_invocations: number
    llm_failures: number
    op_add: number
    op_replace: number
    op_retract: number
    op_reinforce: number
    op_noop: number
  }
}

export interface ReconcileLogEntry {
  timestamp_ms: number
  candidate: string
  op: string
  target_id: string
  latency_ms: number
  from_fallback: boolean
}

export interface ReconcileLogResponse {
  entries: ReconcileLogEntry[]
}

export interface ApiKeyInfo {
  id: string
  label: string
  key_prefix: string
  created_at: number
  last_used_at: number
}

export interface ApiKeyListResponse {
  keys: ApiKeyInfo[]
}

export interface ApiKeyCreateResponse {
  key: string
  message: string
}
