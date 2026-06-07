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
  layer?: 'Raw' | 'Derived'
  metadata?: Record<string, string>
}

export interface MemoryListResponse {
  total: number; page: number; per_page: number; memories: Memory[]
}

export interface GraphEdge {
  from_id: string; to_id: string; type: string; weight: number; created_at?: number
  // Populated by /v1/graph/neighbors/{id} so the UI can render neighbor
  // phase + content preview without N+1 lookups.
  other_phase?: string
  other_confidence?: string
  other_preview?: string
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

// GateLogEntry / ForgetLogEntry / RecallStaleLogEntry / ReconcileLogEntry
// and their response types removed in Phase 4 — superseded by MemoryEventEntry
// + EventsQueryResponse / MemoryTraceResponse / TraceResponse below.

export const api = {
  health:    () => fetchApi<HealthResponse>('/v1/health'),
  metrics:   () => fetchApi<Metrics>('/v1/metrics'),
  coverage:  () => fetchApi<CoverageStats>('/v1/metacognition/coverage'),
  conflicts: () => fetchApi<ConflictInfo[]>('/v1/metacognition/conflicts'),

  listMemories: (page = 1, perPage = 50, owner = '', phase = '', q = '',
                  agent_id = '', userId = '', layer = '',
                  includeTombstone = false) => {
    const params = new URLSearchParams({
      page: String(page), per_page: String(perPage),
      owner, phase, q, agent_id, user_id: userId,
    })
    if (layer) params.set('layer', layer)
    if (includeTombstone) params.set('include_tombstone', '1')
    return fetchApi<MemoryListResponse>(`/v1/memories/list?${params.toString()}`)
  },

  getMemory:  (id: string) => fetchApi<Memory>(`/v1/memories/${id}`),
  getHistory: (id: string) => fetchApi<Memory[]>(`/v1/memories/${id}/history`),
  deleteMemory: (id: string) => fetchApi<{deleted: boolean}>(`/v1/memories/${id}`, { method: 'DELETE' }),
  archiveMemory: (id: string) => fetchApi<{archived: boolean}>(`/v1/memories/${id}/archive`, { method: 'POST' }),

  listEdges: (page = 1, perPage = 500) =>
    fetchApi<EdgeListResponse>(`/v1/graph/edges?page=${page}&per_page=${perPage}`),
  getNeighbors: (id: string) => fetchApi<GraphEdge[]>(`/v1/graph/neighbors/${id}`),
  getNeighborsIncoming: (id: string) =>
    fetchApi<GraphEdge[]>(`/v1/graph/neighbors/${id}?include_incoming=1`),

  listSessions: () => fetchApi<SessionSummary[]>('/v1/sessions/list'),
  sessionSummary: (id: string) => fetchApi<SessionSummary>(`/v1/sessions/${id}/summary`),

  storeMemory: (content: string, scope = 'private', agent_id = 'default') =>
    fetchApi<{ memory_ids: string[]; async_refinement_scheduled: boolean }>(
      '/v1/memories', {
        method: 'POST',
        body: JSON.stringify({ content, scope, agent_id }),
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

  // Pipeline/Gate/Forget/Stale legacy API removed in Phase 4.
  // All observability now goes through api.events() / api.memoryTrace() /
  // api.traceById() / api.eventsStats() / api.resurrectEvent() defined below.

  // --- Auth Key Management ---
  createApiKey: (label: string) =>
    fetchApi<ApiKeyCreateResponse>('/v1/auth/keys', {
      method: 'POST',
      body: JSON.stringify({ label }),
    }),
  listApiKeys: () => fetchApi<ApiKeyListResponse>('/v1/auth/keys'),
  revokeApiKey: (keyId: string) =>
    fetchApi<{ status: string }>(`/v1/auth/keys/${keyId}`, { method: 'DELETE' }),

  // --- Unified Memory Event Log (Phase 3) ---
  events: (params: {
    limit?: number
    kind?: EventKind | ''
    status?: EventStatus | ''
    memoryId?: string
    traceId?: string
    agent_id?: string
    sinceMs?: number
    untilMs?: number
  } = {}) => {
    const q = new URLSearchParams()
    q.set('limit', String(params.limit ?? 200))
    if (params.kind)      q.set('kind', params.kind)
    if (params.status)    q.set('status', params.status)
    if (params.memoryId)  q.set('memory_id', params.memoryId)
    if (params.traceId)   q.set('trace_id', params.traceId)
    if (params.agent_id)  q.set('agent_id', params.agent_id)
    if (params.sinceMs)   q.set('since_ms', String(params.sinceMs))
    if (params.untilMs)   q.set('until_ms', String(params.untilMs))
    return fetchApi<EventsQueryResponse>(`/v1/events?${q.toString()}`)
  },
  eventsStats: (params: {
    agent_id?: string
    memoryId?: string
    traceId?: string
    status?: EventStatus | ''
    sinceMs?: number
    untilMs?: number
  } = {}) => {
    // Intentionally excludes `kind` — if the page is filtered to a single kind,
    // we still want the other kind cards populated so they remain clickable as
    // "switch to this kind within the current scope".
    const q = new URLSearchParams()
    if (params.agent_id)  q.set('agent_id', params.agent_id)
    if (params.memoryId)  q.set('memory_id', params.memoryId)
    if (params.traceId)   q.set('trace_id', params.traceId)
    if (params.status)    q.set('status', params.status)
    if (params.sinceMs)   q.set('since_ms', String(params.sinceMs))
    if (params.untilMs)   q.set('until_ms', String(params.untilMs))
    return fetchApi<EventsStatsResponse>(`/v1/events/stats?${q.toString()}`)
  },
  memoryTrace: (id: string) =>
    fetchApi<MemoryTraceResponse>(`/v1/memories/${id}/trace`),
  traceById: (traceId: string) =>
    fetchApi<TraceResponse>(`/v1/traces/${traceId}`),
  resurrectEvent: (eventId: string,
                    body: { content?: string; strategy?: string } = {}) =>
    fetchApi<{ memory_id: string; source_event_id: string; strategy: string }>(
      `/v1/admin/resurrect/${eventId}`, {
        method: 'POST',
        body: JSON.stringify(body),
      }),
}

// ── Unified observability types ────────────────────────────────────────────

export type EventKind =
  | 'Store' | 'Embed' | 'Gate' | 'Derive' | 'Reconcile'
  | 'LineagePropagate' | 'GraphEdge' | 'VersionUpdate'
  | 'Recall' | 'RecallSemantic' | 'RecallFilter' | 'RecallStale'
  | 'GcDecay' | 'GcArchive' | 'GcTombstone' | 'GcVacuum'
  | 'Consolidate' | 'Resurrect'
  | 'ProviderCall' | 'Error'

export type EventStatus = 'Ok' | 'Rejected' | 'Deferred' | 'Failed' | 'NoOp'

export interface MemoryEventEntry {
  event_id: string
  parent_event_id: string
  trace_id: string
  memory_id: string
  timestamp_ms: number
  duration_ms: number
  agent_id: string
  kind: EventKind
  status: EventStatus
  summary: string
  attrs?: Record<string, string>
}

export interface EventsQueryResponse {
  entries: MemoryEventEntry[]
  memory_size: number
}

export interface EventsStatsResponse {
  by_kind: Record<string, number>
  by_status: Record<string, number>
  total: number
  memory_size: number
}

export interface TraceGroup {
  trace_id: string
  events: MemoryEventEntry[]
  event_count: number
  start_ms: number
  end_ms: number
}

export interface MemoryTraceResponse {
  memory_id: string
  total_events: number
  traces: TraceGroup[]
}

export interface TraceResponse {
  trace_id: string
  event_count: number
  start_ms: number
  end_ms: number
  events: MemoryEventEntry[]
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

// PipelineStats / ReconcileLogEntry / ReconcileLogResponse types removed in
// Phase 4. Reconciler decisions now flow as MemoryEvent{kind=Reconcile}
// queryable via /v1/events?kind=Reconcile.

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
