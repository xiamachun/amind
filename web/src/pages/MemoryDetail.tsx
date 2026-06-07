import { useEffect, useMemo, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { api, type Memory, type GraphEdge, type MemoryTraceResponse,
         type MemoryEventEntry } from '../api/client'

const KIND_COLORS: Record<string, string> = {
  Store: 'bg-indigo-900/40 text-indigo-300',
  Embed: 'bg-cyan-900/40 text-cyan-300',
  Gate: 'bg-emerald-900/40 text-emerald-300',
  Reconcile: 'bg-violet-900/40 text-violet-300',
  RecallStale: 'bg-rose-900/40 text-rose-300',
  GcDecay: 'bg-sky-900/40 text-sky-300',
  GcArchive: 'bg-amber-900/40 text-amber-300',
  GcTombstone: 'bg-rose-900/40 text-rose-300',
  Resurrect: 'bg-emerald-900/40 text-emerald-300',
  Recall: 'bg-teal-900/40 text-teal-300',
}

const fmtMs = (ms: number) => ms ? new Date(ms).toLocaleString() : '-'
const fmtDur = (ms: number) => !ms ? '0' : (ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`)

// ── Lifecycle inference ──────────────────────────────────────────────────────
// Picks ONE label that best describes the memory's current life stage, based
// on its current phase + recent events. Order matters: stronger signals first.
type LifecycleLabel = {
  text: string
  className: string
  hint: string
}

const RECALL_RECENT_MS = 7 * 24 * 60 * 60 * 1000  // 7 days

function deriveLifecycle(memory: Memory, events: MemoryEventEntry[]): LifecycleLabel {
  if (memory.phase === 'Tombstone') {
    return { text: 'Tombstoned',
             className: 'bg-rose-900/50 text-rose-200 border-rose-700',
             hint: 'Soft-deleted — awaiting physical vacuum' }
  }
  if (memory.phase === 'Archived') {
    return { text: 'Archived',
             className: 'bg-amber-900/40 text-amber-200 border-amber-700',
             hint: 'Moved to cold storage; recall priority lowered' }
  }
  if (memory.phase === 'Versioned') {
    return { text: 'Versioned',
             className: 'bg-blue-900/40 text-blue-200 border-blue-700',
             hint: 'Superseded by a newer version' }
  }
  const now = Date.now()
  const recentRecall = events.find(e =>
    (e.kind === 'Recall' || e.kind === 'RecallSemantic') &&
    e.timestamp_ms > now - RECALL_RECENT_MS
  )
  if (recentRecall) {
    return { text: 'Recalled',
             className: 'bg-teal-900/40 text-teal-200 border-teal-700',
             hint: `Recently surfaced via recall (last 7d)` }
  }
  const gcEvent = events.find(e =>
    e.kind === 'GcDecay' || e.kind === 'GcArchive')
  if (gcEvent) {
    return { text: 'Decaying',
             className: 'bg-sky-900/40 text-sky-200 border-sky-700',
             hint: 'GC has flagged this memory for forget-score accumulation' }
  }
  return { text: 'Cold',
           className: 'bg-gray-800 text-gray-300 border-gray-700',
           hint: 'No recall or GC activity yet' }
}

// ── Per-kind aggregate counters ──────────────────────────────────────────────
function tallyByKind(events: MemoryEventEntry[]) {
  const out: Record<string, number> = {}
  const outStatus: Record<string, number> = {}
  for (const e of events) {
    out[e.kind] = (out[e.kind] || 0) + 1
    if (e.kind === 'Gate') {
      const s = e.status || 'Ok'
      outStatus[`Gate ${s}`] = (outStatus[`Gate ${s}`] || 0) + 1
    }
  }
  return { byKind: out, gateBreakdown: outStatus }
}

// ── forget_score sparkline ───────────────────────────────────────────────────
// Pulls forget_score from each GC event (in chronological order) and draws a
// tiny inline SVG line chart. Returns null when there's nothing to plot.
function ForgetScoreSparkline({ events }: { events: MemoryEventEntry[] }) {
  const pts = useMemo(() => {
    return events
      .filter(e => e.kind === 'GcDecay' || e.kind === 'GcArchive' || e.kind === 'GcTombstone')
      .filter(e => e.attrs?.forget_score)
      .map(e => ({
        ts: e.timestamp_ms,
        score: parseFloat(e.attrs!.forget_score),
        kind: e.kind,
      }))
      .filter(p => Number.isFinite(p.score))
      .sort((a, b) => a.ts - b.ts)
  }, [events])

  if (pts.length < 1) return null

  const W = 240, H = 60, P = 6
  // Pin y-axis to [0, 1] so the threshold band is visually meaningful even
  // when all observed scores happen to cluster in a narrow range.
  const yMin = 0, yMax = 1
  const xStep = pts.length === 1 ? 0 : (W - 2 * P) / (pts.length - 1)
  const yFor = (s: number) => H - P - ((s - yMin) / (yMax - yMin)) * (H - 2 * P)
  const xFor = (i: number) => P + i * xStep
  const linePath = pts.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xFor(i).toFixed(1)} ${yFor(p.score).toFixed(1)}`
  ).join(' ')

  // Decay threshold reference line (configured at 0.30 in amind defaults).
  const decayY = yFor(0.30)
  const archiveY = yFor(0.60)
  const tombstoneY = yFor(0.85)

  const latest = pts[pts.length - 1]

  return (
    <div className="flex items-center gap-4">
      <svg width={W} height={H} className="bg-gray-950 rounded">
        {/* Threshold guide lines */}
        <line x1={P} x2={W - P} y1={decayY} y2={decayY}
              stroke="#0369a1" strokeDasharray="2 3" strokeWidth={0.5} opacity={0.5} />
        <line x1={P} x2={W - P} y1={archiveY} y2={archiveY}
              stroke="#b45309" strokeDasharray="2 3" strokeWidth={0.5} opacity={0.5} />
        <line x1={P} x2={W - P} y1={tombstoneY} y2={tombstoneY}
              stroke="#be123c" strokeDasharray="2 3" strokeWidth={0.5} opacity={0.5} />
        {/* Line */}
        <path d={linePath} stroke="#8b5cf6" strokeWidth={1.4} fill="none" />
        {/* Dots */}
        {pts.map((p, i) => (
          <circle key={i} cx={xFor(i)} cy={yFor(p.score)} r={2.2}
                  fill={p.kind === 'GcTombstone' ? '#f43f5e'
                       : p.kind === 'GcArchive' ? '#f59e0b'
                       : '#a78bfa'}>
            <title>{`${new Date(p.ts).toLocaleString()} · ${p.kind} · score=${p.score.toFixed(3)}`}</title>
          </circle>
        ))}
        {/* Y-axis labels */}
        <text x={W - 2} y={decayY - 1}    textAnchor="end" fontSize="7" fill="#475569">0.30 decay</text>
        <text x={W - 2} y={archiveY - 1}  textAnchor="end" fontSize="7" fill="#475569">0.60 archive</text>
        <text x={W - 2} y={tombstoneY - 1} textAnchor="end" fontSize="7" fill="#475569">0.85 tombstone</text>
      </svg>
      <div className="text-xs space-y-0.5">
        <div className="text-gray-400">latest score</div>
        <div className="text-violet-300 font-mono text-base">{latest.score.toFixed(3)}</div>
        <div className="text-gray-500">{pts.length} GC sample{pts.length !== 1 ? 's' : ''}</div>
      </div>
    </div>
  )
}

export default function MemoryDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [memory, setMemory] = useState<Memory | null>(null)
  const [history, setHistory] = useState<Memory[]>([])
  const [neighbors, setNeighbors] = useState<GraphEdge[]>([])
  const [trace, setTrace] = useState<MemoryTraceResponse | null>(null)
  const [error, setError] = useState('')

  // Derived facts produced from this raw memory (only meaningful when this
  // memory is itself a Raw layer record; for a Derived record this stays
  // empty). We fetch the incoming-edge view of neighbors and pick out the
  // DerivedFrom edges, then load each derived fact's record for display.
  const [derivedFacts, setDerivedFacts] = useState<Memory[]>([])

  useEffect(() => {
    if (!id) return
    api.getMemory(id).then(setMemory).catch(e => setError(e.message))
    api.getHistory(id).then(setHistory).catch(() => {})
    api.getNeighbors(id).then(setNeighbors).catch(() => {})
    api.memoryTrace(id).then(setTrace).catch(() => {})
    // Walk incoming DerivedFrom edges → fetch each derived record. These edges
    // are written from derived → raw, so include_incoming=1 surfaces them.
    api.getNeighborsIncoming(id).then(async edges => {
      const derived_ids = edges
        .filter(e => e.type === 'DerivedFrom' && String(e.to_id) === String(id))
        .map(e => String(e.from_id))
      if (derived_ids.length === 0) { setDerivedFacts([]); return }
      const records = await Promise.all(
        derived_ids.map(did => api.getMemory(did).catch(() => null))
      )
      setDerivedFacts(records.filter((r): r is Memory => r !== null))
    }).catch(() => {})
  }, [id])

  const fmtTime = (ts: number) => ts ? new Date(ts * 1000).toLocaleString() : '-'

  const handleDelete = async () => {
    if (!id || !confirm('Delete this memory?')) return
    await api.deleteMemory(id)
    navigate('/memories')
  }

  if (error) return <div className="text-red-400 p-8">Error: {error}</div>
  if (!memory) return <div className="text-gray-500 p-8">Loading...</div>

  // Flatten all events from the trace response for analytics — lifecycle,
  // aggregate counters, and the forget-score sparkline all consume the same
  // chronological list and avoid re-fetching.
  const allEvents: MemoryEventEntry[] = trace
    ? trace.traces.flatMap(t => t.events)
    : []
  const lifecycle = deriveLifecycle(memory, allEvents)
  const { byKind, gateBreakdown } = tallyByKind(allEvents)
  const totalEvents = allEvents.length

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/memories')}
          className="text-gray-500 hover:text-gray-300 text-sm">&larr; Back</button>
        <h2 className="text-lg font-semibold text-gray-200 flex-1">Memory Detail</h2>
        <button onClick={handleDelete}
          className="px-3 py-1.5 rounded-lg text-sm bg-red-900/30 text-red-400 hover:bg-red-900/50">
          Delete
        </button>
      </div>

      {/* Content Card */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <div className="flex items-start gap-3">
          <span className={`shrink-0 inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-md border ${lifecycle.className}`}
                title={lifecycle.hint}>
            {lifecycle.text}
          </span>
          <span className="text-xs text-gray-500 self-center">{lifecycle.hint}</span>
        </div>
        <div>
          <p className="text-xs text-gray-500 uppercase mb-1">Content</p>
          <p className="text-gray-200 whitespace-pre-wrap leading-relaxed">{memory.content}</p>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 pt-3 border-t border-gray-800">
          <div>
            <p className="text-xs text-gray-500">ID</p>
            <p className="text-xs text-gray-400 font-mono break-all">{memory.memory_id}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Agent / User</p>
            <p className="text-sm text-indigo-300">{memory.agent_id}{memory.user_id ? ` / ${memory.user_id}` : ''}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Phase</p>
            <p className="text-sm text-gray-300">{memory.phase}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Confidence</p>
            <p className="text-sm text-gray-300">{memory.confidence}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Importance</p>
            <p className="text-sm text-gray-300">{memory.importance.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Version</p>
            <p className="text-sm text-gray-300">{memory.version}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Access Count</p>
            <p className="text-sm text-gray-300">{memory.access_count}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Embedding</p>
            <p className="text-sm text-gray-300">{memory.has_embedding ? 'Yes' : 'No'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Created</p>
            <p className="text-xs text-gray-400">{fmtTime(memory.created_at)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Last Accessed</p>
            <p className="text-xs text-gray-400">{fmtTime(memory.last_accessed)}</p>
          </div>
        </div>
      </div>

      {/* Activity summary — per-kind counters give an at-a-glance health view
          before drilling into individual events below. */}
      {totalEvents > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-400">Activity Summary</h3>
            <span className="text-xs text-gray-500">{totalEvents} event{totalEvents !== 1 ? 's' : ''} touched this memory</span>
          </div>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {['Store', 'Embed', 'Gate', 'Reconcile', 'Recall',
              'GcDecay', 'GcArchive', 'GcTombstone', 'Resurrect'].map(k => {
              const n = byKind[k] || 0
              if (n === 0) return null
              return (
                <Link key={k}
                      to={`/events?memory_id=${id}&kind=${k}`}
                      className={`block px-3 py-2 rounded-lg border border-gray-800 hover:border-gray-600 ${KIND_COLORS[k] || 'bg-gray-800/40 text-gray-300'}`}>
                  <div className="text-[10px] uppercase tracking-wider opacity-70">{k}</div>
                  <div className="text-lg font-semibold mt-0.5">{n}</div>
                </Link>
              )
            })}
          </div>
          {Object.keys(gateBreakdown).length > 0 && (
            <div className="text-xs text-gray-500 flex gap-4 pt-1 border-t border-gray-800/50">
              <span className="text-gray-600">Gate breakdown:</span>
              {Object.entries(gateBreakdown).map(([k, n]) => (
                <span key={k}><span className="text-gray-400">{k}</span> <span className="text-gray-200">{n}</span></span>
              ))}
            </div>
          )}
          {/* Sparkline only renders when we have GC samples to plot. */}
          <ForgetScoreSparkline events={allEvents} />
        </div>
      )}

      {/* Derived facts produced from this memory — what Stage 2 LLM extraction
          gave back. Each is the actual structured fact that recall surfaces;
          the raw above is just the evidence. */}
      {derivedFacts.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">
              Derived Facts ({derivedFacts.length})
            </h3>
            <span className="text-[11px] text-gray-500">
              Extracted by Stage 2 — these are what recall actually returns
            </span>
          </div>
          <div className="space-y-2">
            {derivedFacts.map(d => {
              const phaseColor =
                d.phase === 'Tombstone' ? 'bg-rose-900/40 text-rose-300' :
                d.phase === 'Archived'  ? 'bg-amber-900/40 text-amber-300' :
                d.phase === 'Versioned' ? 'bg-blue-900/40 text-blue-300' :
                                           'bg-emerald-900/40 text-emerald-300'
              return (
                <div key={d.memory_id}
                     onClick={() => navigate(`/memories/${d.memory_id}`)}
                     className="grid grid-cols-[120px_1fr_60px_60px] items-center gap-3 px-3 py-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 cursor-pointer">
                  <span className="text-xs font-mono text-blue-300 truncate" title={d.memory_id}>
                    #{d.memory_id.slice(-12)}
                  </span>
                  <span className="text-xs text-gray-200 truncate" title={d.content}>
                    {d.content}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${phaseColor} text-center`}>
                    {d.phase}
                  </span>
                  <span className="text-xs text-gray-500 text-right" title="importance">
                    {d.importance.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Graph Neighbors */}
      {neighbors.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Graph Neighbors ({neighbors.length})</h3>
            <span className="text-[11px] text-gray-500" title="Edge weight = cosine similarity of embeddings (HNSW). Type is threshold-based: ≥0.85 ConflictsWith, 0.5–0.85 Related.">
              w = embedding cosine similarity ⓘ
            </span>
          </div>
          {/* When a ConflictsWith edge is present, surface a one-line caveat
              so users don't read it as "amind detected a real contradiction".
              The current implementation labels any pair with cosine ≥ 0.85
              as ConflictsWith, which captures both real conflicts and
              near-duplicates. Real fact-level conflict resolution lives on
              the derived layer via Supersedes edges. */}
          {neighbors.some(e => e.type === 'ConflictsWith') && (
            <div className="text-[11px] text-amber-300/90 bg-amber-900/20 border border-amber-900/40 rounded px-3 py-1.5 mb-3 leading-relaxed">
              <span className="font-medium">Note:</span> ConflictsWith edges are similarity-based
              (cosine ≥ 0.85). They flag near-duplicates as well as true contradictions —
              for fact-level conflict resolution, look at <span className="font-mono text-violet-300">Supersedes</span> edges
              on the derived facts produced by this memory.
            </div>
          )}
          <div className="space-y-2">
            {neighbors.map((e, i) => {
              const other = e.from_id === id ? e.to_id : e.from_id
              const phase = e.other_phase || ''
              const phaseColor =
                phase === 'Tombstone' ? 'bg-rose-900/40 text-rose-300' :
                phase === 'Archived'  ? 'bg-amber-900/40 text-amber-300' :
                phase === 'Versioned' ? 'bg-blue-900/40 text-blue-300' :
                phase === 'Active'    ? 'bg-emerald-900/40 text-emerald-300' :
                phase === 'Missing'   ? 'bg-gray-800 text-gray-500' :
                                        'bg-gray-800 text-gray-400'
              const typeColor =
                e.type === 'ConflictsWith' ? 'bg-rose-900/40 text-rose-300' :
                e.type === 'Supersedes'    ? 'bg-violet-900/40 text-violet-300' :
                e.type === 'Derives'       ? 'bg-purple-900/40 text-purple-300' :
                                              'bg-indigo-900/40 text-indigo-300'
              return (
                <div key={i} onClick={() => navigate(`/memories/${other}`)}
                  className="grid grid-cols-[110px_140px_90px_1fr_60px] items-center gap-3 px-3 py-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 cursor-pointer">
                  <span className={`text-xs px-2 py-0.5 rounded ${typeColor} text-center`}>{e.type}</span>
                  <span className="text-xs text-gray-400 font-mono truncate" title={other}>#{other.slice(-12)}</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${phaseColor} text-center`}>{phase || '—'}</span>
                  <span className="text-xs text-gray-300 truncate" title={e.other_preview}>
                    {e.other_preview || <span className="text-gray-600">—</span>}
                  </span>
                  <span className="text-xs text-gray-500 text-right">w={e.weight.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Version History */}
      {history.length > 1 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Version History ({history.length})</h3>
          <div className="space-y-2">
            {history.map((h, i) => (
              <div key={i} className="px-3 py-2 rounded-lg bg-gray-800/50 text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-indigo-400">v{h.version}</span>
                  <span className="text-xs text-gray-500">{fmtTime(h.created_at)}</span>
                </div>
                <p className="text-gray-400 text-xs mt-1 truncate">{h.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trace — unified observability events grouped by trace */}
      {trace && trace.total_events > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Trace ({trace.total_events} events across {trace.traces.length} trace{trace.traces.length !== 1 ? 's' : ''})
          </h3>
          <div className="space-y-4">
            {trace.traces.map(tg => (
              <div key={tg.trace_id} className="border border-gray-800 rounded-lg overflow-hidden">
                <div className="px-3 py-2 bg-gray-800/50 flex items-center gap-3 text-xs">
                  <Link to={`/events?trace_id=${tg.trace_id}`}
                        className="font-mono text-violet-300 hover:underline"
                        title="Drill into full trace (including sibling memories) on the Events page">
                    trace #{tg.trace_id.slice(-12)}
                  </Link>
                  <span className="text-gray-500">{tg.event_count} event{tg.event_count !== 1 ? 's' : ''} for this memory</span>
                  <span className="text-gray-500">total {fmtDur(tg.end_ms - tg.start_ms)}</span>
                  <span className="ml-auto text-gray-500">{fmtMs(tg.start_ms)}</span>
                </div>
                <div className="divide-y divide-gray-800/50">
                  {tg.events.map(e => (
                    <div key={e.event_id}
                         className={`px-3 py-2 flex items-start gap-3 text-xs ${
                           e.memory_id === id ? '' : 'opacity-60'
                         }`}>
                      <span className={`shrink-0 px-2 py-0.5 rounded ${KIND_COLORS[e.kind] || 'bg-gray-800 text-gray-300'}`}>
                        {e.kind}
                      </span>
                      <span className="text-gray-500 shrink-0 w-12 text-right">{fmtDur(e.duration_ms)}</span>
                      <span className={`shrink-0 ${
                        e.status === 'Ok' ? 'text-emerald-400' :
                        e.status === 'Failed' ? 'text-red-400' :
                        e.status === 'Rejected' ? 'text-rose-400' :
                        e.status === 'Deferred' ? 'text-amber-400' : 'text-gray-500'
                      }`}>{e.status}</span>
                      {e.memory_id !== id && e.memory_id !== '0' && (
                        <Link to={`/memories/${e.memory_id}`}
                              className="text-blue-300 hover:underline font-mono shrink-0">
                          #{e.memory_id.slice(-8)}
                        </Link>
                      )}
                      <span className="text-gray-300 flex-1 truncate" title={e.summary}>
                        {e.summary || '—'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
