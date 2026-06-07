import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  api,
  type EventKind,
  type EventStatus,
  type MemoryEventEntry,
  type EventsStatsResponse,
} from '../api/client'

// Color per event kind (group by lifecycle stage)
const KIND_COLORS: Record<string, string> = {
  Store:            'bg-indigo-900/40 text-indigo-300',
  Embed:            'bg-cyan-900/40 text-cyan-300',
  Gate:             'bg-emerald-900/40 text-emerald-300',
  Derive:           'bg-purple-900/40 text-purple-300',
  Reconcile:        'bg-violet-900/40 text-violet-300',
  LineagePropagate: 'bg-orange-900/40 text-orange-300',
  GraphEdge:        'bg-gray-700/40 text-gray-300',
  VersionUpdate:    'bg-blue-900/40 text-blue-300',
  Recall:           'bg-teal-900/40 text-teal-300',
  RecallSemantic:   'bg-teal-900/40 text-teal-300',
  RecallFilter:     'bg-amber-900/40 text-amber-300',
  RecallStale:      'bg-rose-900/40 text-rose-300',
  GcDecay:          'bg-sky-900/40 text-sky-300',
  GcArchive:        'bg-amber-900/40 text-amber-300',
  GcTombstone:      'bg-rose-900/40 text-rose-300',
  GcVacuum:         'bg-fuchsia-900/40 text-fuchsia-300',
  Consolidate:      'bg-pink-900/40 text-pink-300',
  Resurrect:        'bg-emerald-900/40 text-emerald-300',
  ProviderCall:     'bg-gray-800/40 text-gray-400',
  Error:            'bg-red-900/40 text-red-300',
}

const STATUS_COLORS: Record<string, string> = {
  Ok:       'text-emerald-400',
  Rejected: 'text-rose-400',
  Deferred: 'text-amber-400',
  Failed:   'text-red-400',
  NoOp:     'text-gray-500',
}

const ALL_KINDS: EventKind[] = [
  'Store', 'Embed', 'Gate', 'Derive', 'Reconcile',
  'LineagePropagate', 'GraphEdge', 'VersionUpdate',
  'Recall', 'RecallSemantic', 'RecallFilter', 'RecallStale',
  'GcDecay', 'GcArchive', 'GcTombstone', 'GcVacuum',
  'Consolidate', 'Resurrect', 'ProviderCall', 'Error',
]

const ALL_STATUSES: EventStatus[] = ['Ok', 'Rejected', 'Deferred', 'Failed', 'NoOp']

const fmtTime = (ms: number) => ms ? new Date(ms).toLocaleString() : '-'
const fmtDur = (ms: number) => {
  if (!ms) return '0'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export default function Events() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [entries, setEntries] = useState<MemoryEventEntry[]>([])
  const [stats, setStats] = useState<EventsStatsResponse | null>(null)
  const [ringSize, setRingSize] = useState(0)
  const [loading, setLoading] = useState(false)
  const [resurrectMsg, setResurrectMsg] = useState('')

  // URL-driven filters; preset by Layout quick-filter links.
  const kindFilter = (params.get('kind') as EventKind) || ''
  const statusFilter = (params.get('status') as EventStatus) || ''
  const agentIdFilter = params.get('agent_id') || ''
  const memFilter = params.get('memory_id') || ''
  const traceFilter = params.get('trace_id') || ''

  const updateParam = (key: string, val: string) => {
    const p = new URLSearchParams(params)
    if (val) p.set(key, val)
    else p.delete(key)
    setParams(p)
  }

  const load = useCallback(() => {
    setLoading(true)
    Promise.all([
      api.events({
        limit: 200,
        kind: kindFilter || undefined,
        status: statusFilter || undefined,
        memoryId: memFilter || undefined,
        traceId: traceFilter || undefined,
        agent_id: agentIdFilter || undefined,
      }),
      api.eventsStats({
        agent_id: agentIdFilter || undefined,
        memoryId:  memFilter || undefined,
        traceId:   traceFilter || undefined,
        status:    statusFilter || undefined,
      }),
    ])
      .then(([rows, st]) => {
        setEntries(rows.entries)
        setRingSize(rows.memory_size)
        setStats(st)
      })
      .finally(() => setLoading(false))
  }, [kindFilter, statusFilter, memFilter, traceFilter, agentIdFilter])

  useEffect(() => { load() }, [load])

  // Page-level title and workflow buttons specialize when a kind filter is set.
  const pageTitle = kindFilter
    ? `Events · ${kindFilter}`
    : 'Events (unified)'

  const handleResurrect = async (e: MemoryEventEntry) => {
    if (!confirm(`Resurrect event #${e.event_id.slice(-10)}?\nContent: ${e.summary}`)) return
    setResurrectMsg('')
    try {
      const r = await api.resurrectEvent(e.event_id, { strategy: 'coexist' })
      setResurrectMsg(`✅ Resurrected → memory #${r.memory_id.slice(-10)}`)
      setTimeout(() => { setResurrectMsg(''); load() }, 1800)
    } catch (err: any) {
      setResurrectMsg(`❌ ${err?.message || 'resurrect failed'}`)
    }
  }

  // Useful per-filter quick-stat highlights.
  const headlineCount = useMemo(() => {
    if (!stats) return 0
    if (kindFilter) return stats.by_kind[kindFilter] || 0
    return stats.total
  }, [stats, kindFilter])

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">{pageTitle}</h2>
        <span className="text-xs text-gray-500">
          {headlineCount.toLocaleString()} events · ring={ringSize.toLocaleString()}
        </span>
        <button onClick={load}
          className="ml-auto px-3 py-1.5 text-sm rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700">
          Refresh
        </button>
      </div>

      {/* Stats strip */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {Object.entries(stats.by_kind).slice(0, 10).map(([k, n]) => (
            <button key={k} onClick={() => updateParam('kind', kindFilter === k ? '' : k)}
              className={`text-left bg-gray-900 border rounded-lg px-3 py-2 ${
                kindFilter === k ? 'border-indigo-500/60' : 'border-gray-800 hover:border-gray-700'
              }`}>
              <div className="text-[10px] text-gray-500 uppercase tracking-wider">{k}</div>
              <div className="text-lg font-semibold text-gray-200 mt-0.5">{n.toLocaleString()}</div>
            </button>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap bg-gray-900 border border-gray-800 rounded-xl px-3 py-2">
        <select value={kindFilter} onChange={e => updateParam('kind', e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300">
          <option value="">All kinds</option>
          {ALL_KINDS.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <select value={statusFilter} onChange={e => updateParam('status', e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300">
          <option value="">All statuses</option>
          {ALL_STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <input value={agentIdFilter} onChange={e => updateParam('agent_id', e.target.value)}
          placeholder="agent_id..."
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 w-48" />
        <input value={memFilter} onChange={e => updateParam('memory_id', e.target.value)}
          placeholder="memory_id..."
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 w-44 font-mono text-xs" />
        <input value={traceFilter} onChange={e => updateParam('trace_id', e.target.value)}
          placeholder="trace_id..."
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-300 w-44 font-mono text-xs" />
        {(kindFilter || statusFilter || agentIdFilter || memFilter || traceFilter) && (
          <button onClick={() => setParams(new URLSearchParams())}
            className="ml-auto text-xs px-2 py-1 rounded bg-gray-800 text-gray-400 hover:bg-gray-700">
            Clear filters
          </button>
        )}
      </div>

      {resurrectMsg && (
        <div className={`text-sm rounded p-2 ${
          resurrectMsg.startsWith('✅') ? 'bg-emerald-900/40 text-emerald-300'
                                        : 'bg-rose-900/40 text-rose-300'}`}>
          {resurrectMsg}
        </div>
      )}

      {/* Entries */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="text-left px-3 py-3 w-32">Kind</th>
              <th className="text-left px-3 py-3 w-20">Status</th>
              <th className="text-left px-3 py-3 w-28">NS</th>
              <th className="text-left px-3 py-3 w-32">Memory</th>
              <th className="text-left px-3 py-3 w-32">Trace</th>
              <th className="text-left px-3 py-3 w-20">Dur</th>
              <th className="text-left px-3 py-3">Summary</th>
              <th className="text-left px-3 py-3 w-44">When</th>
              <th className="text-left px-3 py-3 w-24">Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={9} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : entries.length === 0 ? (
              <tr><td colSpan={9} className="px-4 py-8 text-center text-gray-500">No events match the filter.</td></tr>
            ) : entries.map(e => (
              <tr key={e.event_id} className="border-b border-gray-800/40 hover:bg-gray-800/30">
                <td className="px-3 py-2">
                  <span className={`text-xs px-2 py-0.5 rounded ${KIND_COLORS[e.kind] || 'bg-gray-800 text-gray-300'}`}>
                    {e.kind}
                  </span>
                </td>
                <td className={`px-3 py-2 text-xs ${STATUS_COLORS[e.status] || ''}`}>{e.status}</td>
                <td className="px-3 py-2 text-xs font-mono text-blue-300 truncate max-w-[100px]"
                    title={e.agent_id}>{e.agent_id || <span className="text-gray-600">—</span>}</td>
                <td className="px-3 py-2 text-xs">
                  {e.memory_id !== '0' ? (
                    <button onClick={() => navigate(`/memories/${e.memory_id}`)}
                            className="font-mono text-blue-300 hover:underline">
                      #{e.memory_id.slice(-10)}
                    </button>
                  ) : <span className="text-gray-600">—</span>}
                </td>
                <td className="px-3 py-2 text-xs">
                  {e.trace_id !== '0' ? (
                    <button onClick={() => updateParam('trace_id', e.trace_id)}
                            className="font-mono text-violet-300 hover:underline"
                            title="Pin filter to this trace">
                      #{e.trace_id.slice(-10)}
                    </button>
                  ) : <span className="text-gray-600">—</span>}
                </td>
                <td className="px-3 py-2 text-xs text-gray-400">{fmtDur(e.duration_ms)}</td>
                <td className="px-3 py-2 text-gray-300">
                  <div className="text-xs truncate max-w-md" title={e.summary}>
                    {e.summary || <span className="text-gray-600">—</span>}
                  </div>
                  {e.attrs && Object.keys(e.attrs).length > 0 && (
                    <div className="text-[10px] text-gray-500 mt-0.5 truncate max-w-md"
                         title={JSON.stringify(e.attrs)}>
                      {Object.entries(e.attrs).slice(0, 3).map(([k, v]) =>
                        <span key={k} className="mr-2"><span className="text-gray-600">{k}=</span>{v}</span>
                      )}
                    </div>
                  )}
                </td>
                <td className="px-3 py-2 text-xs text-gray-500">{fmtTime(e.timestamp_ms)}</td>
                <td className="px-3 py-2">
                  {e.kind === 'Gate' && (e.status === 'Rejected' || e.status === 'Deferred') ? (
                    <button onClick={() => handleResurrect(e)}
                            className="text-xs px-2 py-1 rounded bg-indigo-700 hover:bg-indigo-600 text-white">
                      Resurrect ⤴
                    </button>
                  ) : <span className="text-xs text-gray-600">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
