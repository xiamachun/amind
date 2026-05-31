import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type RecallStaleLogEntry } from '../api/client'

const fmtTime = (ms: number) => ms ? new Date(ms).toLocaleString() : '-'
const fmtCreated = (sec: number) => sec ? new Date(sec * 1000).toLocaleString() : '-'

export default function RecallStaleLog() {
  const navigate = useNavigate()
  const [entries, setEntries] = useState<RecallStaleLogEntry[]>([])
  const [enabled, setEnabled] = useState(true)
  const [total, setTotal] = useState(0)
  const [memorySize, setMemorySize] = useState(0)
  const [loading, setLoading] = useState(false)
  const [nsFilter, setNsFilter] = useState('')

  const load = useCallback(() => {
    setLoading(true)
    api.recallStaleLog({ limit: 200, namespace: nsFilter || undefined })
      .then(r => {
        setEntries(r.entries)
        setEnabled(r.enabled)
        setTotal(r.stats.total)
        setMemorySize(r.memory_size)
      })
      .catch(e => { console.error('recallStaleLog failed', e); setEntries([]) })
      .finally(() => setLoading(false))
  }, [nsFilter])

  useEffect(() => { load() }, [load])

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">Recall Stale Log</h2>
        <span className="text-xs text-gray-500">
          {total.toLocaleString()} filter events · ring={memorySize}
        </span>
        <button
          onClick={load}
          className="ml-auto px-3 py-1.5 text-sm rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700">
          Refresh
        </button>
      </div>

      {!enabled ? (
        <div className="bg-gray-900 border border-amber-900/50 rounded-xl px-4 py-3 text-sm text-amber-300">
          ⚠️ Aggregate staleness filter is <span className="font-mono">disabled</span>{' '}
          (<span className="font-mono">v2_aggregate_staleness_filter = false</span>).
          No filtering is happening at recall time.
        </div>
      ) : (
        <div className="bg-gray-900 border border-emerald-900/50 rounded-xl px-4 py-3 text-sm text-emerald-300">
          ✅ Filter is <b>active</b> — drops list-style aggregate memories that are
          rendered stale by newer atomic facts in the same recall result set.
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <input value={nsFilter} onChange={e => setNsFilter(e.target.value)}
          placeholder="namespace filter (exact match)..."
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-600 w-72" />
      </div>

      {/* Entries */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="text-left px-4 py-3 w-32">Action</th>
              <th className="text-left px-4 py-3 w-32">Namespace</th>
              <th className="text-left px-4 py-3 w-32">Stale Memory</th>
              <th className="text-left px-4 py-3">Aggregate Content</th>
              <th className="text-left px-4 py-3 w-64">Why filtered</th>
              <th className="text-left px-4 py-3 w-40">Query</th>
              <th className="text-left px-4 py-3 w-40">When</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : entries.length === 0 ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                No filter events yet. Events appear here when an old aggregate memory
                is filtered because newer atomic facts cover IDs it doesn't enumerate.
              </td></tr>
            ) : entries.map((e, i) => (
              <tr key={`${e.timestamp_ms}-${e.aggregate_id}-${i}`}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    e.action === 'Filter' ? 'bg-rose-900/40 text-rose-300'
                                          : 'bg-amber-900/40 text-amber-300'}`}>
                    {e.action}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs font-mono text-blue-300 truncate max-w-[120px]"
                    title={e.namespace}>{e.namespace || <span className="text-gray-600 italic">—</span>}</td>
                <td className="px-4 py-3 text-xs">
                  <button onClick={() => navigate(`/memories/${e.aggregate_id}`)}
                          className="font-mono text-blue-300 hover:underline">
                    #{e.aggregate_id.slice(-10)}
                  </button>
                  <div className="text-[10px] text-gray-500 mt-0.5">
                    created {fmtCreated(e.aggregate_created_at)}
                  </div>
                </td>
                <td className="px-4 py-3 text-gray-300">
                  <div className="text-xs truncate max-w-md" title={e.aggregate_preview}>
                    {e.aggregate_preview || <span className="text-gray-600 italic">—</span>}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">
                    enumerates: {e.witness_in_aggregate.slice(0, 6).map(id =>
                      <span key={id} className="font-mono mr-1">{id}</span>
                    )}
                    {e.witness_in_aggregate.length > 6 && '…'}
                  </div>
                </td>
                <td className="px-4 py-3 text-xs">
                  <div className="text-amber-300">missing IDs in newer facts:</div>
                  <div className="mt-0.5">
                    {e.witness_in_newer.map(id =>
                      <span key={id} className="font-mono mr-1 text-emerald-300">{id}</span>
                    )}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1">
                    {e.newer_fact_ids.length} newer fact{e.newer_fact_ids.length !== 1 ? 's' : ''}
                  </div>
                </td>
                <td className="px-4 py-3 text-xs text-gray-400 truncate max-w-[160px]" title={e.query}>
                  {e.query}
                </td>
                <td className="px-4 py-3 text-xs text-gray-500">{fmtTime(e.timestamp_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
