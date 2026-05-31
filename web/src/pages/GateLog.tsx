import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type GateLogEntry, type GateLogStats } from '../api/client'

type Strategy = 'coexist' | 'replace_conflict' | 'update_existing'

const strategyLabels: Record<Strategy, string> = {
  coexist: 'Coexist (keep both)',
  replace_conflict: 'Replace conflict (tombstone the old one)',
  update_existing: 'Update existing (no new record)',
}

const decisionColors: Record<string, string> = {
  Accepted: 'bg-emerald-900/40 text-emerald-300',
  Rejected: 'bg-rose-900/40 text-rose-300',
  Deferred: 'bg-amber-900/40 text-amber-300',
}

const fmtTime = (ms: number) => ms ? new Date(ms).toLocaleString() : '-'

export default function GateLog() {
  const navigate = useNavigate()
  const [entries, setEntries] = useState<GateLogEntry[]>([])
  const [stats, setStats] = useState<GateLogStats>({
    accepted: 0, rejected: 0, deferred: 0, resurrected: 0, total: 0,
  })
  const [loading, setLoading] = useState(false)
  const [decisionFilter, setDecisionFilter] = useState<string>('') // '' = Rejected+Deferred
  const [nsFilter, setNsFilter] = useState('')
  const [showAccepted, setShowAccepted] = useState(false)
  const [resurrectTarget, setResurrectTarget] = useState<GateLogEntry | null>(null)
  const [strategy, setStrategy] = useState<Strategy>('coexist')
  const [resurrectMsg, setResurrectMsg] = useState('')

  const load = useCallback(() => {
    setLoading(true)
    const decision = decisionFilter || (showAccepted ? '' : '')
    api.gateLog({
      limit: 200,
      decision: decision || undefined,
      namespace: nsFilter || undefined,
    }).then(r => {
      let rows = r.entries
      // Default view: hide Accepted unless toggled (Accepted volume dwarfs the rest)
      if (!showAccepted && !decisionFilter) {
        rows = rows.filter(e => e.decision !== 'Accepted')
      }
      setEntries(rows)
      setStats(r.stats)
    }).finally(() => setLoading(false))
  }, [decisionFilter, nsFilter, showAccepted])

  useEffect(() => { load() }, [load])

  async function doResurrect() {
    if (!resurrectTarget) return
    try {
      const resp = await api.gateResurrect(resurrectTarget.entry_id, strategy)
      setResurrectMsg(`✅ Resurrected to memory #${resp.memory_id}` +
                      (resp.replaced_id ? ` (replaced #${resp.replaced_id})` : ''))
      setTimeout(() => {
        setResurrectMsg('')
        setResurrectTarget(null)
        load()
      }, 1800)
    } catch (e: any) {
      setResurrectMsg(`❌ ${e?.message || 'resurrect failed'}`)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">Gate Log</h2>
        <span className="text-xs text-gray-500">
          {stats.total.toLocaleString()} verdicts · {stats.resurrected} resurrected · ring={entries.length}
        </span>
        <button
          onClick={load}
          className="ml-auto px-3 py-1.5 text-sm rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700">
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">Accepted</div>
          <div className="text-2xl font-semibold text-emerald-400 mt-1">{stats.accepted.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">Rejected</div>
          <div className="text-2xl font-semibold text-rose-400 mt-1">{stats.rejected.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">Deferred</div>
          <div className="text-2xl font-semibold text-amber-400 mt-1">{stats.deferred.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">Resurrected</div>
          <div className="text-2xl font-semibold text-blue-400 mt-1">{stats.resurrected.toLocaleString()}</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <select value={decisionFilter} onChange={e => setDecisionFilter(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300">
          <option value="">All decisions</option>
          <option value="Accepted">Accepted</option>
          <option value="Rejected">Rejected</option>
          <option value="Deferred">Deferred</option>
        </select>
        <input value={nsFilter} onChange={e => setNsFilter(e.target.value)}
          placeholder="namespace filter (exact match)..."
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-600 w-72" />
        {!decisionFilter && (
          <label className="flex items-center gap-2 text-sm text-gray-400 ml-auto">
            <input type="checkbox" checked={showAccepted} onChange={e => setShowAccepted(e.target.checked)} />
            Show Accepted
          </label>
        )}
      </div>

      {/* Entries */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="text-left px-4 py-3 w-28">Decision</th>
              <th className="text-left px-4 py-3">Content</th>
              <th className="text-left px-4 py-3 w-36">Namespace</th>
              <th className="text-left px-4 py-3 w-20">Layer</th>
              <th className="text-left px-4 py-3 w-44">Created</th>
              <th className="text-left px-4 py-3 w-32">Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : entries.length === 0 ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-500">No entries match the filter.</td></tr>
            ) : entries.map(e => (
              <tr key={e.entry_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${decisionColors[e.decision] || ''}`}>
                    {e.decision}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-300 max-w-md">
                  <div className="truncate" title={e.content}>{e.content}</div>
                  <div className="text-xs text-gray-500 mt-1 italic truncate" title={e.reason}>
                    {e.reason || '—'}
                  </div>
                </td>
                <td className="px-4 py-3 text-xs font-mono text-blue-300 truncate max-w-[140px]" title={e.namespace}>
                  {e.namespace}
                </td>
                <td className="px-4 py-3 text-xs text-gray-500">{e.layer}</td>
                <td className="px-4 py-3 text-xs text-gray-500">{fmtTime(e.timestamp_ms)}</td>
                <td className="px-4 py-3">
                  {e.resurrected_to !== '0' ? (
                    <button
                      onClick={() => navigate(`/memories/${e.resurrected_to}`)}
                      className="text-xs px-2 py-1 rounded bg-blue-900/50 text-blue-300 hover:bg-blue-900">
                      ✓ → #{e.resurrected_to.slice(-6)}
                    </button>
                  ) : e.decision === 'Accepted' ? (
                    <span className="text-xs text-gray-600">—</span>
                  ) : (
                    <button
                      onClick={() => { setResurrectTarget(e); setStrategy('coexist'); setResurrectMsg('') }}
                      className="text-xs px-3 py-1 rounded bg-indigo-700 hover:bg-indigo-600 text-white">
                      Resurrect ⤴
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Resurrect dialog */}
      {resurrectTarget && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-[480px] p-5 space-y-4">
            <h3 className="text-base font-semibold text-gray-200">Resurrect rejected memory</h3>
            <div className="bg-gray-800 rounded-lg p-3 text-sm text-gray-300 max-h-32 overflow-y-auto">
              {resurrectTarget.content}
            </div>
            <div className="text-xs text-gray-500">
              Reason: <span className="font-mono">{resurrectTarget.reason || '—'}</span>
              {resurrectTarget.conflict_with_id !== '0' && (
                <>
                  <br />Conflict target:
                  <button onClick={() => navigate(`/memories/${resurrectTarget.conflict_with_id}`)}
                    className="ml-1 text-indigo-400 underline">
                    #{resurrectTarget.conflict_with_id.slice(-8)}
                  </button>
                </>
              )}
            </div>

            <div className="space-y-2">
              <div className="text-sm text-gray-400">Strategy:</div>
              {(['coexist', 'replace_conflict', 'update_existing'] as Strategy[]).map(s => (
                <label key={s} className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                  <input type="radio" name="strategy" checked={strategy === s}
                    onChange={() => setStrategy(s)}
                    disabled={s !== 'coexist' && resurrectTarget.conflict_with_id === '0'} />
                  <span>{strategyLabels[s]}</span>
                  {s !== 'coexist' && resurrectTarget.conflict_with_id === '0' && (
                    <span className="text-xs text-gray-600">(no conflict target)</span>
                  )}
                </label>
              ))}
            </div>

            {resurrectMsg && (
              <div className={`text-sm rounded p-2 ${
                resurrectMsg.startsWith('✅') ? 'bg-emerald-900/40 text-emerald-300'
                                              : 'bg-rose-900/40 text-rose-300'}`}>
                {resurrectMsg}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <button onClick={() => { setResurrectTarget(null); setResurrectMsg('') }}
                className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700">
                Cancel
              </button>
              <button onClick={doResurrect}
                className="px-3 py-1.5 text-sm rounded bg-indigo-700 hover:bg-indigo-600 text-white">
                Confirm Resurrect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
