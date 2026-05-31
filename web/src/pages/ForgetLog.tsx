import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  api,
  type ForgetLogEntry,
  type ForgetLogStats,
  type ForgetLogConfig,
} from '../api/client'

const decisionColors: Record<string, string> = {
  Decay:             'bg-sky-900/40 text-sky-300',
  Archive:           'bg-amber-900/40 text-amber-300',
  Tombstone:         'bg-rose-900/40 text-rose-300',
  Vacuum:            'bg-fuchsia-900/40 text-fuchsia-300',
  DropFromHNSW:      'bg-violet-900/40 text-violet-300',
  ResolveConflict:   'bg-emerald-900/40 text-emerald-300',
  LineageInvalidate: 'bg-orange-900/40 text-orange-300',
  GateReject:        'bg-rose-900/40 text-rose-300',
  GateDefer:         'bg-amber-900/40 text-amber-300',
}

const decisionExplain: Record<string, string> = {
  Decay:     'importance ↓ — record stays Active but ranks lower in recall',
  Archive:   'phase → Archived — cold-store, not in default recall',
  Tombstone: 'phase → Tombstone — soft-deleted, recoverable until LSM compaction',
  Vacuum:    'physically removed by LSM compaction — no longer recoverable',
}

const fmtTime = (ms: number) => ms ? new Date(ms).toLocaleString() : '-'

export default function ForgetLog() {
  const navigate = useNavigate()
  const [entries, setEntries] = useState<ForgetLogEntry[]>([])
  const [stats, setStats] = useState<ForgetLogStats>({
    decay: 0, archive: 0, tombstone: 0, vacuum: 0, other: 0, total: 0,
  })
  const [config, setConfig] = useState<ForgetLogConfig | null>(null)
  const [memorySize, setMemorySize] = useState(0)
  const [loading, setLoading] = useState(false)
  const [decisionFilter, setDecisionFilter] = useState<string>('')

  const load = useCallback(() => {
    setLoading(true)
    api.forgetLog({
      limit: 200,
      decision: decisionFilter || undefined,
    }).then(r => {
      setEntries(r.entries)
      setStats(r.stats)
      setConfig(r.config)
      setMemorySize(r.memory_size)
    }).catch(e => {
      console.error('forgetLog load failed', e)
      setEntries([])
    }).finally(() => setLoading(false))
  }, [decisionFilter])

  useEffect(() => { load() }, [load])

  const enabled  = config?.enabled ?? false
  const shadow   = config?.shadow_mode ?? true

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">Forget Log</h2>
        <span className="text-xs text-gray-500">
          {stats.total.toLocaleString()} decisions · ring={memorySize}
        </span>
        <button
          onClick={load}
          className="ml-auto px-3 py-1.5 text-sm rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700">
          Refresh
        </button>
      </div>

      {/* Engine state banner */}
      {!enabled ? (
        <div className="bg-gray-900 border border-amber-900/50 rounded-xl px-4 py-3 text-sm text-amber-300">
          ⚠️ ForgetEngine is <span className="font-mono">disabled</span>{' '}
          (<span className="font-mono">v2_forget_score_enabled = false</span>).
          Enable it in <span className="font-mono">amind.conf</span> and restart to start collecting GC decisions.
        </div>
      ) : shadow ? (
        <div className="bg-gray-900 border border-sky-900/50 rounded-xl px-4 py-3 text-sm text-sky-300">
          🛡️ Engine running in <b>Shadow mode</b> — decisions are computed and logged here, but
          memories are <b>not</b> actually modified. Set{' '}
          <span className="font-mono">v2_global_shadow_mode = false</span> to enforce.
        </div>
      ) : (
        <div className="bg-gray-900 border border-emerald-900/50 rounded-xl px-4 py-3 text-sm text-emerald-300">
          ✅ Engine is <b>active</b> — Tombstone decisions soft-delete memories. Recovery is possible
          until LSM compaction physically purges the row.
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-5 gap-3">
        <Stat label="Decay"     value={stats.decay}     color="text-sky-400" />
        <Stat label="Archive"   value={stats.archive}   color="text-amber-400" />
        <Stat label="Tombstone" value={stats.tombstone} color="text-rose-400" />
        <Stat label="Vacuum"    value={stats.vacuum}    color="text-fuchsia-400" />
        <Stat label="Other"     value={stats.other}     color="text-gray-400" />
      </div>

      {/* Config card */}
      {config && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 grid grid-cols-2 md:grid-cols-5 gap-x-6 gap-y-2 text-xs">
          <Cell k="decay ≥"     v={config.decay_threshold.toFixed(2)} />
          <Cell k="archive ≥"   v={config.archive_threshold.toFixed(2)} />
          <Cell k="tombstone ≥" v={config.tombstone_threshold.toFixed(2)} />
          <Cell k="GC every"    v={`${config.gc_interval_seconds}s`} />
          <Cell k="sample"      v={`${(config.sample_ratio * 100).toFixed(0)}%`} />
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <select value={decisionFilter} onChange={e => setDecisionFilter(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300">
          <option value="">All decisions</option>
          <option value="Decay">Decay</option>
          <option value="Archive">Archive</option>
          <option value="Tombstone">Tombstone</option>
          <option value="Vacuum">Vacuum</option>
        </select>
      </div>

      {/* Entries */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="text-left px-4 py-3 w-28">Decision</th>
              <th className="text-left px-4 py-3 w-32">Namespace</th>
              <th className="text-left px-4 py-3 w-36">Memory</th>
              <th className="text-left px-4 py-3">Content</th>
              <th className="text-left px-4 py-3 w-32">State</th>
              <th className="text-left px-4 py-3 w-56">Reason</th>
              <th className="text-left px-4 py-3 w-40">When</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : entries.length === 0 ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                No GC decisions yet. {!enabled && 'Enable the engine to populate this log.'}
              </td></tr>
            ) : entries.map((e, i) => (
              <tr key={`${e.timestamp_ms}-${e.memory_id}-${i}`}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${decisionColors[e.decision] || 'bg-gray-800 text-gray-300'}`}
                        title={decisionExplain[e.decision] || ''}>
                    {e.decision}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs font-mono text-blue-300 truncate max-w-[120px]"
                    title={e.namespace || '(unknown)'}>
                  {e.namespace || <span className="text-gray-600 italic">unknown</span>}
                </td>
                <td className="px-4 py-3 text-xs">
                  <button onClick={() => navigate(`/memories/${e.memory_id}`)}
                          className="font-mono text-blue-300 hover:underline">
                    #{e.memory_id.slice(-10)}
                  </button>
                </td>
                <td className="px-4 py-3 text-gray-300">
                  <div className="text-xs truncate max-w-md" title={e.content}>
                    {e.content || <span className="text-gray-600 italic">—</span>}
                  </div>
                </td>
                <td className="px-4 py-3 text-xs text-gray-400">
                  <span className="font-mono">{e.before_state}</span>
                  <span className="mx-1 text-gray-600">→</span>
                  <span className={`font-mono ${e.before_state !== e.after_state ? 'text-amber-300' : ''}`}>
                    {e.after_state}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-300">
                  <div className="text-xs italic truncate max-w-[220px]" title={e.reason}>
                    {e.reason || '—'}
                  </div>
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

function Stat({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
      <div className="text-xs text-gray-500 uppercase tracking-wider">{label}</div>
      <div className={`text-2xl font-semibold ${color} mt-1`}>{value.toLocaleString()}</div>
    </div>
  )
}

function Cell({ k, v }: { k: string, v: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-gray-500 uppercase tracking-wider">{k}</span>
      <span className="font-mono text-gray-300">{v}</span>
    </div>
  )
}
