import { useEffect, useState, useCallback } from 'react'
import { api, type PipelineStats, type ReconcileLogEntry } from '../api/client'
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6']

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1 text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  )
}

export default function Pipeline() {
  const [stats, setStats] = useState<PipelineStats | null>(null)
  const [log, setLog] = useState<ReconcileLogEntry[]>([])
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      const [s, l] = await Promise.all([api.pipelineStats(), api.reconcileLog(200)])
      setStats(s)
      setLog(l.entries || [])
    } catch (e: any) {
      setError(e.message)
    }
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [load])

  if (error) return <div className="text-red-400 p-8">Failed to load: {error}</div>
  if (!stats) return <div className="text-gray-500 p-8">Loading...</div>

  const r = stats.reconciler
  const opData = [
    { name: 'ADD', value: r.op_add },
    { name: 'REPLACE', value: r.op_replace },
    { name: 'RETRACT', value: r.op_retract },
    { name: 'REINFORCE', value: r.op_reinforce },
    { name: 'NOOP', value: r.op_noop },
  ].filter(d => d.value > 0)

  const latencyData = log.slice(0, 100).reverse().map((e, i) => ({
    idx: i,
    latency: e.latency_ms,
    time: new Date(e.timestamp_ms).toLocaleTimeString(),
  }))

  const successRate = r.llm_invocations > 0
    ? (((r.llm_invocations - r.llm_failures) / r.llm_invocations) * 100).toFixed(1) + '%'
    : 'N/A'

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">Pipeline</h2>

      {/* Stage 2 Queue */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard label="Queue Depth" value={stats.queue_depth} sub={`/ ${stats.queue_capacity}`} />
        <StatCard label="Executor Threads" value={stats.executor_threads} />
        <StatCard label="Tasks Completed" value={stats.tasks_completed} />
        <StatCard label="Tasks Failed" value={stats.tasks_failed} />
        <StatCard label="Reconciler Calls" value={r.total_calls} sub={`LLM: ${r.llm_invocations}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Reconciler Decision Distribution */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Reconciler Decisions</h3>
          <p className="text-xs text-gray-500 mb-4">LLM success rate: {successRate} ({r.llm_failures} failures)</p>
          {opData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={opData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={85} label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}>
                  {opData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-60 flex items-center justify-center text-gray-600">No reconcile decisions yet</div>
          )}
        </div>

        {/* Reconcile Latency Timeline */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Reconcile Latency (ms)</h3>
          {latencyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={latencyData}>
                <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.time || ''} />
                <Line type="monotone" dataKey="latency" stroke="#6366f1" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-60 flex items-center justify-center text-gray-600">No latency data yet</div>
          )}
        </div>
      </div>

      {/* Recent Reconcile Events */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Recent Reconcile Events</h3>
        {log.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-left text-xs">
                  <th className="pb-2 pr-4">Time</th>
                  <th className="pb-2 pr-4">Op</th>
                  <th className="pb-2 pr-4">Latency</th>
                  <th className="pb-2 pr-4">Target</th>
                  <th className="pb-2">Candidate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {log.slice(0, 50).map((e, i) => (
                  <tr key={i} className="hover:bg-gray-800/50">
                    <td className="py-2 pr-4 text-gray-400 text-xs whitespace-nowrap">
                      {new Date(e.timestamp_ms).toLocaleTimeString()}
                    </td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        e.op === 'REPLACE' ? 'bg-amber-900/40 text-amber-300' :
                        e.op === 'ADD' ? 'bg-green-900/40 text-green-300' :
                        e.op === 'RETRACT' ? 'bg-red-900/40 text-red-300' :
                        e.op === 'REINFORCE' ? 'bg-blue-900/40 text-blue-300' :
                        'bg-gray-800 text-gray-400'
                      }`}>
                        {e.op}{e.from_fallback ? ' (fallback)' : ''}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-gray-300 text-xs font-mono">{e.latency_ms}ms</td>
                    <td className="py-2 pr-4 text-gray-500 text-xs font-mono">
                      {e.target_id !== '0' ? e.target_id : '-'}
                    </td>
                    <td className="py-2 text-gray-400 text-xs truncate max-w-xs">{e.candidate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-600 text-center py-6">No reconcile events recorded yet</p>
        )}
      </div>
    </div>
  )
}
