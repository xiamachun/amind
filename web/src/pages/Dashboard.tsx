import { useEffect, useState } from 'react'
import { api, type Metrics, type CoverageStats } from '../api/client'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const COLORS = ['#6366f1','#22d3ee','#f59e0b','#ef4444','#10b981','#8b5cf6','#ec4899','#14b8a6']

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1 text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  )
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [coverage, setCoverage] = useState<CoverageStats | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api.metrics(), api.coverage()])
      .then(([m, c]) => { setMetrics(m); setCoverage(c) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="text-red-400 p-8">Failed to load: {error}</div>
  if (!metrics || !coverage) return <div className="text-gray-500 p-8">Loading...</div>

  const reuseRate = metrics.pool_total_acquired > 0
    ? ((metrics.pool_total_reused / metrics.pool_total_acquired) * 100).toFixed(1) + '%'
    : 'N/A'

  const ownerData = Object.entries(coverage.scope_distribution || {}).map(([name, value]) => ({ name, value }))
  const phaseData = Object.entries(coverage.phase_distribution || {}).map(([name, value]) => ({ name, value }))

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">Dashboard</h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Memories" value={coverage.total} sub={`${coverage.active} active`} />
        <StatCard label="Graph Edges" value={metrics.graph_edges} />
        <StatCard label="Pool Connections" value={metrics.pool_connections}
          sub={metrics.pool_circuit_open ? 'Circuit OPEN' : 'Circuit closed'} />
        <StatCard label="Pool Reuse Rate" value={reuseRate}
          sub={`${metrics.pool_total_reused}/${metrics.pool_total_acquired}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Memories by Scope</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={ownerData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                outerRadius={90} label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}>
                {ownerData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Memories by Phase</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={phaseData}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="value" name="Count" radius={[6, 6, 0, 0]}>
                {phaseData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Confidence Distribution</h3>
        <div className="flex gap-4">
          {Object.entries(coverage.confidence_distribution || {}).map(([k, v]) => (
            <div key={k} className="flex-1 text-center">
              <p className="text-xl font-bold text-white">{v}</p>
              <p className="text-xs text-gray-500 mt-1">{k}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
