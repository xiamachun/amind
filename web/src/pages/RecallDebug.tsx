import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const SCORE_COLORS: Record<string, string> = {
  semantic: '#6366f1',
  keyword: '#22d3ee',
  graph: '#f59e0b',
  recency: '#10b981',
}

interface DebugResult {
  memory_id: string
  content: string
  score: number
  semantic_score: number
  keyword_score: number
  graph_score: number
  recency_score: number
}

export default function RecallDebug() {
  const [query, setQuery] = useState('')
  const [agentId, setAgentId] = useState('')
  const [results, setResults] = useState<DebugResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const doRecall = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/v1/memories/recall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, agent_id: agentId || undefined, top_k: 10, debug: true }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const items = (data.results || []).map((r: any) => ({
        memory_id: r.memory_id || String(r.memory?.memory_id || ''),
        content: r.memory?.content || r.content || '',
        score: r.score ?? 0,
        semantic_score: r.semantic_score ?? 0,
        keyword_score: r.keyword_score ?? 0,
        graph_score: r.graph_score ?? 0,
        recency_score: r.recency_score ?? 0,
      }))
      setResults(items)
    } catch (e: any) {
      setError(e.message)
    }
    setLoading(false)
  }

  const chartData = results.map((r, i) => ({
    name: `#${i + 1}`,
    semantic: r.semantic_score,
    keyword: r.keyword_score,
    graph: r.graph_score,
    recency: r.recency_score,
    total: r.score,
  }))

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">Recall Debugger</h2>

      {/* Input */}
      <div className="flex gap-3">
        <input value={query} onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') doRecall() }}
          placeholder="Enter recall query..."
          className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-indigo-500 focus:outline-none" />
        <input value={agentId} onChange={e => setAgentId(e.target.value)}
          placeholder="agent_id (optional)"
          className="w-48 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-indigo-500 focus:outline-none" />
        <button onClick={doRecall} disabled={loading}
          className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors">
          {loading ? 'Searching...' : 'Debug Recall'}
        </button>
      </div>

      {error && <div className="text-red-400 text-sm">{error}</div>}

      {/* Score Breakdown Chart */}
      {chartData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Score Breakdown (stacked)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="semantic" stackId="a" fill={SCORE_COLORS.semantic} name="Semantic" />
              <Bar dataKey="keyword" stackId="a" fill={SCORE_COLORS.keyword} name="Keyword" />
              <Bar dataKey="graph" stackId="a" fill={SCORE_COLORS.graph} name="Graph" />
              <Bar dataKey="recency" stackId="a" fill={SCORE_COLORS.recency} name="Recency" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Results Table */}
      {results.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Results ({results.length})</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-left text-xs">
                  <th className="pb-2 pr-3">#</th>
                  <th className="pb-2 pr-3">Score</th>
                  <th className="pb-2 pr-3">Semantic</th>
                  <th className="pb-2 pr-3">Keyword</th>
                  <th className="pb-2 pr-3">Graph</th>
                  <th className="pb-2 pr-3">Recency</th>
                  <th className="pb-2">Content</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {results.map((r, i) => (
                  <tr key={i} className="hover:bg-gray-800/50">
                    <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
                    <td className="py-2 pr-3 text-white font-mono text-xs">{r.score.toFixed(3)}</td>
                    <td className="py-2 pr-3 text-indigo-300 font-mono text-xs">{r.semantic_score.toFixed(3)}</td>
                    <td className="py-2 pr-3 text-cyan-300 font-mono text-xs">{r.keyword_score.toFixed(3)}</td>
                    <td className="py-2 pr-3 text-amber-300 font-mono text-xs">{r.graph_score.toFixed(3)}</td>
                    <td className="py-2 pr-3 text-emerald-300 font-mono text-xs">{r.recency_score.toFixed(3)}</td>
                    <td className="py-2 text-gray-300 text-xs truncate max-w-md">{r.content}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
