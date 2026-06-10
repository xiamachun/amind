import { useEffect, useRef, useState } from 'react'
import { api, type Memory } from '../api/client'
import { Network, type Options } from 'vis-network'
import { DataSet } from 'vis-data'

export default function Graph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)
  const [stats, setStats] = useState({ nodes: 0, edges: 0 })
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<Memory | null>(null)
  const [agentIds, setAgentIds] = useState<string[]>([])
  const [filterAgent, setFilterAgent] = useState('')
  const [filterUser, setFilterUser] = useState('')
  const [userSuggestions, setUserSuggestions] = useState<string[]>([])

  // Load agent list on mount
  useEffect(() => {
    api.listAgentIds().then(list => setAgentIds(list.map(a => a.agent_id))).catch(() => {})
    // Collect user_id suggestions from a broad memories fetch
    api.listMemories(1, 5000).then(r => {
      const uids = Array.from(new Set((r.memories || []).map(m => m.user_id).filter(Boolean)))
      setUserSuggestions(uids)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!containerRef.current) return
    setLoading(true)

    api.listEdges(1, 2000, filterAgent, filterUser).then(async (res) => {
      const edges = res.edges || []
      const nodeIds = new Set<string>()
      edges.forEach(e => { nodeIds.add(e.from_id); nodeIds.add(e.to_id) })

      // Fetch memory content for labels
      const memMap: Record<string, Memory> = {}
      const ids = Array.from(nodeIds)
      const batchSize = 20
      for (let i = 0; i < Math.min(ids.length, 200); i += batchSize) {
        const batch = ids.slice(i, i + batchSize)
        const results = await Promise.allSettled(batch.map(id => api.getMemory(id)))
        results.forEach((r, j) => {
          if (r.status === 'fulfilled') memMap[batch[j]] = r.value
        })
      }

      const nodeColor: Record<string, string> = {
        private: '#6366f1', agent_shared: '#10b981', system: '#ec4899',
      }

      const nodes = new DataSet(
        ids.map(id => {
          const m = memMap[id]
          const label = m ? m.content.slice(0, 30) + (m.content.length > 30 ? '...' : '') : id.slice(0, 8)
          const color = m ? (nodeColor[m.scope] || '#6b7280') : '#6b7280'
          return { id, label, color, font: { color: '#e5e7eb', size: 11 } }
        })
      )

      const edgeTypeColor: Record<string, string> = {
        semantic: '#6366f1', temporal: '#22d3ee', causal: '#f59e0b',
        references: '#10b981', contradicts: '#ef4444',
      }

      const edgeDS = new DataSet(
        edges.map((e, i) => ({
          id: i,
          from: e.from_id,
          to: e.to_id,
          label: e.type,
          color: { color: edgeTypeColor[e.type] || '#4b5563', opacity: 0.7 },
          font: { color: '#6b7280', size: 9 },
          width: Math.max(1, e.weight * 3),
          arrows: 'to',
        }))
      )

      const options: Options = {
        physics: {
          solver: 'forceAtlas2Based',
          forceAtlas2Based: { gravitationalConstant: -40, springLength: 120, damping: 0.5 },
          stabilization: { iterations: 150 },
        },
        interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
        nodes: {
          shape: 'dot', size: 14,
          borderWidth: 1, borderWidthSelected: 3,
          font: { color: '#e5e7eb', size: 11 },
        },
        edges: {
          smooth: { enabled: true, type: 'continuous', roundness: 0.2 },
        },
      }

      const net = new Network(containerRef.current!, { nodes, edges: edgeDS }, options)
      networkRef.current = net
      setStats({ nodes: ids.length, edges: edges.length })
      setLoading(false)

      net.on('click', (params) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0] as string
          if (memMap[nodeId]) setSelected(memMap[nodeId])
        } else {
          setSelected(null)
        }
      })
    }).catch(() => setLoading(false))

    return () => { networkRef.current?.destroy() }
  }, [filterAgent, filterUser])

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex items-center gap-4 flex-wrap">
        <h2 className="text-lg font-semibold text-gray-200">Knowledge Graph</h2>
        <span className="text-xs text-gray-500">{stats.nodes} nodes, {stats.edges} edges</span>
        {loading && <span className="text-xs text-indigo-400 animate-pulse">Loading graph...</span>}

        <div className="flex items-center gap-2 ml-auto">
          <select value={filterAgent} onChange={e => setFilterAgent(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200
                       focus:outline-none focus:ring-2 focus:ring-indigo-500">
            <option value="">All Agents</option>
            {agentIds.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
          <input type="text" placeholder="Filter by user..."
            list="graph-user-list"
            value={filterUser}
            onChange={e => setFilterUser(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 w-40
                       focus:outline-none focus:ring-2 focus:ring-indigo-500" />
          <datalist id="graph-user-list">
            {userSuggestions.map(u => <option key={u} value={u} />)}
          </datalist>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex gap-4">
        <div ref={containerRef}
          className="flex-1 bg-gray-900 border border-gray-800 rounded-xl overflow-hidden" />

        {selected && (
          <div className="w-72 bg-gray-900 border border-gray-800 rounded-xl p-4 overflow-y-auto space-y-3">
            <h3 className="text-sm font-medium text-gray-300">Selected Memory</h3>
            <p className="text-xs text-gray-400 whitespace-pre-wrap">{selected.content}</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div><span className="text-gray-500">Agent:</span> <span className="text-indigo-300">{selected.agent_id}</span></div>
              <div><span className="text-gray-500">Phase:</span> <span className="text-gray-300">{selected.phase}</span></div>
              <div><span className="text-gray-500">Score:</span> <span className="text-gray-300">{selected.importance.toFixed(3)}</span></div>
              <div><span className="text-gray-500">Version:</span> <span className="text-gray-300">{selected.version}</span></div>
            </div>
            <p className="text-xs text-gray-600 font-mono break-all">{selected.memory_id}</p>
          </div>
        )}
      </div>
    </div>
  )
}
