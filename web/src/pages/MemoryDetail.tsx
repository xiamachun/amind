import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, type Memory, type GraphEdge } from '../api/client'

export default function MemoryDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [memory, setMemory] = useState<Memory | null>(null)
  const [history, setHistory] = useState<Memory[]>([])
  const [neighbors, setNeighbors] = useState<GraphEdge[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!id) return
    api.getMemory(id).then(setMemory).catch(e => setError(e.message))
    api.getHistory(id).then(setHistory).catch(() => {})
    api.getNeighbors(id).then(setNeighbors).catch(() => {})
  }, [id])

  const fmtTime = (ts: number) => ts ? new Date(ts * 1000).toLocaleString() : '-'

  const handleDelete = async () => {
    if (!id || !confirm('Delete this memory?')) return
    await api.deleteMemory(id)
    navigate('/memories')
  }

  if (error) return <div className="text-red-400 p-8">Error: {error}</div>
  if (!memory) return <div className="text-gray-500 p-8">Loading...</div>

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
            <p className="text-xs text-gray-500">Owner</p>
            <p className="text-sm text-indigo-300">{memory.owner}</p>
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

      {/* Graph Neighbors */}
      {neighbors.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Graph Neighbors ({neighbors.length})</h3>
          <div className="space-y-2">
            {neighbors.map((e, i) => {
              const other = e.from_id === id ? e.to_id : e.from_id
              return (
                <div key={i} onClick={() => navigate(`/memories/${other}`)}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 cursor-pointer">
                  <span className="text-xs px-2 py-0.5 rounded bg-indigo-900/50 text-indigo-300">{e.type}</span>
                  <span className="text-xs text-gray-400 font-mono truncate flex-1">{other}</span>
                  <span className="text-xs text-gray-500">w={e.weight.toFixed(2)}</span>
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
    </div>
  )
}
