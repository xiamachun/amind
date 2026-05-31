import { useEffect, useState } from 'react'
import { api, type HealthResponse, type Metrics, type ConflictInfo } from '../api/client'

export default function System() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [conflicts, setConflicts] = useState<ConflictInfo[]>([])
  const [backupStatus, setBackupStatus] = useState('')

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    api.metrics().then(setMetrics).catch(() => {})
    api.conflicts()
      .then(data => {
        // Handle both raw array and wrapped {conflicts: [...]} format
        const arr = Array.isArray(data) ? data : (data as any)?.conflicts || []
        setConflicts(arr.slice(0, 50))  // Limit to 50 for performance
      })
      .catch(() => setConflicts([]))
  }, [])

  const doBackup = async (type: string) => {
    setBackupStatus('Exporting...')
    try {
      await api.backupExport(type)
      setBackupStatus(`${type} backup exported successfully`)
    } catch (e) {
      setBackupStatus(`Error: ${(e as Error).message}`)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-lg font-semibold text-gray-200">System</h2>

      {/* Health */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Health</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-xs text-gray-500">Status</p>
            <p className={health?.status === 'ok' ? 'text-green-400' : 'text-red-400'}>
              {health?.status || 'unknown'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Version</p>
            <p className="text-gray-300">{health?.version || '-'}</p>
          </div>
        </div>
      </div>

      {/* Connection Pool */}
      {metrics && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Connection Pool</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-xs text-gray-500">Active Connections</p>
              <p className="text-xl font-bold text-white">{metrics.pool_connections}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Total Acquired</p>
              <p className="text-xl font-bold text-white">{metrics.pool_total_acquired}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Total Reused</p>
              <p className="text-xl font-bold text-white">{metrics.pool_total_reused}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Circuit Breaker</p>
              <p className={`text-sm font-medium ${metrics.pool_circuit_open ? 'text-red-400' : 'text-green-400'}`}>
                {metrics.pool_circuit_open ? 'OPEN (failing)' : 'Closed (healthy)'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Conflicts */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          Conflicts ({conflicts.length})
        </h3>
        {conflicts.length === 0 ? (
          <p className="text-sm text-gray-500">No conflicts detected</p>
        ) : (
          <div className="space-y-2">
            {conflicts.map((c, i) => (
              <div key={i} className="px-3 py-2 rounded-lg bg-red-900/20 border border-red-900/30">
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-red-400 font-medium">{c.conflict_type || 'Conflict'}</span>
                  <span className="text-gray-500">{String(c.memory_a).slice(0, 12)} vs {String(c.memory_b).slice(0, 12)}</span>
                </div>
                <p className="text-xs text-gray-400 mt-1">{c.explanation || ''}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Backup */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Backup / Export</h3>
        <div className="flex gap-3">
          <button onClick={() => doBackup('memories')}
            className="px-4 py-2 rounded-lg text-sm bg-indigo-600 text-white hover:bg-indigo-500">
            Export Memories
          </button>
          <button onClick={() => doBackup('graph')}
            className="px-4 py-2 rounded-lg text-sm bg-indigo-600 text-white hover:bg-indigo-500">
            Export Graph
          </button>
          <button onClick={() => doBackup('full')}
            className="px-4 py-2 rounded-lg text-sm bg-indigo-600 text-white hover:bg-indigo-500">
            Full Backup
          </button>
        </div>
        {backupStatus && (
          <p className="text-xs text-gray-400 mt-2">{backupStatus}</p>
        )}
      </div>
    </div>
  )
}
