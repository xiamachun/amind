import { useEffect, useState } from 'react'
import { api, type SessionSummary } from '../api/client'

export default function Sessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.listSessions()
      .then(setSessions)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const fmtDuration = (start: number, last: number) => {
    if (!start || !last) return '-'
    const s = last - start
    if (s < 60) return `${s}s`
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-200">Sessions</h2>

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="text-left px-4 py-3">Session ID</th>
              <th className="text-left px-4 py-3">Agent</th>
              <th className="text-left px-4 py-3 w-20">Turns</th>
              <th className="text-left px-4 py-3 w-20">Memories</th>
              <th className="text-left px-4 py-3 w-20">Facts</th>
              <th className="text-left px-4 py-3">Intent</th>
              <th className="text-left px-4 py-3">Duration</th>
              <th className="text-left px-4 py-3 w-20">Status</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : sessions.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-500">No sessions</td></tr>
            ) : sessions.map(s => (
              <tr key={s.session_id} className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors">
                <td className="px-4 py-3 text-gray-400 font-mono text-xs">{s.session_id.slice(0, 12)}...</td>
                <td className="px-4 py-3 text-gray-300 text-xs">{s.agent_id || '-'}</td>
                <td className="px-4 py-3 text-gray-300">{s.turn_count}</td>
                <td className="px-4 py-3 text-gray-300">{s.memory_count}</td>
                <td className="px-4 py-3 text-gray-300">{s.fact_count}</td>
                <td className="px-4 py-3 text-gray-400 text-xs truncate max-w-xs">{s.current_intent || '-'}</td>
                <td className="px-4 py-3 text-gray-500 text-xs">{fmtDuration(s.started_at, s.last_turn_at)}</td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    s.active ? 'bg-green-900/50 text-green-300' : 'bg-gray-800 text-gray-500'
                  }`}>
                    {s.active ? 'Active' : 'Ended'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
