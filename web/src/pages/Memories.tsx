import { useEffect, useState, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, type Memory } from '../api/client'

export default function Memories() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [memories, setMemories] = useState<Memory[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)

  // Store form state
  const [showForm, setShowForm] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newOwner, setNewOwner] = useState('user')
  const [storing, setStoring] = useState(false)
  const [storeMsg, setStoreMsg] = useState('')

  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [deleting, setDeleting] = useState(false)
  const [confirmAction, setConfirmAction] = useState<'archive' | 'delete' | null>(null)

  const page = Number(searchParams.get('page') || '1')
  const perPage = 20
  const owner = searchParams.get('owner') || ''
  const phase = searchParams.get('phase') || ''
  const q = searchParams.get('q') || ''
  const agent_id = searchParams.get('agent_id') || ''
  const userId = searchParams.get('user_id') || ''
  const layer = searchParams.get('layer') || ''   // '' | 'Raw' | 'Derived'

  const loadMemories = useCallback(() => {
    setLoading(true)
    api.listMemories(page, perPage, owner, phase, q, agent_id, userId, layer)
      .then(r => { setMemories(r.memories || []); setTotal(r.total) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [page, owner, phase, q, agent_id, userId, layer])

  useEffect(() => { loadMemories() }, [loadMemories])

  const totalPages = Math.ceil(total / perPage)

  const updateFilter = (key: string, val: string) => {
    const p = new URLSearchParams(searchParams)
    if (val) p.set(key, val); else p.delete(key)
    p.set('page', '1')
    setSearchParams(p)
  }

  const fmtTime = (ts: number) => ts ? new Date(ts * 1000).toLocaleString() : '-'

  const handleStore = async () => {
    if (!newContent.trim()) return
    setStoring(true)
    setStoreMsg('')
    try {
      const res = await api.storeMemory(newContent.trim(), newOwner)
      setStoreMsg(`Stored! IDs: ${(res.memory_ids || []).join(', ')}`)
      setNewContent('')
      setTimeout(() => { setStoreMsg(''); setShowForm(false); loadMemories() }, 1500)
    } catch (e) {
      setStoreMsg(`Error: ${(e as Error).message}`)
    } finally {
      setStoring(false)
    }
  }

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  const toggleAll = () => {
    if (selected.size === memories.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(memories.map(m => m.memory_id)))
    }
  }

  const executeBatchAction = async (action: 'archive' | 'delete') => {
    setDeleting(true)
    setConfirmAction(null)
    try {
      const fn = action === 'archive' ? api.archiveMemory : api.deleteMemory
      await Promise.all([...selected].map(id => fn(id)))
    } catch { /* ignore individual failures */ }
    setSelected(new Set())
    setDeleting(false)
    loadMemories()
  }

  useEffect(() => { setSelected(new Set()) }, [page])

  const ownerColors: Record<string, string> = {
    User: 'bg-blue-900/50 text-blue-300',
    Project: 'bg-purple-900/50 text-purple-300',
    Agent: 'bg-green-900/50 text-green-300',
    Session: 'bg-yellow-900/50 text-yellow-300',
    Shared: 'bg-pink-900/50 text-pink-300',
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">Memories</h2>
        <button onClick={() => setShowForm(!showForm)}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            showForm ? 'bg-gray-700 text-gray-300' : 'bg-indigo-600 text-white hover:bg-indigo-500'
          }`}>
          {showForm ? 'Cancel' : '+ Store Memory'}
        </button>
      </div>

      {/* Store Memory Form */}
      {showForm && (
        <div className="bg-gray-900 border border-indigo-800/50 rounded-xl p-5 space-y-3">
          <textarea value={newContent} onChange={e => setNewContent(e.target.value)}
            placeholder="Enter memory content... (supports multi-line)"
            rows={4}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y" />
          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-500">Scope:</label>
            <select value={newOwner} onChange={e => setNewOwner(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200">
              <option value="private">Private</option>
              <option value="agent_shared">Agent Shared</option>
              <option value="system">System</option>
            </select>
            <button onClick={handleStore} disabled={storing || !newContent.trim()}
              className="px-4 py-1.5 rounded-lg text-sm bg-indigo-600 text-white hover:bg-indigo-500
                         disabled:opacity-40 disabled:cursor-not-allowed font-medium">
              {storing ? 'Storing...' : 'Store'}
            </button>
            {storeMsg && (
              <span className={`text-xs ${storeMsg.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
                {storeMsg}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Layer tabs — Raw is what the user said, Derived is what amind
          extracted as facts. They behave very differently downstream so it's
          useful to see them separately. */}
      <div className="flex items-center gap-1 bg-gray-900 border border-gray-800 rounded-lg p-1 w-fit">
        {[
          { val: '',        label: 'All',     hint: 'Both raw memories and derived facts' },
          { val: 'Raw',     label: 'Raw',     hint: 'Original user messages — the evidence layer' },
          { val: 'Derived', label: 'Derived', hint: 'Facts extracted from raw — what recall actually uses' },
        ].map(t => (
          <button key={t.val}
                  onClick={() => updateFilter('layer', t.val)}
                  title={t.hint}
                  className={`px-3 py-1 rounded text-xs transition-colors ${
                    layer === t.val
                      ? 'bg-indigo-700 text-white'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                  }`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <input type="text" placeholder="Search content..."
          defaultValue={q}
          onKeyDown={e => e.key === 'Enter' && updateFilter('q', (e.target as HTMLInputElement).value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 w-64
                     focus:outline-none focus:ring-2 focus:ring-indigo-500" />
        <select value={owner} onChange={e => updateFilter('owner', e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200">
          <option value="">All Scopes</option>
          {['User','Project','Agent','Session','Shared'].map(o =>
            <option key={o} value={o}>{o}</option>)}
        </select>
        <select value={phase} onChange={e => updateFilter('phase', e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200">
          <option value="">All Phases</option>
          {['Active','Versioned','Archived','Tombstone'].map(p =>
            <option key={p} value={p}>{p}</option>)}
        </select>
        <input type="text" placeholder="Filter by user..."
          defaultValue={userId}
          onKeyDown={e => e.key === 'Enter' && updateFilter('user_id', (e.target as HTMLInputElement).value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 w-40
                     focus:outline-none focus:ring-2 focus:ring-indigo-500" />
        <input type="text" placeholder="agent_id..."
          defaultValue={agent_id}
          onKeyDown={e => e.key === 'Enter' && updateFilter('agent_id', (e.target as HTMLInputElement).value)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 w-36
                     focus:outline-none focus:ring-2 focus:ring-indigo-500" />
        <button onClick={loadMemories} title="Refresh"
          className="bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-2.5 py-2 text-gray-300
                     transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
          </svg>
        </button>
        <span className="text-xs text-gray-500 ml-auto flex items-center gap-2">
          {selected.size > 0 && !deleting && (
            <>
              <button onClick={() => setConfirmAction('archive')}
                className="px-3 py-1 rounded-lg text-xs font-medium bg-yellow-900/50 text-yellow-300
                           hover:bg-yellow-800/60 border border-yellow-700/50 transition-colors">
                Archive ({selected.size})
              </button>
              <button onClick={() => setConfirmAction('delete')}
                className="px-3 py-1 rounded-lg text-xs font-medium bg-red-900/60 text-red-300
                           hover:bg-red-800/80 border border-red-700/50 transition-colors">
                Delete ({selected.size})
              </button>
            </>
          )}
          {deleting && <span className="text-yellow-400">Processing...</span>}
          {total} records
        </span>
      </div>

      {/* Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase">
              <th className="px-3 py-3 w-10">
                <input type="checkbox"
                  checked={memories.length > 0 && selected.size === memories.length}
                  onChange={toggleAll}
                  className="rounded border-gray-600 bg-gray-800 text-indigo-500 focus:ring-indigo-500 cursor-pointer" />
              </th>
              <th className="text-left px-4 py-3">Content</th>
              <th className="text-left px-4 py-3 w-32">User / Session</th>
              <th className="text-left px-4 py-3 w-20">Layer</th>
              <th className="text-left px-4 py-3 w-24">Scope</th>
              <th className="text-left px-4 py-3 w-24">Phase</th>
              <th className="text-left px-4 py-3 w-20">Score</th>
              <th className="text-left px-4 py-3 w-36">Created</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
            ) : memories.length === 0 ? (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-500">No memories found</td></tr>
            ) : memories.map(m => {
              const meta = m.metadata || {}
              const userTag = meta.user_id || meta.session_id || ''
              const sessionTag = meta.session_id && meta.session_id !== userTag ? meta.session_id : ''
              return (
              <tr key={m.memory_id}
                onClick={() => navigate(`/memories/${m.memory_id}`)}
                className="border-b border-gray-800/50 hover:bg-gray-800/40 cursor-pointer transition-colors">
                <td className="px-3 py-3" onClick={e => e.stopPropagation()}>
                  <input type="checkbox"
                    checked={selected.has(m.memory_id)}
                    onChange={() => toggleSelect(m.memory_id)}
                    className="rounded border-gray-600 bg-gray-800 text-indigo-500 focus:ring-indigo-500 cursor-pointer" />
                </td>
                <td className="px-4 py-3 text-gray-300 truncate max-w-md">
                  {m.content.length > 100 ? m.content.slice(0, 100) + '...' : m.content}
                </td>
                <td className="px-4 py-3 text-xs">
                  {userTag ? (
                    <div className="flex flex-col gap-0.5">
                      <span className="font-mono text-blue-300 truncate max-w-[110px]" title={userTag}>{userTag}</span>
                      {sessionTag && (
                        <span className="font-mono text-gray-500 truncate max-w-[110px]" title={sessionTag}>{sessionTag}</span>
                      )}
                    </div>
                  ) : (
                    <span className="text-gray-600">—</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  {m.layer === 'Derived' ? (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-violet-900/40 text-violet-300"
                          title="Fact extracted by Stage 2 — what recall actually surfaces">
                      Derived
                    </span>
                  ) : m.layer === 'Raw' ? (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400"
                          title="Original user message — the evidence layer">
                      Raw
                    </span>
                  ) : <span className="text-xs text-gray-600">—</span>}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${ownerColors[m.owner] || 'bg-gray-800 text-gray-400'}`}>
                    {m.owner}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-400 text-xs">{m.phase}</td>
                <td className="px-4 py-3 text-gray-400">{m.importance.toFixed(2)}</td>
                <td className="px-4 py-3 text-gray-500 text-xs">{fmtTime(m.created_at)}</td>
              </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button disabled={page <= 1}
            onClick={() => { const p = new URLSearchParams(searchParams); p.set('page', String(page - 1)); setSearchParams(p) }}
            className="px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-300 disabled:opacity-40 hover:bg-gray-700">
            Prev
          </button>
          <span className="text-sm text-gray-500">Page {page} / {totalPages}</span>
          <button disabled={page >= totalPages}
            onClick={() => { const p = new URLSearchParams(searchParams); p.set('page', String(page + 1)); setSearchParams(p) }}
            className="px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-300 disabled:opacity-40 hover:bg-gray-700">
            Next
          </button>
        </div>
      )}
      {/* Confirm Modal */}
      {confirmAction && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
          onClick={() => setConfirmAction(null)}>
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 space-y-4"
            onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-gray-200">
              {confirmAction === 'archive' ? 'Archive Memories' : 'Delete Memories'}
            </h3>
            <p className="text-sm text-gray-400">
              {selected.size} records selected.
            </p>
            {confirmAction === 'archive' ? (
              <div className="bg-yellow-900/20 border border-yellow-800/50 rounded-lg p-3 text-sm text-yellow-200/80 space-y-1">
                <p><strong>Archive</strong> will:</p>
                <ul className="list-disc list-inside space-y-0.5 text-xs">
                  <li>Mark selected records as <span className="text-yellow-300">Archived</span></li>
                  <li>Remove from search results (Recall)</li>
                  <li><strong>Not</strong> affect related Derived records</li>
                  <li>Can be restored to Active later</li>
                </ul>
              </div>
            ) : (
              <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-3 text-sm text-red-200/80 space-y-1">
                <p><strong>Delete</strong> will:</p>
                <ul className="list-disc list-inside space-y-0.5 text-xs">
                  <li>Mark selected records as <span className="text-red-300">Tombstone</span> (permanent)</li>
                  <li>Cascade: related Derived records will be <span className="text-red-300">Invalidated</span></li>
                  <li>All affected records removed from search</li>
                  <li>Cannot be easily undone</li>
                </ul>
              </div>
            )}
            <div className="flex justify-end gap-3 pt-2">
              <button onClick={() => setConfirmAction(null)}
                className="px-4 py-2 rounded-lg text-sm bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
                Cancel
              </button>
              <button onClick={() => executeBatchAction(confirmAction)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  confirmAction === 'archive'
                    ? 'bg-yellow-700 text-yellow-100 hover:bg-yellow-600'
                    : 'bg-red-700 text-red-100 hover:bg-red-600'
                }`}>
                {confirmAction === 'archive' ? 'Confirm Archive' : 'Confirm Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
