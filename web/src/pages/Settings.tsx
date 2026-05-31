import { useEffect, useState, useCallback } from 'react'
import { api, type Variable, type ApiKeyInfo } from '../api/client'

const CATEGORIES = [
  { key: '',           label: 'All' },
  { key: 'server',     label: 'Server' },
  { key: 'memory',     label: 'Memory' },
  { key: 'retrieval',  label: 'Retrieval' },
  { key: 'provider',   label: 'Provider' },
  { key: 'lsm',        label: 'LSM' },
  { key: 'hnsw',       label: 'HNSW' },
  { key: 'logging',    label: 'Logging' },
]

export default function Settings() {
  const [tab, setTab] = useState<'variables' | 'apikeys'>('variables')
  const [vars, setVars] = useState<Variable[]>([])
  const [filter, setFilter] = useState('')
  const [catFilter, setCatFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [editName, setEditName] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null)
  const [reloading, setReloading] = useState(false)

  // API Keys state
  const [keys, setKeys] = useState<ApiKeyInfo[]>([])
  const [keysLoading, setKeysLoading] = useState(false)
  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [createdKey, setCreatedKey] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await api.listVariables('%')
      setVars(r.variables || [])
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])

  const loadKeys = useCallback(async () => {
    setKeysLoading(true)
    try {
      const r = await api.listApiKeys()
      setKeys(r.keys || [])
    } catch { /* ignore */ }
    setKeysLoading(false)
  }, [])

  useEffect(() => { if (tab === 'apikeys') loadKeys() }, [tab, loadKeys])

  const handleCreateKey = async () => {
    if (!newKeyLabel.trim()) return
    setCreating(true)
    try {
      const r = await api.createApiKey(newKeyLabel.trim())
      setCreatedKey(r.key)
      setNewKeyLabel('')
      loadKeys()
    } catch (e: any) {
      flash(e.message || 'Failed to create key', false)
    }
    setCreating(false)
  }

  const handleRevokeKey = async (id: string, label: string) => {
    if (!confirm(`Revoke key "${label || id}"?`)) return
    try {
      await api.revokeApiKey(id)
      flash(`Key "${label || id}" revoked`, true)
      loadKeys()
    } catch (e: any) {
      flash(e.message || 'Failed to revoke key', false)
    }
  }

  // Show toast for 3s
  const flash = (msg: string, ok: boolean) => {
    setToast({ msg, ok })
    setTimeout(() => setToast(null), 3000)
  }

  const handleSet = async (name: string) => {
    setSaving(true)
    try {
      const r = await api.setVariable(name, editValue)
      if (r.ok) {
        flash(`${name} = ${r.new_value}  (was ${r.old_value})`, true)
        setEditName(null)
        load()
      }
    } catch (e: any) {
      flash(e.message || 'Failed to set variable', false)
    }
    setSaving(false)
  }

  const handleReload = async () => {
    setReloading(true)
    try {
      const r = await api.reloadConfig()
      flash(`Config reloaded — ${r.changed} variable(s) changed`, r.ok)
      load()
    } catch (e: any) {
      flash(e.message || 'Reload failed', false)
    }
    setReloading(false)
  }

  const filtered = vars.filter(v => {
    if (catFilter && v.category !== catFilter) return false
    if (filter) {
      const q = filter.toLowerCase()
      return v.name.toLowerCase().includes(q) || v.description.toLowerCase().includes(q)
    }
    return true
  })

  const dynamicCount = vars.filter(v => v.mode === 'DYNAMIC').length
  const readonlyCount = vars.filter(v => v.mode === 'READONLY').length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-gray-400 text-sm mt-1">
            {tab === 'variables'
              ? `${vars.length} variables — ${dynamicCount} dynamic, ${readonlyCount} readonly`
              : `${keys.length} active API keys`}
          </p>
        </div>
        {tab === 'variables' && (
          <button onClick={handleReload} disabled={reloading}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors">
            {reloading ? 'Reloading…' : '↻ Reload Config'}
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800 pb-0">
        {[
          { key: 'variables' as const, label: 'Variables' },
          { key: 'apikeys' as const, label: 'API Keys' },
        ].map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === t.key
                ? 'border-indigo-500 text-indigo-300'
                : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Toast */}
      {toast && (
        <div className={`px-4 py-3 rounded-lg text-sm ${toast.ok ? 'bg-green-900/50 text-green-300 border border-green-800' : 'bg-red-900/50 text-red-300 border border-red-800'}`}>
          {toast.msg}
        </div>
      )}

      {/* API Keys Tab */}
      {tab === 'apikeys' && (
        <div className="space-y-5">
          {/* Create Key */}
          <div className="flex gap-3">
            <input value={newKeyLabel} onChange={e => setNewKeyLabel(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleCreateKey() }}
              placeholder="Key label (e.g. production, dev-local)"
              className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-indigo-500 focus:outline-none" />
            <button onClick={handleCreateKey} disabled={creating || !newKeyLabel.trim()}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors">
              {creating ? 'Creating…' : 'Create Key'}
            </button>
          </div>

          {/* Newly created key display */}
          {createdKey && (
            <div className="bg-green-900/30 border border-green-800 rounded-xl p-4 space-y-2">
              <p className="text-green-300 text-sm font-medium">Key created — copy it now, it won't be shown again:</p>
              <code className="block bg-gray-900 rounded-lg px-4 py-3 text-sm text-white font-mono break-all select-all">
                {createdKey}
              </code>
              <button onClick={() => { navigator.clipboard.writeText(createdKey); flash('Copied!', true) }}
                className="text-xs text-indigo-400 hover:text-indigo-300">
                Copy to clipboard
              </button>
            </div>
          )}

          {/* Keys table */}
          {keysLoading ? (
            <p className="text-gray-500 text-center py-10">Loading…</p>
          ) : keys.length === 0 ? (
            <p className="text-gray-500 text-center py-10">No API keys created yet</p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-gray-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-900 text-gray-400 text-left">
                    <th className="px-4 py-3 font-medium">Label</th>
                    <th className="px-4 py-3 font-medium">Key Prefix</th>
                    <th className="px-4 py-3 font-medium">Created</th>
                    <th className="px-4 py-3 font-medium">Last Used</th>
                    <th className="px-4 py-3 font-medium w-20"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {keys.map(k => (
                    <tr key={k.id} className="hover:bg-gray-900/50 group">
                      <td className="px-4 py-3 text-white">{k.label || <span className="text-gray-500 italic">unlabeled</span>}</td>
                      <td className="px-4 py-3 font-mono text-xs text-gray-300">{k.key_prefix}…</td>
                      <td className="px-4 py-3 text-xs text-gray-400">
                        {k.created_at ? new Date(k.created_at * 1000).toLocaleString() : '—'}
                      </td>
                      <td className="px-4 py-3 text-xs text-gray-400">
                        {k.last_used_at ? new Date(k.last_used_at * 1000).toLocaleString() : 'never'}
                      </td>
                      <td className="px-4 py-3">
                        <button onClick={() => handleRevokeKey(k.id, k.label)}
                          className="opacity-0 group-hover:opacity-100 text-xs text-red-400 hover:text-red-300 transition-opacity">
                          Revoke
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Variables Tab */}
      {tab === 'variables' && <>
      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <input value={filter} onChange={e => setFilter(e.target.value)}
          placeholder="SHOW VARIABLES LIKE …" spellCheck={false}
          className="flex-1 min-w-[200px] px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:border-indigo-500 focus:outline-none" />
        <div className="flex gap-1 flex-wrap">
          {CATEGORIES.map(c => (
            <button key={c.key} onClick={() => setCatFilter(c.key)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                catFilter === c.key
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
              }`}>
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <p className="text-gray-500 text-center py-10">Loading…</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900 text-gray-400 text-left">
                <th className="px-4 py-3 font-medium">Variable</th>
                <th className="px-4 py-3 font-medium">Value</th>
                <th className="px-4 py-3 font-medium">Default</th>
                <th className="px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3 font-medium">Mode</th>
                <th className="px-4 py-3 font-medium">Category</th>
                <th className="px-4 py-3 font-medium w-16"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.map(v => (
                <tr key={v.name} className="hover:bg-gray-900/50 group">
                  <td className="px-4 py-3">
                    <div className="text-white font-mono text-xs">{v.name}</div>
                    <div className="text-gray-500 text-xs mt-0.5 max-w-xs truncate">{v.description}</div>
                  </td>
                  <td className="px-4 py-3">
                    {editName === v.name ? (
                      <div className="flex gap-2 items-center">
                        <input value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          onKeyDown={e => { if (e.key === 'Enter') handleSet(v.name); if (e.key === 'Escape') setEditName(null) }}
                          autoFocus
                          className="px-2 py-1 bg-gray-800 border border-indigo-500 rounded text-white text-xs font-mono w-40 focus:outline-none" />
                        <button onClick={() => handleSet(v.name)} disabled={saving}
                          className="text-xs text-green-400 hover:text-green-300">✓</button>
                        <button onClick={() => setEditName(null)}
                          className="text-xs text-gray-500 hover:text-gray-300">✕</button>
                      </div>
                    ) : (
                      <span className={`font-mono text-xs ${v.value !== v.default_value ? 'text-amber-300' : 'text-gray-300'}`}>
                        {v.value}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-gray-500">{v.default_value}</td>
                  <td className="px-4 py-3">
                    <span className="text-xs text-gray-400">{v.type}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      v.mode === 'DYNAMIC'
                        ? 'bg-green-900/40 text-green-400 border border-green-800'
                        : 'bg-gray-800 text-gray-500 border border-gray-700'
                    }`}>
                      {v.mode}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-400">{v.category}</td>
                  <td className="px-4 py-3">
                    {v.mode === 'DYNAMIC' && editName !== v.name && (
                      <button onClick={() => { setEditName(v.name); setEditValue(v.value) }}
                        className="opacity-0 group-hover:opacity-100 text-xs text-indigo-400 hover:text-indigo-300 transition-opacity">
                        SET
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={7} className="px-4 py-10 text-center text-gray-500">No variables match</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      </>}
    </div>
  )
}
