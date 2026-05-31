import { NavLink, Outlet } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api, type HealthResponse } from '../api/client'

const nav = [
  { to: '/',          label: 'Dashboard',  icon: '\u{1F4CA}' },
  { to: '/memories',  label: 'Memories',   icon: '\u{1F9E0}' },
  { to: '/gate-log',  label: 'Gate Log',   icon: '\u{1F6AB}' },
  { to: '/forget-log', label: 'Forget Log', icon: '\u{1F32B}️' },
  { to: '/recall-stale-log', label: 'Stale Filter', icon: '\u{1F9F9}' },
  { to: '/graph',     label: 'Graph',      icon: '\u{1F578}\uFE0F' },
  { to: '/sessions',  label: 'Sessions',   icon: '\u{1F4AC}' },
  { to: '/pipeline',     label: 'Pipeline',     icon: '\u{2693}' },
  { to: '/recall-debug', label: 'Recall Debug', icon: '\u{1F50D}' },
  { to: '/system',    label: 'System',     icon: '\u2699\uFE0F' },
  { to: '/settings',  label: 'Settings',   icon: '🔧' },
]

export default function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    const check = () => api.health().then(setHealth).catch(() => setHealth(null))
    check()
    const t = setInterval(check, 10000)
    return () => clearInterval(t)
  }, [])

  const online = health?.status === 'ok'

  return (
    <div className="flex h-screen">
      <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="px-5 py-6">
          <h1 className="text-xl font-bold tracking-wide text-indigo-400">amind</h1>
          <p className="text-xs text-gray-500 mt-1">Memory Engine Dashboard</p>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {nav.map(n => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-300 font-medium'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }>
              <span className="text-base">{n.icon}</span>
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-gray-800">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className={`w-2 h-2 rounded-full ${online ? 'bg-green-500' : 'bg-red-500'}`} />
            {online ? `v${health?.version}` : 'offline'}
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto p-6 bg-gray-950">
        <Outlet />
      </main>
    </div>
  )
}
