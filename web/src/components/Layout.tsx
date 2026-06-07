import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api, type HealthResponse } from '../api/client'

// Nav layout: top-level pages + 4 quick-filter shortcuts into the unified
// Events page (per docs/arch/统一可观测层重构-MemoryEvent.md Plan C).
const nav = [
  { to: '/',          label: 'Dashboard',  icon: '\u{1F4CA}' },
  { to: '/memories',  label: 'Memories',   icon: '\u{1F9E0}' },
  { to: '/events',                     label: 'Events',      icon: '\u{1F4E1}' },
  { to: '/events?kind=Gate',           label: 'Gate',        icon: '\u{1F6AB}', sub: true },
  { to: '/events?kind=GcDecay',        label: 'Forget',      icon: '\u{1F32B}️', sub: true },
  { to: '/events?kind=RecallStale',    label: 'Stale Filter',icon: '\u{1F9F9}', sub: true },
  { to: '/events?kind=Reconcile',      label: 'Reconcile',   icon: '\u{1F501}', sub: true },
  { to: '/graph',        label: 'Graph',        icon: '\u{1F578}️' },
  { to: '/sessions',     label: 'Sessions',     icon: '\u{1F4AC}' },
  { to: '/recall-debug', label: 'Recall Debug', icon: '\u{1F50D}' },
  { to: '/system',       label: 'System',       icon: '⚙️' },
  { to: '/settings',     label: 'Settings',     icon: '🔧' },
]

export default function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const location = useLocation()

  useEffect(() => {
    const check = () => api.health().then(setHealth).catch(() => setHealth(null))
    check()
    const t = setInterval(check, 10000)
    return () => clearInterval(t)
  }, [])

  const online = health?.status === 'ok'

  // Highlight the quick-filter sub-link when the current URL matches both
  // the base path and the kind query param.
  const isActiveFor = (to: string) => {
    const [pathOnly, query] = to.split('?')
    if (location.pathname !== pathOnly) return false
    if (!query) return location.search === ''
    const target = new URLSearchParams(query)
    const current = new URLSearchParams(location.search)
    for (const [k, v] of target.entries()) {
      if (current.get(k) !== v) return false
    }
    return true
  }

  return (
    <div className="flex h-screen">
      <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="px-5 py-6">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="Amind" className="w-8 h-8 rounded" />
            <div>
              <h1 className="text-xl font-bold tracking-wide text-indigo-400">Amind</h1>
              <p className="text-xs text-gray-500 mt-0.5">Memory Engine Dashboard</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {nav.map(n => {
            const active = isActiveFor(n.to)
            return (
              <NavLink key={n.to} to={n.to} end={n.to === '/'}
                className={`flex items-center gap-3 ${n.sub ? 'pl-8 pr-3 py-1.5 text-xs' : 'px-3 py-2.5 text-sm'} rounded-lg transition-colors ${
                  active
                    ? 'bg-indigo-600/20 text-indigo-300 font-medium'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}>
                <span className={n.sub ? 'text-xs' : 'text-base'}>{n.icon}</span>
                {n.label}
              </NavLink>
            )
          })}
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
