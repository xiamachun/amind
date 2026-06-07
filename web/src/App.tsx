import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Memories from './pages/Memories'
import MemoryDetail from './pages/MemoryDetail'
import Events from './pages/Events'
import Graph from './pages/Graph'
import Sessions from './pages/Sessions'
import RecallDebug from './pages/RecallDebug'
import System from './pages/System'
import Settings from './pages/Settings'

// Legacy routes (gate-log, forget-log, recall-stale-log, pipeline) now
// redirect to the unified Events page with a preset filter. The old page
// files are kept until Phase 4 cleanup but are no longer imported/rendered.

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/memories" element={<Memories />} />
        <Route path="/memories/:id" element={<MemoryDetail />} />
        <Route path="/events" element={<Events />} />
        <Route path="/gate-log"          element={<Navigate to="/events?kind=Gate" replace />} />
        <Route path="/forget-log"        element={<Navigate to="/events?kind=GcDecay" replace />} />
        <Route path="/recall-stale-log"  element={<Navigate to="/events?kind=RecallStale" replace />} />
        <Route path="/pipeline"          element={<Navigate to="/events?kind=Reconcile" replace />} />
        <Route path="/graph" element={<Graph />} />
        <Route path="/sessions" element={<Sessions />} />
        <Route path="/recall-debug" element={<RecallDebug />} />
        <Route path="/system" element={<System />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
