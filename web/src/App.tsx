import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Memories from './pages/Memories'
import MemoryDetail from './pages/MemoryDetail'
import GateLog from './pages/GateLog'
import ForgetLog from './pages/ForgetLog'
import RecallStaleLog from './pages/RecallStaleLog'
import Graph from './pages/Graph'
import Sessions from './pages/Sessions'
import Pipeline from './pages/Pipeline'
import RecallDebug from './pages/RecallDebug'
import System from './pages/System'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/memories" element={<Memories />} />
        <Route path="/memories/:id" element={<MemoryDetail />} />
        <Route path="/gate-log" element={<GateLog />} />
        <Route path="/forget-log" element={<ForgetLog />} />
        <Route path="/recall-stale-log" element={<RecallStaleLog />} />
        <Route path="/graph" element={<Graph />} />
        <Route path="/sessions" element={<Sessions />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/recall-debug" element={<RecallDebug />} />
        <Route path="/system" element={<System />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
