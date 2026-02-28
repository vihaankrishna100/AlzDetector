import React from 'react'
import { Brain, Activity } from 'lucide-react'

export default function Header() {
  return (
    <header className="border-b border-white/10 bg-black/30 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold gradient-text">ADGENT</h1>
              <p className="text-sm text-slate-400">
                Multi-Agent Alzheimer's Disease Prediction System
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg border border-white/10">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span className="text-xs font-medium text-slate-300">System Ready</span>
          </div>
        </div>
      </div>
    </header>
  )
}
