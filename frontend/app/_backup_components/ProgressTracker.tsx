'use client'

import React from 'react'
import { BarChart3, Clock, CheckCircle } from 'lucide-react'
import AgentStatus from './AgentStatus'

interface Agent {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  progress?: number
  message?: string
  executionTime?: number
}

interface ProgressTrackerProps {
  agents: Agent[]
  overallProgress: number
  currentPhase: string
  estimatedTimeRemaining?: number
}

export default function ProgressTracker({
  agents,
  overallProgress,
  currentPhase,
  estimatedTimeRemaining
}: ProgressTrackerProps) {
  const completedAgents = agents.filter(agent => agent.status === 'completed').length
  const totalAgents = agents.length

  return (
    <div className="space-y-6">
      {/* Overall Progress */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-night-100 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-neon-400" />
            Workflow Progress
          </h2>
          <div className="text-right">
            <div className="text-2xl font-bold text-neon-400">
              {Math.round(overallProgress)}%
            </div>
            <div className="text-sm text-night-300">
              {completedAgents}/{totalAgents} agents completed
            </div>
          </div>
        </div>

        {/* Overall Progress Bar */}
        <div className="progress-bar mb-4">
          <div 
            className="progress-fill"
            style={{ width: `${overallProgress}%` }}
          />
        </div>

        {/* Current Phase */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-neon-400" />
            <span className="text-night-200 font-medium">
              Current Phase: {currentPhase}
            </span>
          </div>
          
          {estimatedTimeRemaining && (
            <div className="text-sm text-night-300">
              ~{Math.round(estimatedTimeRemaining)}s remaining
            </div>
          )}
        </div>
      </div>

      {/* Agent Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {agents.map((agent) => (
          <AgentStatus
            key={agent.name}
            agentName={agent.name}
            status={agent.status}
            progress={agent.progress}
            message={agent.message}
            executionTime={agent.executionTime}
          />
        ))}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="card text-center">
          <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-400" />
          <div className="text-2xl font-bold text-green-400">{completedAgents}</div>
          <div className="text-sm text-night-300">Completed</div>
        </div>
        
        <div className="card text-center">
          <Clock className="w-8 h-8 mx-auto mb-2 text-neon-400" />
          <div className="text-2xl font-bold text-neon-400">
            {agents.filter(a => a.status === 'running').length}
          </div>
          <div className="text-sm text-night-300">Running</div>
        </div>
        
        <div className="card text-center">
          <div className="w-8 h-8 mx-auto mb-2 text-night-400">
            {totalAgents - completedAgents - agents.filter(a => a.status === 'running').length}
          </div>
          <div className="text-2xl font-bold text-night-400">
            {totalAgents - completedAgents - agents.filter(a => a.status === 'running').length}
          </div>
          <div className="text-sm text-night-300">Pending</div>
        </div>
      </div>
    </div>
  )
}
