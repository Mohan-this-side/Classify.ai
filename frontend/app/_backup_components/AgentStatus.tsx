'use client'

import React from 'react'
import { CheckCircle, Clock, AlertCircle, XCircle, Loader2 } from 'lucide-react'

interface AgentStatusProps {
  agentName: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  progress?: number
  message?: string
  executionTime?: number
}

const statusConfig = {
  pending: {
    icon: Clock,
    color: 'text-night-400',
    bgColor: 'bg-night-800',
    label: 'Pending'
  },
  running: {
    icon: Loader2,
    color: 'text-neon-400',
    bgColor: 'bg-neon-900/20',
    label: 'Running'
  },
  completed: {
    icon: CheckCircle,
    color: 'text-green-400',
    bgColor: 'bg-green-900/20',
    label: 'Completed'
  },
  failed: {
    icon: XCircle,
    color: 'text-red-400',
    bgColor: 'bg-red-900/20',
    label: 'Failed'
  },
  skipped: {
    icon: AlertCircle,
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-900/20',
    label: 'Skipped'
  }
}

export default function AgentStatus({
  agentName,
  status,
  progress = 0,
  message,
  executionTime
}: AgentStatusProps) {
  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <div className={`agent-card ${config.bgColor}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${config.bgColor}`}>
            <Icon 
              className={`w-5 h-5 ${config.color} ${status === 'running' ? 'animate-spin' : ''}`} 
            />
          </div>
          <div>
            <h3 className="font-medium text-night-100 capitalize">
              {agentName.replace('_', ' ')}
            </h3>
            <p className={`text-sm ${config.color}`}>
              {config.label}
              {executionTime && ` (${executionTime.toFixed(1)}s)`}
            </p>
          </div>
        </div>
        
        {status === 'running' && (
          <div className="text-right">
            <div className="text-sm text-neon-400 font-medium">
              {Math.round(progress)}%
            </div>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {status === 'running' && (
        <div className="mt-3">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Message */}
      {message && (
        <div className="mt-2 text-sm text-night-300">
          {message}
        </div>
      )}
    </div>
  )
}
