'use client'

import { useState, useEffect } from 'react'
import { 
  CheckCircle, Clock, AlertCircle, Zap, Bot, Eye, Settings, 
  Cpu, TrendingUp, FileText, Brain, Sparkles, Activity
} from 'lucide-react'

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
}

const agentConfig = {
  data_cleaning: { icon: Bot, name: 'Data Cleaning', desc: 'Cleaning and preprocessing dataset', color: 'blue' },
  data_discovery: { icon: Eye, name: 'Data Discovery', desc: 'Researching similar datasets and best practices', color: 'green' },
  eda_analysis: { icon: TrendingUp, name: 'EDA Analysis', desc: 'Performing exploratory data analysis', color: 'yellow' },
  feature_engineering: { icon: Settings, name: 'Feature Engineering', desc: 'Creating and selecting optimal features', color: 'purple' },
  ml_building: { icon: Cpu, name: 'ML Model Building', desc: 'Training and optimizing machine learning models', color: 'red' },
  model_evaluation: { icon: Activity, name: 'Model Evaluation', desc: 'Evaluating model performance and metrics', color: 'indigo' },
  technical_reporting: { icon: FileText, name: 'Technical Reporting', desc: 'Generating comprehensive analysis reports', color: 'pink' },
  project_manager: { icon: Brain, name: 'Project Manager', desc: 'Orchestrating the entire workflow', color: 'orange' }
}

export default function ProgressTracker({ agents, overallProgress, currentPhase }: ProgressTrackerProps) {
  const [startTime] = useState(Date.now())
  const [elapsedTime, setElapsedTime] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - startTime)
    }, 1000)
    return () => clearInterval(interval)
  }, [startTime])

  const formatTime = (milliseconds: number) => {
    const seconds = Math.floor(milliseconds / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`
    return `${seconds}s`
  }

  const getAgentIcon = (agentName: string, status: string) => {
    const config = agentConfig[agentName as keyof typeof agentConfig]
    const IconComponent = config?.icon || Bot
    
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-cyber-400" />
      case 'running':
        return <IconComponent className="w-6 h-6 text-neon-400 animate-pulse" />
      case 'failed':
        return <AlertCircle className="w-6 h-6 text-error-400" />
      default:
        return <IconComponent className="w-6 h-6 text-gray-500" />
    }
  }

  const getStatusClasses = (status: string) => {
    switch (status) {
      case 'completed':
        return 'border-cyber-500/50 bg-cyber-500/5'
      case 'running':
        return 'border-neon-500/50 bg-neon-500/10 shadow-neon'
      case 'failed':
        return 'border-error-500/50 bg-error-500/5'
      default:
        return 'border-gray-500/20 bg-gray-500/5'
    }
  }

  const completedCount = agents.filter(a => a.status === 'completed').length
  const runningCount = agents.filter(a => a.status === 'running').length
  const failedCount = agents.filter(a => a.status === 'failed').length

  return (
    <div className="space-y-8">
      {/* Header with Current Phase */}
      <div className="card">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-neon-500/20 rounded-xl">
            <Sparkles className="w-8 h-8 text-neon-400 animate-pulse" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-white">AI Agent Workflow</h2>
            <p className="text-gray-400 mt-1">{currentPhase}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-neon-400">{Math.round(overallProgress)}%</div>
            <div className="text-sm text-gray-400">Complete</div>
          </div>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="progress-bar mb-4">
          <div
            className="progress-fill transition-all duration-500"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
        
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Elapsed: {formatTime(elapsedTime)}</span>
          <span className="text-gray-400">{completedCount} of {agents.length} agents complete</span>
        </div>
      </div>

      {/* Agent Timeline */}
      <div className="card">
        <div className="flex items-center gap-2 mb-6">
          <Brain className="w-6 h-6 text-electric-400" />
          <h3 className="text-xl font-semibold text-white">Agent Progress</h3>
        </div>
        
        <div className="space-y-4">
          {agents.map((agent, index) => {
            const config = agentConfig[agent.name as keyof typeof agentConfig]
            return (
              <div
                key={agent.name}
                className={`relative p-4 rounded-2xl border transition-all duration-500 ${
                  getStatusClasses(agent.status)
                }`}
              >
                {/* Connection Line */}
                {index < agents.length - 1 && (
                  <div className="absolute left-8 top-14 w-px h-8 bg-gradient-to-b from-neon-500/30 to-transparent" />
                )}
                
                <div className="flex items-start gap-4">
                  <div className="relative">
                    {getAgentIcon(agent.name, agent.status)}
                    {agent.status === 'running' && (
                      <div className="absolute inset-0 bg-neon-500 rounded-full animate-ping opacity-20" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-white">
                        {config?.name || agent.name}
                      </h4>
                      <div className="flex items-center gap-2">
                        {agent.executionTime && (
                          <span className="text-xs text-gray-400">
                            {formatTime(agent.executionTime * 1000)}
                          </span>
                        )}
                        <div className={`px-2 py-1 rounded-lg text-xs font-medium ${
                          agent.status === 'completed' ? 'bg-cyber-500/20 text-cyber-400' :
                          agent.status === 'running' ? 'bg-neon-500/20 text-neon-400' :
                          agent.status === 'failed' ? 'bg-error-500/20 text-error-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {agent.status}
                        </div>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-400 mb-3">
                      {config?.desc || `${agent.name} processing...`}
                    </p>
                    
                    {agent.message && (
                      <div className="bg-dark-800/50 rounded-lg p-3 mb-3">
                        <p className="text-sm text-gray-300">{agent.message}</p>
                      </div>
                    )}
                    
                    {agent.status === 'running' && agent.progress !== undefined && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Progress</span>
                          <span className="text-neon-400">{Math.round(agent.progress)}%</span>
                        </div>
                        <div className="progress-bar h-2">
                          <div
                            className="progress-fill h-2 transition-all duration-300"
                            style={{ width: `${agent.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-cyber-400 mb-2">{completedCount}</div>
          <div className="text-sm text-gray-400">Completed</div>
          <div className="w-8 h-1 bg-cyber-400 rounded-full mx-auto mt-2" />
        </div>
        
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-neon-400 mb-2">{runningCount}</div>
          <div className="text-sm text-gray-400">Running</div>
          <div className="w-8 h-1 bg-neon-400 rounded-full mx-auto mt-2 animate-pulse" />
        </div>
        
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-gray-400 mb-2">{agents.length - completedCount - runningCount - failedCount}</div>
          <div className="text-sm text-gray-400">Pending</div>
          <div className="w-8 h-1 bg-gray-400 rounded-full mx-auto mt-2" />
        </div>
        
        <div className="card text-center py-6">
          <div className="text-3xl font-bold text-error-400 mb-2">{failedCount}</div>
          <div className="text-sm text-gray-400">Failed</div>
          <div className="w-8 h-1 bg-error-400 rounded-full mx-auto mt-2" />
        </div>
      </div>
    </div>
  )
}
