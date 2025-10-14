'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, Clock, AlertCircle, Brain, Database, BarChart3, Wrench, Cpu, Target, FileText } from 'lucide-react'

interface AgentStatusProps {
  sessionId: string
  onComplete: (data: any) => void
  onError: (error: string) => void
}

interface Agent {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  startTime?: string
  endTime?: string
  executionTime?: number
  message?: string
  icon: React.ComponentType<any>
}

export default function AgentStatus({ sessionId, onComplete, onError }: AgentStatusProps) {
  const [agents, setAgents] = useState<Agent[]>([
    {
      id: 'data_cleaning',
      name: 'Data Cleaning Agent',
      description: 'Cleaning and preprocessing the dataset',
      status: 'pending',
      progress: 0,
      icon: Database
    },
    {
      id: 'data_discovery',
      name: 'Data Discovery Agent',
      description: 'Researching similar datasets and approaches',
      status: 'pending',
      progress: 0,
      icon: Brain
    },
    {
      id: 'eda_analysis',
      name: 'EDA Analysis Agent',
      description: 'Performing exploratory data analysis',
      status: 'pending',
      progress: 0,
      icon: BarChart3
    },
    {
      id: 'feature_engineering',
      name: 'Feature Engineering Agent',
      description: 'Creating and selecting features for ML',
      status: 'pending',
      progress: 0,
      icon: Wrench
    },
    {
      id: 'ml_building',
      name: 'ML Model Builder Agent',
      description: 'Training and optimizing ML models',
      status: 'pending',
      progress: 0,
      icon: Cpu
    },
    {
      id: 'model_evaluation',
      name: 'Model Evaluation Agent',
      description: 'Evaluating model performance',
      status: 'pending',
      progress: 0,
      icon: Target
    },
    {
      id: 'technical_reporting',
      name: 'Technical Reporter Agent',
      description: 'Generating comprehensive reports',
      status: 'pending',
      progress: 0,
      icon: FileText
    }
  ])

  const [workflowStatus, setWorkflowStatus] = useState('initialized')
  const [currentAgent, setCurrentAgent] = useState<string | null>(null)

  useEffect(() => {
    // Simulate agent execution (in real app, this would come from WebSocket)
    const interval = setInterval(() => {
      setAgents(prevAgents => {
        const updatedAgents = [...prevAgents]
        let hasRunningAgent = false
        let completedAgents = 0

        updatedAgents.forEach((agent, index) => {
          if (agent.status === 'running') {
            hasRunningAgent = true
            agent.progress = Math.min(100, agent.progress + Math.random() * 15)
            
            if (agent.progress >= 100) {
              agent.status = 'completed'
              agent.progress = 100
              agent.endTime = new Date().toISOString()
              agent.executionTime = Math.random() * 30 + 10 // 10-40 seconds
              agent.message = 'Agent completed successfully'
            }
          } else if (agent.status === 'completed') {
            completedAgents++
          } else if (agent.status === 'pending' && !hasRunningAgent && index === completedAgents) {
            agent.status = 'running'
            agent.progress = 0
            agent.startTime = new Date().toISOString()
            agent.message = 'Agent started execution'
            setCurrentAgent(agent.id)
          }
        })

        // Check if all agents are completed
        if (completedAgents === updatedAgents.length) {
          setWorkflowStatus('completed')
          onComplete({
            sessionId,
            status: 'completed',
            agents: updatedAgents,
            completedAt: new Date().toISOString()
          })
        }

        return updatedAgents
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [sessionId, onComplete])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success-600" />
      case 'running':
        return <Clock className="w-5 h-5 text-primary-600 animate-pulse" />
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-error-600" />
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'agent-status-completed'
      case 'running':
        return 'agent-status-running'
      case 'failed':
        return 'agent-status-failed'
      default:
        return 'agent-status-pending'
    }
  }

  const getAgentIcon = (agent: Agent) => {
    const IconComponent = agent.icon
    return <IconComponent className="w-6 h-6" />
  }

  return (
    <div className="space-y-6">
      {/* Workflow Status Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Multi-Agent Workflow</h3>
            <p className="text-sm text-gray-600">
              {workflowStatus === 'completed' 
                ? 'All agents have completed successfully' 
                : 'AI agents are working on your classification task'
              }
            </p>
          </div>
          <div className={`agent-status ${getStatusColor(workflowStatus)}`}>
            {workflowStatus === 'completed' ? 'Completed' : 'Running'}
          </div>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className={`agent-card ${
              agent.status === 'running' ? 'ring-2 ring-primary-200' : ''
            }`}
          >
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className={`p-2 rounded-lg ${
                  agent.status === 'completed' ? 'bg-success-100' :
                  agent.status === 'running' ? 'bg-primary-100' :
                  agent.status === 'failed' ? 'bg-error-100' :
                  'bg-gray-100'
                }`}>
                  {getAgentIcon(agent)}
                </div>
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-gray-900 truncate">
                    {agent.name}
                  </h4>
                  {getStatusIcon(agent.status)}
                </div>
                
                <p className="text-xs text-gray-600 mb-3">
                  {agent.description}
                </p>
                
                {agent.status === 'running' && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>Progress</span>
                      <span>{Math.round(agent.progress)}%</span>
                    </div>
                    <div className="progress-bar h-1">
                      <div
                        className="progress-fill h-1"
                        style={{ width: `${agent.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {agent.status === 'completed' && agent.executionTime && (
                  <div className="text-xs text-gray-500">
                    Completed in {agent.executionTime.toFixed(1)}s
                  </div>
                )}
                
                {agent.message && (
                  <div className="text-xs text-gray-600 mt-2">
                    {agent.message}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Current Agent Details */}
      {currentAgent && (
        <div className="card bg-primary-50 border-primary-200">
          <div className="flex items-center space-x-3">
            <div className="loading-spinner" />
            <div>
              <h4 className="text-sm font-medium text-primary-900">
                Currently Running: {agents.find(a => a.id === currentAgent)?.name}
              </h4>
              <p className="text-xs text-primary-700">
                {agents.find(a => a.id === currentAgent)?.description}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <div className="text-2xl font-bold text-success-600">
            {agents.filter(a => a.status === 'completed').length}
          </div>
          <div className="text-sm text-gray-600">Completed</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-600">
            {agents.filter(a => a.status === 'running').length}
          </div>
          <div className="text-sm text-gray-600">Running</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-gray-600">
            {agents.filter(a => a.status === 'pending').length}
          </div>
          <div className="text-sm text-gray-600">Pending</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-error-600">
            {agents.filter(a => a.status === 'failed').length}
          </div>
          <div className="text-sm text-gray-600">Failed</div>
        </div>
      </div>
    </div>
  )
}
