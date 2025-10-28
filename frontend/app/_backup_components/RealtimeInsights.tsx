'use client'

import React, { useState, useEffect } from 'react'
import { Brain, Eye, BarChart3, Settings, Cpu, TrendingUp, FileText, Lightbulb, Sparkles, ChevronRight } from 'lucide-react'

interface Insight {
  id: string
  agent: string
  message: string
  timestamp: Date
  type: 'info' | 'success' | 'warning' | 'error'
}

interface RealtimeInsightsProps {
  insights: Insight[]
  currentAgent?: string
  isRunning: boolean
}

const agentIcons = {
  data_cleaning: Brain,
  data_discovery: Eye,
  eda_analysis: BarChart3,
  feature_engineering: Settings,
  ml_building: Cpu,
  model_evaluation: TrendingUp,
  technical_reporter: FileText,
  project_manager: Lightbulb
}

const agentNames = {
  data_cleaning: 'Data Cleaning',
  data_discovery: 'Data Discovery',
  eda_analysis: 'EDA Analysis',
  feature_engineering: 'Feature Engineering',
  ml_building: 'ML Model Builder',
  model_evaluation: 'Model Evaluation',
  technical_reporter: 'Technical Reporter',
  project_manager: 'Project Manager'
}

export default function RealtimeInsights({ insights, currentAgent, isRunning }: RealtimeInsightsProps) {
  const [displayedInsights, setDisplayedInsights] = useState<Insight[]>([])

  useEffect(() => {
    setDisplayedInsights(insights.slice(-10)) // Show last 10 insights
  }, [insights])

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'success': return '✅'
      case 'warning': return '⚠️'
      case 'error': return '❌'
      default: return 'ℹ️'
    }
  }

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'success': return 'text-green-400'
      case 'warning': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-blue-400'
    }
  }

  if (!isRunning && insights.length === 0) {
    return null
  }

  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-neon-500/20 rounded-lg">
          <Sparkles className="w-6 h-6 text-neon-400" />
        </div>
        <h2 className="text-2xl font-semibold text-white">
          Real-time Agent Insights
        </h2>
        {isRunning && (
          <div className="flex items-center gap-2 text-sm text-neon-400">
            <div className="w-2 h-2 bg-neon-400 rounded-full animate-pulse" />
            <span>Live</span>
          </div>
        )}
      </div>

      {/* Current Agent Status */}
      {currentAgent && isRunning && (
        <div className="mb-6 p-4 bg-neon-500/10 rounded-lg border border-neon-500/20">
          <div className="flex items-center gap-3">
            {React.createElement(agentIcons[currentAgent as keyof typeof agentIcons] || Brain, {
              className: "w-5 h-5 text-neon-400"
            })}
            <div>
              <h3 className="font-medium text-white">
                {agentNames[currentAgent as keyof typeof agentNames] || currentAgent}
              </h3>
              <p className="text-sm text-gray-400">Currently processing...</p>
            </div>
            <div className="ml-auto">
              <div className="loading-spinner" />
            </div>
          </div>
        </div>
      )}

      {/* Insights List */}
      <div className="space-y-3 max-h-96 overflow-y-auto scrollbar-hide">
        {displayedInsights.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Waiting for agent insights...</p>
          </div>
        ) : (
          displayedInsights.map((insight, index) => (
            <div
              key={insight.id || index}
              className={`p-3 rounded-lg border transition-all duration-300 ${
                insight.type === 'success' 
                  ? 'bg-green-500/10 border-green-500/20' 
                  : insight.type === 'warning'
                  ? 'bg-yellow-500/10 border-yellow-500/20'
                  : insight.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20'
                  : 'bg-blue-500/10 border-blue-500/20'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-1">
                  {React.createElement(agentIcons[insight.agent as keyof typeof agentIcons] || Brain, {
                    className: "w-4 h-4 text-neon-400"
                  })}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-white">
                      {agentNames[insight.agent as keyof typeof agentNames] || insight.agent}
                    </span>
                    <span className={`text-xs ${getInsightColor(insight.type)}`}>
                      {getInsightIcon(insight.type)}
                    </span>
                    <span className="text-xs text-gray-500">
                      {insight.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    {insight.message}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Agent Progress Overview */}
      {isRunning && (
        <div className="mt-6 pt-4 border-t border-gray-700">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Agent Progress</h3>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(agentNames).map(([key, name]) => {
              const Icon = agentIcons[key as keyof typeof agentIcons] || Brain
              const isCurrent = currentAgent === key
              const isCompleted = displayedInsights.some(insight => 
                insight.agent === key && insight.type === 'success'
              )
              
              return (
                <div
                  key={key}
                  className={`flex items-center gap-2 p-2 rounded-lg text-xs transition-all ${
                    isCurrent 
                      ? 'bg-neon-500/20 text-neon-400' 
                      : isCompleted
                      ? 'bg-green-500/10 text-green-400'
                      : 'bg-gray-700/50 text-gray-400'
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  <span className="truncate">{name}</span>
                  {isCurrent && <div className="w-2 h-2 bg-neon-400 rounded-full animate-pulse ml-auto" />}
                  {isCompleted && <span className="text-green-400 ml-auto">✓</span>}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
