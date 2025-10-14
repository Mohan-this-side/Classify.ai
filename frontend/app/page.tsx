'use client'

import React, { useState, useEffect } from 'react'
import { toast } from 'react-hot-toast'
import { 
  Upload, Brain, BarChart3, Download, FileText, Code, Database, 
  Zap, Bot, TrendingUp, Cpu, Eye, Settings, Lightbulb,
  ChevronRight, Sparkles, CircuitBoard, Network
} from 'lucide-react'
import FileUpload from '@/components/FileUpload'
import AgentStatus from '@/components/AgentStatus'
import ProgressTracker from '@/components/ProgressTracker'
import ResultsViewer from '@/components/ResultsViewer'
import RealtimeInsights from '@/components/RealtimeInsights'
import PlotViewer from '@/components/PlotViewer'
import ApiKeySettings from '@/components/ApiKeySettings'

interface WorkflowState {
  id: string
  status: string
  progress: number
  currentPhase: string
  agents: Array<{
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
    progress?: number
    message?: string
    executionTime?: number
  }>
  results?: any
}

interface Insight {
  id: string
  agent: string
  message: string
  timestamp: Date
  type: 'info' | 'success' | 'warning' | 'error'
}

interface Plot {
  id: string
  name: string
  path: string
  description?: string
  agent: string
  timestamp: Date
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [description, setDescription] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [insights, setInsights] = useState<Insight[]>([])
  const [currentAgent, setCurrentAgent] = useState<string | null>(null)
  const [plots, setPlots] = useState<Plot[]>([])

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      if (!workflow?.id) return
      
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//localhost:8000/ws/${workflow.id}`
      const websocket = new WebSocket(wsUrl)
      
      websocket.onopen = () => {
        console.log('WebSocket connected')
        setWs(websocket)
      }
      
      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data)
        console.log('WebSocket message:', data)
        
        if (data.type === 'workflow_update') {
          setWorkflow(prev => ({
            ...prev,
            ...data.data
          }))
        } else if (data.type === 'agent_started') {
          setCurrentAgent(data.agent)
          setInsights(prev => [...prev, {
            id: `${data.agent}_${Date.now()}`,
            agent: data.agent,
            message: data.message || `Starting ${data.agent} agent...`,
            timestamp: new Date(),
            type: 'info'
          }])
          setWorkflow(prev => ({
            ...prev,
            currentPhase: `Starting ${data.agent}...`,
            agents: prev?.agents.map(agent => 
              agent.name === data.agent 
                ? { ...agent, status: 'running', message: data.message }
                : agent
            ) || []
          }))
          toast.success(`Starting ${data.agent} agent...`)
        } else if (data.type === 'agent_completed') {
          setCurrentAgent(null)
          setInsights(prev => [...prev, {
            id: `${data.agent}_completed_${Date.now()}`,
            agent: data.agent,
            message: data.message || `Completed ${data.agent} agent successfully`,
            timestamp: new Date(),
            type: 'success'
          }])
          
          // Update plots if EDA agent completed
          if (data.agent === 'eda_analysis' && data.plots) {
            const newPlots = data.plots.map((plotPath: string, index: number) => ({
              id: `eda_${index}_${Date.now()}`,
              name: `EDA Plot ${index + 1}`,
              path: plotPath,
              description: `Exploratory data analysis visualization`,
              agent: 'eda_analysis',
              timestamp: new Date()
            }))
            setPlots(prev => [...prev, ...newPlots])
          }
          
          setWorkflow(prev => ({
            ...prev,
            currentPhase: `Completed ${data.agent}`,
            progress: data.progress || prev?.progress || 0,
            agents: prev?.agents.map(agent => 
              agent.name === data.agent 
                ? { ...agent, status: 'completed', message: data.message }
                : agent
            ) || []
          }))
          toast.success(`Completed ${data.agent} agent!`)
        } else if (data.type === 'agent_failed') {
          setCurrentAgent(null)
          setInsights(prev => [...prev, {
            id: `agent_failed_${Date.now()}`,
            agent: data.agent,
            message: data.message || `Agent ${data.agent} failed`,
            timestamp: new Date(),
            type: 'error'
          }])
          setWorkflow(prev => ({
            ...prev,
            agents: prev?.agents.map(agent => 
              agent.name === data.agent 
                ? { ...agent, status: 'failed', message: data.message }
                : agent
            ) || []
          }))
          toast.error(`Agent ${data.agent} failed: ${data.error}`)
        } else if (data.type === 'workflow_completed') {
          setCurrentAgent(null)
          setInsights(prev => [...prev, {
            id: `workflow_completed_${Date.now()}`,
            agent: 'project_manager',
            message: 'Workflow completed successfully! All agents have finished processing.',
            timestamp: new Date(),
            type: 'success'
          }])
          setWorkflow(prev => ({
            ...prev,
            status: 'completed',
            progress: 100,
            currentPhase: 'Workflow completed!'
          }))
          toast.success('Workflow completed successfully!')
          setIsRunning(false)
        }
      }
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected')
        setWs(null)
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000)
      }
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    }

    if (isRunning && workflow?.id) {
      connectWebSocket()
    }

    return () => {
      if (ws) {
        ws.close()
      }
    }
  }, [isRunning, workflow?.id])

  const startWorkflow = async () => {
    if (!file || !targetColumn || !description) {
      toast.error('Please fill in all required fields')
      return
    }

    if (!apiKey) {
      toast.error('Please enter your Gemini API key')
      return
    }

    setIsRunning(true)
    setWorkflow(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('target_column', targetColumn)
      formData.append('description', description)
      formData.append('api_key', apiKey)
      formData.append('user_id', 'web_user')

      const response = await fetch('http://localhost:8000/api/workflow/start', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to start workflow')
      }

      const result = await response.json()
      console.log('Workflow started:', result)

      // Initialize workflow state
      setWorkflow({
        id: result.workflow_id,
        status: 'running',
        progress: 0,
        currentPhase: 'Initializing...',
        agents: [
          { name: 'data_cleaning', status: 'pending' },
          { name: 'data_discovery', status: 'pending' },
          { name: 'eda_analysis', status: 'pending' },
          { name: 'feature_engineering', status: 'pending' },
          { name: 'ml_building', status: 'pending' },
          { name: 'model_evaluation', status: 'pending' },
          { name: 'technical_reporter', status: 'pending' },
          { name: 'project_manager', status: 'pending' }
        ]
      })

      toast.success('Workflow started successfully!')
    } catch (error) {
      console.error('Error starting workflow:', error)
      toast.error('Failed to start workflow')
      setIsRunning(false)
    }
  }

  const checkWorkflowStatus = async () => {
    if (!workflow) return

    try {
      const response = await fetch(`http://localhost:8000/api/workflow/status/${workflow.id}`)
      const status = await response.json()
      
      setWorkflow(prev => ({
        ...prev,
        status: status.status,
        progress: status.progress || 0,
        currentPhase: status.current_phase || 'Processing...',
        agentStatus: status.agent_status || {},
        completedAgents: status.completed_agents || [],
        errors: status.errors || []
      }))

      if (status.status === 'completed' || status.status === 'failed') {
        setIsRunning(false)
        if (status.status === 'completed') {
          // Fetch results
          const resultsResponse = await fetch(`http://localhost:8000/api/workflow/results/${workflow.id}`)
          const results = await resultsResponse.json()
          
          setWorkflow(prev => ({
            ...prev,
            results: results
          }))
          
          toast.success('Workflow completed successfully!')
        } else {
          toast.error('Workflow failed')
        }
      }
    } catch (error) {
      console.error('Error checking workflow status:', error)
    }
  }

  // Poll for status updates
  useEffect(() => {
    if (isRunning && workflow) {
      const interval = setInterval(checkWorkflowStatus, 2000)
      return () => clearInterval(interval)
    }
  }, [isRunning, workflow])

  const handleDownload = async (type: string) => {
    if (!workflow?.id) return
    
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/download/${workflow.id}/${type}`)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        
        // Set appropriate file extension based on file type
        let extension = 'txt'
        if (type === 'notebook') extension = 'ipynb'
        else if (type === 'model') extension = 'joblib'
        else if (type === 'plots') extension = 'zip'
        
        a.download = `${type}_${workflow.id}.${extension}`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
        toast.success(`${type} downloaded successfully!`)
      } else {
        const errorText = await response.text()
        console.error(`Download failed: ${errorText}`)
        toast.error(`Failed to download ${type}: ${errorText}`)
      }
    } catch (error) {
      console.error(`Error downloading ${type}:`, error)
      toast.error(`Error downloading ${type}`)
    }
  }

  return (
    <div className="min-h-screen relative">
      {/* Animated Background */}
      <div className="fixed inset-0 bg-cyber-grid bg-cyber-grid -z-10 opacity-30" />
      
      {/* Hero Header */}
      <header className="relative">
        <div className="absolute inset-0 bg-neon-glow opacity-20 blur-3xl" />
        <div className="relative bg-dark-950/90 backdrop-blur-xl border-b border-neon-500/20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-4">
                <div className="relative">
                  <CircuitBoard className="w-16 h-16 text-neon-500 animate-pulse" />
                  <div className="absolute inset-0 bg-neon-500 blur-xl opacity-50 animate-ping" />
                </div>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4">
                <span className="gradient-text">DS Capstone</span>
                <br />
                <span className="neon-text">Multi-Agent System</span>
              </h1>
              <p className="text-xl text-gray-300 mb-6 max-w-3xl mx-auto">
                Harness the power of AI agents working in harmony to transform your data into actionable insights. 
                Our intelligent system handles the entire machine learning pipeline from data cleaning to model deployment.
              </p>
              <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-electric-400" />
                  <span>Powered by LangGraph</span>
                </div>
                <div className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-neon-400" />
                  <span>FastAPI Backend</span>
                </div>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyber-400" />
                  <span>Real-time Processing</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Agent Overview Section */}
      {!workflow && (
        <section className="py-16 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-white mb-4">Intelligent Agent Workflow</h2>
              <p className="text-gray-400 max-w-2xl mx-auto">
                Our system employs 8 specialized AI agents that work together to deliver comprehensive data science solutions
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              {[
                { icon: Bot, name: 'Data Cleaning', desc: 'Handles missing values, outliers, and data quality issues', color: 'text-blue-400' },
                { icon: Eye, name: 'Data Discovery', desc: 'Researches similar datasets and domain knowledge', color: 'text-green-400' },
                { icon: BarChart3, name: 'EDA Analysis', desc: 'Performs comprehensive exploratory data analysis', color: 'text-yellow-400' },
                { icon: Settings, name: 'Feature Engineering', desc: 'Creates and selects optimal features', color: 'text-purple-400' },
                { icon: Cpu, name: 'ML Model Builder', desc: 'Trains and optimizes machine learning models', color: 'text-red-400' },
                { icon: TrendingUp, name: 'Model Evaluation', desc: 'Evaluates model performance and metrics', color: 'text-indigo-400' },
                { icon: FileText, name: 'Technical Reporter', desc: 'Generates comprehensive analysis reports', color: 'text-pink-400' },
                { icon: Lightbulb, name: 'Project Manager', desc: 'Orchestrates the entire workflow', color: 'text-orange-400' }
              ].map((agent, index) => (
                <div key={agent.name} className="card group cursor-pointer">
                  <div className={`${agent.color} mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <agent.icon className="w-8 h-8" />
                  </div>
                  <h3 className="text-white font-semibold mb-2">{agent.name}</h3>
                  <p className="text-gray-400 text-sm">{agent.desc}</p>
                  <div className="mt-4 h-1 bg-gray-700 rounded-full overflow-hidden">
                    <div className={`h-full ${agent.color.replace('text', 'bg')} rounded-full transition-all duration-1000`} 
                         style={{ width: workflow ? '100%' : '0%' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Main Workflow Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className={`grid gap-8 ${workflow ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2'}`}>
          {/* Upload Section */}
          {!workflow && (
            <div className="space-y-6">
              <div className="card">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-neon-500/20 rounded-lg">
                    <Upload className="w-6 h-6 text-neon-400" />
                  </div>
                  <h2 className="text-2xl font-semibold text-white">
                    Upload Your Dataset
                  </h2>
                </div>
                
                <FileUpload
                  onFileSelect={setFile}
                  onTargetColumnChange={setTargetColumn}
                  onDescriptionChange={setDescription}
                  targetColumn={targetColumn}
                  description={description}
                  disabled={isRunning}
                />

                {/* API Key Settings */}
                <div className="mt-6">
                  <ApiKeySettings
                    onApiKeyChange={setApiKey}
                    initialApiKey={apiKey}
                    disabled={isRunning}
                  />
                </div>

                <div className="mt-8">
                  <button
                    onClick={startWorkflow}
                    disabled={isRunning || !file || !targetColumn || !description || !apiKey}
                    className="btn-primary w-full text-lg py-4"
                  >
                    {isRunning ? (
                      <>
                        <div className="loading-spinner mr-3" />
                        <span>Initializing Agents...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-6 h-6 mr-3" />
                        <span>Start AI Analysis</span>
                        <ChevronRight className="w-5 h-5 ml-3" />
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Quick Start Examples */}
              <div className="card">
                <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                  <Lightbulb className="w-5 h-5 text-yellow-400" />
                  Quick Start Examples
                </h3>
                <div className="space-y-4">
                  {[
                    {
                      icon: Database,
                      title: 'Iris Classification',
                      desc: 'Classic flower species prediction',
                      action: () => {
                        setTargetColumn('species')
                        setDescription('Classify iris flower species using sepal and petal measurements for botanical research')
                      }
                    },
                    {
                      icon: TrendingUp,
                      title: 'Customer Analytics',
                      desc: 'Behavior-based segmentation',
                      action: () => {
                        setTargetColumn('segment')
                        setDescription('Segment customers based on purchasing behavior and demographics for targeted marketing')
                      }
                    },
                    {
                      icon: BarChart3,
                      title: 'Financial Prediction',
                      desc: 'Risk assessment modeling',
                      action: () => {
                        setTargetColumn('risk_level')
                        setDescription('Predict financial risk levels using transaction history and customer profiles')
                      }
                    }
                  ].map((example, index) => (
                    <button
                      key={index}
                      onClick={example.action}
                      className="btn-secondary w-full p-4 text-left group"
                      disabled={isRunning}
                    >
                      <div className="flex items-center gap-4">
                        <div className="p-2 bg-neon-500/10 rounded-lg group-hover:bg-neon-500/20 transition-colors">
                          <example.icon className="w-5 h-5 text-neon-400" />
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-white mb-1">{example.title}</div>
                          <div className="text-sm text-gray-400">{example.desc}</div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-neon-400 transition-colors" />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          <div className="space-y-6">
            {workflow ? (
              <>
                <ProgressTracker
                  agents={workflow.agents}
                  overallProgress={workflow.progress}
                  currentPhase={workflow.currentPhase}
                />
                
                <RealtimeInsights
                  insights={insights}
                  currentAgent={currentAgent}
                  isRunning={isRunning}
                />
                
                <PlotViewer
                  plots={plots}
                  isGenerating={isRunning && currentAgent === 'eda_analysis'}
                />
                
                {workflow.results && (
                  <ResultsViewer
                    results={workflow.results}
                    onDownload={handleDownload}
                  />
                )}
              </>
            ) : (
              <div className="card text-center py-16">
                <div className="relative mb-8">
                  <div className="w-32 h-32 mx-auto bg-neon-500/10 rounded-full flex items-center justify-center relative overflow-hidden">
                    <Brain className="w-16 h-16 text-neon-400 animate-pulse" />
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neon-500/20 to-transparent animate-shimmer" />
                  </div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  AI Agents Ready
                </h3>
                <p className="text-gray-400 mb-8 max-w-md mx-auto">
                  Upload your dataset to begin the automated analysis. Our intelligent agents will handle everything from data cleaning to model deployment.
                </p>
                <div className="flex justify-center gap-4 text-sm text-gray-500">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyber-400 rounded-full animate-pulse" />
                    <span>Real-time Processing</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-electric-400 rounded-full animate-pulse" />
                    <span>Expert-level Analysis</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}