'use client'

import React, { useState, useEffect } from 'react'
import { toast } from 'react-hot-toast'
import { 
  Upload, CheckCircle, Loader, Circle, Play, Download, MessageSquare, 
  Code, FileText, AlertCircle, BarChart3, Zap, Eye, Wrench, 
  FileSpreadsheet, TrendingUp, X
} from 'lucide-react'

type ViewType = 'upload' | 'workflow' | 'results'

export default function ClassifyAI() {
  const [activeView, setActiveView] = useState<ViewType>('upload')
  const [pmExpanded, setPmExpanded] = useState(true)
  const [pendingApproval, setPendingApproval] = useState(true)
  
  // Upload state
  const [file, setFile] = useState<File | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [columnOptions, setColumnOptions] = useState<string[]>([])
  
  // Workflow state
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [workflowStatus, setWorkflowStatus] = useState<string>('idle')
  const [agents, setAgents] = useState<any[]>([])
  const [pmMessages, setPmMessages] = useState<any[]>([])
  const [sandboxMetrics, setSandboxMetrics] = useState({ cpu: 0, memory: 0, time: 0 })
  const [results, setResults] = useState<any>(null)

  const parseCSVHeaders = async (file: File) => {
    return new Promise<string[]>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const firstLine = text.split('\n')[0]
        const headers = firstLine.split(',').map(h => h.trim().replace(/['"]/g, ''))
        resolve(headers)
      }
      reader.onerror = reject
      reader.readAsText(file.slice(0, 1024))
    })
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      toast.success(`File "${selectedFile.name}" uploaded successfully`)
      
      try {
        const headers = await parseCSVHeaders(selectedFile)
        setColumnOptions(headers)
        toast.success(`Found ${headers.length} columns`)
      } catch (error) {
        console.error('Error parsing CSV:', error)
        toast.error('Could not parse CSV headers')
      }
    }
  }

  const startWorkflow = async () => {
    if (!file || !targetColumn || !apiKey) {
      toast.error('Please fill in all fields')
      return
    }

    try {
      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', file)
      formData.append('target_column', targetColumn)
      formData.append('description', `Classification task for ${targetColumn}`)
      formData.append('api_key', apiKey)
      formData.append('user_id', 'web_user')

      // Send to backend
      const response = await fetch('http://localhost:8000/api/workflow/start', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to start workflow')
      }

      const data = await response.json()
      setWorkflowId(data.workflow_id)
      setWorkflowStatus('running')
      
      // Initialize agents status
      setAgents([
        { id: 'discovery', label: 'Discovery', status: 'pending', time: '' },
        { id: 'eda', label: 'EDA', status: 'pending', time: '' },
        { id: 'cleaning', label: 'Cleaning', status: 'pending', time: '' },
        { id: 'feature', label: 'Feature Eng.', status: 'pending', time: '' },
        { id: 'model', label: 'Model Build', status: 'pending', time: '' },
        { id: 'eval', label: 'Evaluation', status: 'pending', time: '' },
        { id: 'report', label: 'Reporting', status: 'pending', time: '' },
        { id: 'pm', label: 'PM', status: 'active', time: '' },
      ])

      // Switch to workflow view
      setActiveView('workflow')
      toast.success('Workflow started successfully!')

      // Start polling for status
      pollWorkflowStatus(data.workflow_id)
    } catch (error) {
      console.error('Error starting workflow:', error)
      toast.error('Failed to start workflow')
    }
  }

  const pollWorkflowStatus = async (wfId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/workflow/status/${wfId}`)
        const data = await response.json()

        setWorkflowStatus(data.status)

        // Map backend agent names to frontend agent IDs
        const agentMapping: { [key: string]: string } = {
          'data_discovery': 'discovery',
          'eda_analysis': 'eda',
          'data_cleaning': 'cleaning',
          'feature_engineering': 'feature',
          'ml_building': 'model',
          'model_evaluation': 'eval',
          'technical_reporter': 'report'
        }

        // Update agents based on backend status with Layer info
        if (data.agent_status) {
          setAgents(prev => prev.map(agent => {
            // Find matching backend agent name
            const backendAgentName = Object.keys(agentMapping).find(
              key => agentMapping[key] === agent.id
            )
            
            if (backendAgentName && data.agent_status[backendAgentName]) {
              const status = data.agent_status[backendAgentName]
              const layer = data.layer_usage?.[backendAgentName] || 'Layer 1'
              const layerEmoji = layer.includes('2') ? '🐳' : '⚡'
              return { 
                ...agent, 
                status: status === 'running' ? 'active' : status === 'completed' ? 'complete' : status,
                time: status === 'completed' ? `${layerEmoji} ${layer}` : (status === 'running' ? `${layerEmoji} Running...` : '')
              }
            }
            return agent
          }))
        }

        // ✅ Update PM messages from backend
        if (data.pm_messages) {
          setPmMessages(data.pm_messages)
        }
        
        // ✅ Update pending approval from backend
        if (data.pending_approval) {
          setPendingApproval(true)
        } else {
          setPendingApproval(false)
        }

        // ✅ Update sandbox metrics from backend
        if (data.sandbox_metrics) {
          const metrics = data.sandbox_metrics
          setSandboxMetrics({
            cpu: Math.round(metrics.cpu || 0),
            memory: Math.round(metrics.memory || 0),
            time: Math.max(0, Math.round(120 - (metrics.time || 0)))  // Time remaining out of 120s
          })
        }

        // Check if workflow is complete
        if (data.status === 'completed') {
          clearInterval(interval)
          // Reset sandbox metrics
          setSandboxMetrics({ cpu: 0, memory: 0, time: 0 })
          fetchResults(wfId)
        } else if (data.status === 'failed') {
          clearInterval(interval)
          toast.error('Workflow failed')
        }
      } catch (error) {
        console.error('Error polling status:', error)
      }
    }, 2000) // Poll every 2 seconds
  }

  // ✅ Handle approval gate responses
  const handleApprovalResponse = async (action: 'approve' | 'reject' | 'modify', comment?: string) => {
    if (!workflowId) return
    
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/${workflowId}/pm/approval`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action,
          comment: comment || `User ${action}d the workflow`
        })
      })
      
      if (response.ok) {
        toast.success(`Approval ${action}d successfully`)
        setPendingApproval(false)
      } else {
        toast.error(`Failed to ${action} workflow`)
      }
    } catch (error) {
      console.error('Error handling approval:', error)
      toast.error('Failed to process approval')
    }
  }

  // ✅ Handle PM questions
  const handlePMQuestion = async (question: string) => {
    if (!workflowId) return
    
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/${workflowId}/pm/question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question })
      })
      
      if (response.ok) {
        const data = await response.json()
        // Add the Q&A to PM messages
        setPmMessages(prev => [...prev, {
          type: 'question',
          content: question,
          timestamp: new Date().toISOString()
        }, {
          type: 'answer',
          content: data.answer,
          timestamp: new Date().toISOString()
        }])
        toast.success('Question answered!')
      } else {
        toast.error('Failed to get answer')
      }
    } catch (error) {
      console.error('Error asking PM question:', error)
      toast.error('Failed to ask question')
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center space-x-6">
          <h1 className="text-2xl font-bold text-blue-600">Classify AI</h1>
          <nav className="flex space-x-2">
            <NavButton active={activeView === 'upload'} onClick={() => setActiveView('upload')}>
              Upload
            </NavButton>
            <NavButton active={activeView === 'workflow'} onClick={() => setActiveView('workflow')}>
              Workflow
            </NavButton>
            <NavButton active={activeView === 'results'} onClick={() => setActiveView('results')}>
              Results
            </NavButton>
          </nav>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 text-sm ${apiKey ? 'text-green-600' : 'text-gray-400'}`}>
            <div className={`w-2 h-2 rounded-full ${apiKey ? 'bg-green-600 animate-pulse' : 'bg-gray-400'}`}></div>
            <span className="font-medium">{apiKey ? 'API Connected' : 'No API Key'}</span>
          </div>
          <button className="text-gray-600 hover:text-gray-900 font-medium">Help</button>
        </div>
      </header>

      {/* Main Content */}
      {activeView === 'upload' && (
        <UploadView
          file={file}
          handleFileChange={handleFileChange}
          targetColumn={targetColumn}
          setTargetColumn={setTargetColumn}
          columnOptions={columnOptions}
          apiKey={apiKey}
          setApiKey={setApiKey}
          onStart={startWorkflow}
        />
      )}
      
      {activeView === 'workflow' && (
        <WorkflowView
          agents={agents}
          pmExpanded={pmExpanded}
          setPmExpanded={setPmExpanded}
          pendingApproval={pendingApproval}
          setPendingApproval={setPendingApproval}
          pmMessages={pmMessages}
          sandboxMetrics={sandboxMetrics}
          onApprovalResponse={handleApprovalResponse}
          onPMQuestion={handlePMQuestion}
        />
      )}
      
      {activeView === 'results' && <ResultsView results={results} />}
    </div>
  )
}

// ========== COMPONENTS ==========

function NavButton({ active, onClick, children }: any) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium transition-all ${
        active
          ? 'bg-blue-100 text-blue-700 shadow-sm'
          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
      }`}
    >
      {children}
    </button>
  )
}

function UploadView({ file, handleFileChange, targetColumn, setTargetColumn, columnOptions, apiKey, setApiKey, onStart }: any) {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl w-full space-y-6">
        {/* Title */}
        <div className="text-center space-y-2">
          <h1 className="text-5xl font-bold text-blue-600">Classify AI</h1>
          <p className="text-xl text-gray-600">Automated ML Pipeline with Real-Time Education</p>
        </div>

        {/* File Upload */}
        <label className="block border-2 border-dashed border-gray-300 rounded-xl p-12 bg-white hover:border-blue-400 hover:bg-blue-50 transition-all cursor-pointer group">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="flex flex-col items-center space-y-4">
            <Upload className="w-16 h-16 text-gray-400 group-hover:text-blue-500 transition-colors" />
            <div className="text-center">
              <p className="text-lg font-medium text-gray-700 group-hover:text-gray-900">
                Drop your CSV or Excel file here
              </p>
              <p className="text-sm text-gray-500">or click to browse</p>
            </div>
            {file && (
              <div className="text-sm text-blue-600 font-medium">
                ✓ {file.name}
              </div>
            )}
            <p className="text-xs text-gray-400">Max 100MB • Up to 1M rows</p>
          </div>
        </label>

        {/* Target Column */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Target Column</label>
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-3 bg-white text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
            disabled={columnOptions.length === 0}
          >
            <option value="">Select target column...</option>
            {columnOptions.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>

        {/* API Key */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            API Key (Gemini/OpenAI/Anthropic)
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full border border-gray-300 rounded-lg px-4 py-3 bg-white text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
          />
        </div>

        {/* Start Button */}
        <button
          onClick={onStart}
          disabled={!file || !targetColumn || !apiKey}
          className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 rounded-lg font-semibold text-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
        >
          <Play className="w-6 h-6" />
          <span>Start Analysis</span>
        </button>
      </div>
    </div>
  )
}

function WorkflowView({ agents, pmExpanded, setPmExpanded, pendingApproval, setPendingApproval, pmMessages, sandboxMetrics, onApprovalResponse, onPMQuestion }: any) {
  // Map agent IDs to icons
  const iconMap: any = {
    discovery: TrendingUp,
    eda: Eye,
    cleaning: FileSpreadsheet,
    feature: Wrench,
    model: Zap,
    eval: BarChart3,
    report: FileText,
    pm: MessageSquare
  }

  // Add icons to agents
  const agentsWithIcons = agents.map((agent: any) => ({
    ...agent,
    icon: iconMap[agent.id] || Circle
  }))

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all ${pmExpanded ? 'mr-96' : 'mr-0'}`}>
        {/* Timeline */}
        <div className="bg-white border-b border-gray-200 px-6 py-6 overflow-x-auto shadow-sm">
          <div className="flex items-center justify-between min-w-max max-w-6xl mx-auto">
            {agentsWithIcons.map((agent: any, idx: number) => (
              <React.Fragment key={agent.id}>
                <AgentStep {...agent} />
                {idx < agentsWithIcons.length - 1 && (
                  <div className={`w-12 h-1 ${agent.status === 'completed' ? 'bg-blue-600' : 'bg-gray-300'}`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Agent Activity */}
        <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
          <div className="max-w-5xl mx-auto space-y-6">
            {/* Active Agent Card */}
            <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                    <Zap className="w-7 h-7 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">Model Builder Agent</h3>
                    <p className="text-sm text-gray-600">Training classification models with sklearn.Pipeline</p>
                  </div>
                </div>
                <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold flex items-center space-x-2">
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Running</span>
                </span>
              </div>

              {/* Layer 1 */}
              <div className="space-y-4">
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="font-semibold text-green-900">Layer 1: Analysis Complete</span>
                  </div>
                  <ul className="text-sm text-gray-700 space-y-1 ml-7">
                    <li>• Dataset shape: 150 rows, 8 features (4 original + 4 engineered)</li>
                    <li>• Target distribution: Balanced (50/50/50)</li>
                    <li>• No data leakage detected</li>
                    <li>• Recommended: RandomForest (handles non-linear relationships)</li>
                  </ul>
                </div>

                {/* Layer 2 */}
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <Loader className="w-5 h-5 text-blue-600 animate-spin" />
                    <span className="font-semibold text-blue-900">Layer 2: Generating Training Code</span>
                  </div>
                  <div className="ml-7">
                    <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden">
                      <div className="bg-gradient-to-r from-blue-600 to-blue-500 h-3 rounded-full transition-all" style={{width: '65%'}} />
                    </div>
                    <p className="text-xs text-gray-600 mt-2">Building sklearn.Pipeline with preprocessing...</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Sandbox Monitor */}
            <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
              <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                <Code className="w-5 h-5 text-gray-700" />
                <span>Sandbox Execution Monitor</span>
              </h4>
              <div className="grid grid-cols-3 gap-6">
                <MetricBar label="CPU Usage" value={sandboxMetrics.cpu} color="green" />
                <MetricBar label="Memory" value={sandboxMetrics.memory} color="blue" />
                <div>
                  <p className="text-xs text-gray-500 mb-2">Time Remaining</p>
                  <p className="text-lg font-bold text-gray-900">~{sandboxMetrics.time}s</p>
                </div>
              </div>
            </div>

            {/* Completed Agents */}
            <CompletedAgent icon={Wrench} name="Feature Engineering Agent" time="1m 45s" />
            <CompletedAgent icon={FileSpreadsheet} name="Data Cleaning Agent" time="2m 34s" />
          </div>
        </div>
      </div>

      {/* Project Manager Panel */}
      {pmExpanded && (
        <div className="fixed right-0 top-0 h-full w-96 bg-white border-l border-gray-200 shadow-2xl flex flex-col z-50">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-5 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageSquare className="w-6 h-6" />
              <h3 className="font-bold text-lg">Project Manager</h3>
            </div>
            <button onClick={() => setPmExpanded(false)} className="hover:bg-white/20 rounded-lg p-1.5 transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-gray-50">
            {/* ✅ Dynamic PM Messages from Backend */}
            {pmMessages.map((message: any, index: number) => (
              <PMMessage 
                key={index}
                agent={message.agent || 'System'} 
                time={new Date(message.timestamp).toLocaleTimeString()} 
                message={message.content}
                type={message.type}
              />
            ))}

            {/* ✅ Dynamic Approval Gate from Backend */}
            {pendingApproval && (
              <ApprovalGate onApprovalResponse={onApprovalResponse} />
            )}
          </div>

          <div className="p-4 border-t border-gray-200 bg-white">
            <QAInput onPMQuestion={onPMQuestion} />
          </div>
        </div>
      )}

      {/* Floating PM Button */}
      {!pmExpanded && (
        <button
          onClick={() => setPmExpanded(true)}
          className="fixed right-6 bottom-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-full p-5 shadow-2xl hover:shadow-3xl transition-all hover:scale-110 z-50"
        >
          <MessageSquare className="w-7 h-7" />
        </button>
      )}
    </div>
  )
}

function ResultsView({ results }: any) {
  if (!results) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading results...</p>
        </div>
      </div>
    )
  }
  return (
    <div className="flex-1 overflow-y-auto p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <h2 className="text-4xl font-bold text-gray-900">Analysis Complete! 🎉</h2>
          <button className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-all shadow-lg">
            <Download className="w-5 h-5" />
            <span>Download All</span>
          </button>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-6">
          <MetricCard 
            label="Accuracy" 
            value={results?.model_evaluation?.evaluation_metrics?.accuracy ? 
              `${(results.model_evaluation.evaluation_metrics.accuracy * 100).toFixed(1)}%` : 'N/A'} 
            gradient="from-blue-500 to-blue-600" 
          />
          <MetricCard 
            label="F1 Score" 
            value={results?.model_evaluation?.evaluation_metrics?.f1_weighted ? 
              results.model_evaluation.evaluation_metrics.f1_weighted.toFixed(3) : 'N/A'} 
            gradient="from-green-500 to-green-600" 
          />
          <MetricCard 
            label="Precision" 
            value={results?.model_evaluation?.evaluation_metrics?.precision_weighted ? 
              `${(results.model_evaluation.evaluation_metrics.precision_weighted * 100).toFixed(1)}%` : 'N/A'} 
            gradient="from-purple-500 to-purple-600" 
          />
          <MetricCard 
            label="Recall" 
            value={results?.model_evaluation?.evaluation_metrics?.recall_weighted ? 
              `${(results.model_evaluation.evaluation_metrics.recall_weighted * 100).toFixed(1)}%` : 'N/A'} 
            gradient="from-orange-500 to-orange-600" 
          />
        </div>

        {/* EDA */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
            <Eye className="w-7 h-7 text-purple-600" />
            <span>Exploratory Data Analysis</span>
          </h3>
          <div className="grid grid-cols-2 gap-6">
            {results?.eda_analysis?.plots && results.eda_analysis.plots.length > 0 ? (
              results.eda_analysis.plots.map((plot: any, idx: number) => (
                <div key={idx} className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200 p-4">
                  <p className="font-medium text-sm mb-2">{plot.title || plot.name || `Plot ${idx + 1}`}</p>
                  <img 
                    src={`http://localhost:8000${plot.path || plot.url}`} 
                    alt={plot.title || plot.name || `Plot ${idx + 1}`}
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5QbG90IG5vdCBhdmFpbGFibGU8L3RleHQ+PC9zdmc+'
                    }}
                  />
                </div>
              ))
            ) : (
              <div className="col-span-2 text-center py-8 text-gray-500">
                <p>No EDA plots generated. Check backend logs for EDA agent execution.</p>
              </div>
            )}
          </div>
        </div>

        {/* Feature Importance */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Feature Importance</h3>
          <div className="space-y-4">
            {results?.feature_importance ? (
              Object.entries(results.feature_importance)
                .sort(([, a]: any, [, b]: any) => b - a)
                .slice(0, 10)
                .map(([feature, importance]: any) => (
                  <FeatureBar 
                    key={feature} 
                    label={feature} 
                    value={Math.round(importance * 100)} 
                  />
                ))
            ) : (
              <p className="text-gray-500 text-center py-4">No feature importance data available</p>
            )}
          </div>
        </div>

        {/* Deliverables */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
            <FileText className="w-7 h-7" />
            <span>Your Deliverables</span>
          </h3>
          <div className="space-y-3">
            <DeliverableItem name="cleaned_dataset.csv" size="2.3 MB" />
            <DeliverableItem name="trained_model.joblib" size="156 KB" />
            <DeliverableItem name="analysis_notebook.ipynb" size="487 KB" />
          </div>
        </div>
      </div>
    </div>
  )
}

// ========== HELPER COMPONENTS ==========

function AgentStep({ icon: Icon, label, status, time }: any) {
  const colors = {
    completed: 'bg-green-100 text-green-600 border-green-200',
    active: 'bg-blue-100 text-blue-600 border-blue-200 animate-pulse',
    waiting: 'bg-gray-100 text-gray-400 border-gray-200'
  }

  return (
    <div className="flex flex-col items-center space-y-2 min-w-[100px]">
      <div className={`w-14 h-14 rounded-full flex items-center justify-center border-2 ${colors[status as keyof typeof colors]}`}>
        <Icon className="w-7 h-7" />
      </div>
      <div className="text-center">
        <p className="text-sm font-semibold text-gray-900">{label}</p>
        {time && <p className="text-xs text-gray-500">{time}</p>}
      </div>
    </div>
  )
}

function MetricBar({ label, value, color }: any) {
  const colors = {
    green: { bg: 'bg-green-500', light: 'bg-green-100' },
    blue: { bg: 'bg-blue-500', light: 'bg-blue-100' }
  }

  return (
    <div>
      <p className="text-xs text-gray-500 mb-2">{label}</p>
      <div className="flex items-center space-x-3">
        <div className={`flex-1 ${colors[color as keyof typeof colors].light} rounded-full h-3`}>
          <div className={`${colors[color as keyof typeof colors].bg} h-3 rounded-full transition-all`} style={{width: `${value}%`}} />
        </div>
        <span className="text-sm font-bold text-gray-900">{value}%</span>
      </div>
    </div>
  )
}

function CompletedAgent({ icon: Icon, name, time }: any) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 hover:shadow-md transition-all">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Icon className="w-6 h-6 text-green-600" />
          <div>
            <h4 className="font-semibold text-gray-900">{name}</h4>
            <p className="text-sm text-gray-500">Completed in {time}</p>
          </div>
        </div>
        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">View Details</button>
      </div>
    </div>
  )
}

function PMMessage({ agent, time, message, type }: any) {
  const getMessageStyle = () => {
    switch (type) {
      case 'question':
        return 'bg-blue-50 border-blue-200 text-blue-800'
      case 'answer':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'approval_gate':
        return 'bg-amber-50 border-amber-200 text-amber-800'
      default:
        return 'bg-white border-gray-200 text-gray-700'
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-purple-600">{agent}</span>
        <span className="text-xs text-gray-400">{time}</span>
      </div>
      <div className={`rounded-lg p-4 text-sm leading-relaxed shadow-sm border ${getMessageStyle()}`}>
        {message}
      </div>
    </div>
  )
}

// ✅ Approval Gate Component
function ApprovalGate({ onApprovalResponse }: any) {
  return (
    <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-5 space-y-4">
      <div className="flex items-start space-x-3">
        <AlertCircle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-bold text-amber-900 mb-3">⚠️ Approval Required</p>
          <p className="text-sm text-amber-800 mb-4">
            The workflow is paused and waiting for your approval to continue.
          </p>

          <div className="bg-white rounded-lg p-3 mb-4">
            <p className="text-xs text-gray-500 mb-1 font-semibold">Educational Note:</p>
            <p className="text-xs text-gray-700">
              This approval gate allows you to review the current progress and decide whether to continue with the next step.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => onApprovalResponse('approve')}
              className="bg-green-600 text-white py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors"
            >
              ✓ Approve
            </button>
            <button
              onClick={() => onApprovalResponse('modify')}
              className="bg-gray-200 text-gray-700 py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-gray-300 transition-colors"
            >
              Modify
            </button>
            <button
              onClick={() => onApprovalResponse('reject')}
              className="bg-red-100 text-red-700 py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-red-200 transition-colors"
            >
              Reject
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ✅ Q&A Input Component
function QAInput({ onPMQuestion }: any) {
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || isLoading) return

    setIsLoading(true)
    try {
      await onPMQuestion(question.trim())
      setQuestion('')
    } catch (error) {
      console.error('Error asking question:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex space-x-2">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask me anything about the process..."
        className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all"
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={!question.trim() || isLoading}
        className="bg-purple-600 text-white px-4 py-3 rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? <Loader className="w-4 h-4 animate-spin" /> : 'Ask'}
      </button>
    </form>
  )
}

function MetricCard({ label, value, gradient }: any) {
  return (
    <div className={`bg-gradient-to-br ${gradient} text-white rounded-xl p-8 shadow-lg`}>
      <p className="text-sm opacity-90 mb-2 font-medium">{label}</p>
      <p className="text-4xl font-bold">{value}</p>
    </div>
  )
}

function PlotCard({ title, type }: any) {
  return (
    <div className="h-72 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl border-2 border-purple-200 p-6">
      <p className="font-semibold text-gray-900 mb-3">{title}</p>
      <div className="h-52 flex items-center justify-center text-gray-400">
        [{type} visualization]
      </div>
    </div>
  )
}

function FeatureBar({ label, value }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="font-semibold text-gray-900">{label}</span>
        <span className="text-gray-500">{value}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full transition-all" style={{width: `${value}%`}} />
      </div>
    </div>
  )
}

function DeliverableItem({ name, size }: any) {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors border border-gray-200">
      <div className="flex items-center space-x-3">
        <FileText className="w-6 h-6 text-blue-600" />
        <div>
          <p className="font-semibold text-gray-900">{name}</p>
          <p className="text-sm text-gray-500">{size}</p>
        </div>
      </div>
      <div className="flex space-x-2">
        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">View</button>
        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">Download</button>
      </div>
    </div>
  )
}

