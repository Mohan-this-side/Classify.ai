'use client'

import React, { useState, useEffect, useCallback } from 'react'
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
  const [description, setDescription] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [columnOptions, setColumnOptions] = useState<string[]>([])
  
  // Workflow state
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [workflowStatus, setWorkflowStatus] = useState<string>('idle')
  const [agents, setAgents] = useState<any[]>([])
  const [pmMessages, setPmMessages] = useState<any[]>([])
  const [sandboxMetrics, setSandboxMetrics] = useState({ cpu: 0, memory: 0, time: 0 })
  const [results, setResults] = useState<any>(null)
  const [workflowCompletedNotified, setWorkflowCompletedNotified] = useState(false) // ✅ FIX: Track if completion notification shown

  // ✅ FIX: Reset state function (memoized with useCallback)
  const resetAppState = useCallback(() => {
    console.log('🔄 Resetting app state...')
    setActiveView('upload')
    setWorkflowId(null)
    setWorkflowStatus('idle')
    setAgents([])
    setPmMessages([])
    setSandboxMetrics({ cpu: 0, memory: 0, time: 0 })
    setResults(null)
    setWorkflowCompletedNotified(false)
    setPendingApproval(false)
    setFile(null)
    setTargetColumn('')
    setDescription('')
    setApiKey('')
    setColumnOptions([])
  }, [])

  // ✅ FIX: Cancel any running workflows and reset state on page load/refresh
  useEffect(() => {
    const cancelRunningWorkflows = async () => {
      try {
        // Use the cancel-all endpoint for efficiency
        const response = await fetch('http://localhost:8000/api/workflow/cancel-all', {
          method: 'DELETE'
        })
        if (response.ok) {
          const data = await response.json()
          if (data.cancelled_count > 0) {
            console.log(`✅ Cancelled ${data.cancelled_count} running workflow(s) on page load`)
          }
        }
      } catch (error) {
        console.error('Error cancelling running workflows:', error)
        // Try individual cancellation as fallback
        try {
          const listResponse = await fetch('http://localhost:8000/api/workflow/list?status=running&limit=10')
          if (listResponse.ok) {
            const data = await listResponse.json()
            const runningWorkflows = data.workflows || []
            for (const workflow of runningWorkflows) {
              if (workflow.workflow_id) {
                try {
                  await fetch(`http://localhost:8000/api/workflow/${workflow.workflow_id}`, {
                    method: 'DELETE'
                  })
                } catch (e) {
                  // Ignore individual errors
                }
              }
            }
          }
        } catch (e) {
          // Ignore fallback errors
        }
      }
      
      // Reset frontend state
      resetAppState()
    }

    cancelRunningWorkflows()
  }, [resetAppState]) // Run once on mount

  // ✅ Function to navigate to workflow view
  const navigateToWorkflowView = () => {
    console.log('🔄 Navigating to workflow view...')
    // Set workflow view immediately
    setActiveView('workflow')
    // Scroll to top to ensure workflow view is visible
    setTimeout(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }, 100)
  }

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
    // Validate all required fields
    const missingFields = []
    if (!file) missingFields.push('file')
    if (!targetColumn) missingFields.push('target column')
    if (!description || description.trim() === '') missingFields.push('description')
    if (!apiKey || apiKey.trim() === '') missingFields.push('API key')
    
    if (missingFields.length > 0) {
      toast.error(`Please fill in: ${missingFields.join(', ')}`)
      console.error('Missing fields:', missingFields)
      return
    }

    try {
      console.log('Starting workflow with:', {
        fileName: file?.name,
        targetColumn,
        descriptionLength: description.length,
        apiKeyLength: apiKey.length
      })

      // ✅ Navigate to workflow view immediately (before backend response)
      // This gives immediate feedback to the user
      navigateToWorkflowView()

      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', file!)
      formData.append('target_column', targetColumn)
      formData.append('description', description)
      formData.append('api_key', apiKey)
      formData.append('user_id', 'web_user')

      console.log('Sending request to backend...')
      
      // Send to backend
      const response = await fetch('http://localhost:8000/api/workflow/start', {
        method: 'POST',
        body: formData
      })

      console.log('Response status:', response.status, response.statusText)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Backend error:', errorText)
        throw new Error(`Failed to start workflow: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      console.log('Workflow started:', data)
      
      if (!data.workflow_id) {
        throw new Error('No workflow_id returned from backend')
      }

      setWorkflowId(data.workflow_id)
      setWorkflowStatus('running')
      setWorkflowCompletedNotified(false) // ✅ FIX: Reset completion notification flag
      
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

      // Note: Navigation already happened at the start of the function
      // This ensures user sees workflow page immediately
      toast.success('Workflow started successfully!')

      // Start polling for status
      pollWorkflowStatus(data.workflow_id)
    } catch (error: any) {
      console.error('Error starting workflow:', error)
      const errorMessage = error.message || 'Failed to start workflow. Please check console for details.'
      toast.error(errorMessage)
    }
  }

  const pollWorkflowStatus = async (wfId: string) => {
    let intervalId: NodeJS.Timeout | null = null
    let hasCompleted = false // ✅ FIX: Track if workflow has completed to prevent duplicate notifications
    
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
              const layer = data.layer_usage?.[backendAgentName] || 'layer1'
              // Standardize layer naming: always "Layer 1" or "Layer 2"
              const layerName = layer.toLowerCase().includes('layer2') || layer.toLowerCase().includes('2') ? 'Layer 2' : 'Layer 1'
              const layerEmoji = layerName === 'Layer 2' ? '🐳' : '⚡'
              return { 
                ...agent, 
                status: status === 'running' ? 'active' : status === 'completed' ? 'complete' : status,
                time: status === 'completed' ? `${layerEmoji} ${layerName}` : (status === 'running' ? `${layerEmoji} ${layerName}...` : ''),
                layer: layerName
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
        const hasPendingApproval = !!data.pending_approval
        if (hasPendingApproval && !pendingApproval) {
          // Approval gate just appeared - notify user
          setPendingApproval(true)
          
          // Request browser notification permission if not already granted
          if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission()
          }
          
          // Show browser notification if permission granted
          if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('⏸️ Approval Required', {
              body: 'The workflow is paused and waiting for your approval. Click to review.',
              icon: '/favicon.ico',
              tag: 'approval-gate',
              requireInteraction: true
            })
          }
          
          // Show toast notification
          toast('⏸️ Approval Required - Workflow is paused', {
            duration: 10000,
            position: 'top-right',
            icon: '⏸️'
          })
          
          // Auto-expand PM chat if minimized to show approval gate
          if (!pmExpanded) {
            setPmExpanded(true)
          }
        } else if (!hasPendingApproval && pendingApproval) {
          setPendingApproval(false)
        }

        // ✅ Update sandbox metrics from backend
        if (data.sandbox_metrics) {
          const metrics = data.sandbox_metrics
          // Parse CPU and Memory percentages correctly
          const cpuValue = typeof metrics.cpu === 'number' ? metrics.cpu : parseFloat(String(metrics.cpu || '0').replace('%', '')) || 0
          const memoryValue = typeof metrics.memory === 'number' ? metrics.memory : parseFloat(String(metrics.memory || '0').replace('%', '')) || 0
          const timeValue = typeof metrics.time === 'number' ? metrics.time : parseInt(String(metrics.time || '0')) || 0
          
          setSandboxMetrics({
            cpu: Math.min(100, Math.max(0, Math.round(cpuValue))),
            memory: Math.min(100, Math.max(0, Math.round(memoryValue))),
            time: Math.max(0, Math.round(120 - timeValue))  // Time remaining out of 120s
          })
        } else {
          // Reset metrics when no active sandbox execution
          setSandboxMetrics({ cpu: 0, memory: 0, time: 0 })
        }

        // ✅ FIX: Check if workflow is complete (only once)
        if (data.status === 'completed' && !hasCompleted) {
          hasCompleted = true
          clearInterval(interval)
          intervalId = null
          // Reset sandbox metrics
          setSandboxMetrics({ cpu: 0, memory: 0, time: 0 })
          // Fetch results (this will show the single completion notification)
          await fetchResults(wfId)
        } else if (data.status === 'failed' && !hasCompleted) {
          hasCompleted = true
          clearInterval(interval)
          intervalId = null
          toast.error('Workflow failed')
        }
      } catch (error) {
        console.error('Error polling status:', error)
      }
    }, 2000) // Poll every 2 seconds
  }

  // ✅ Fetch workflow results when complete
  const fetchResults = async (wfId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/results/${wfId}`)
      if (!response.ok) {
        throw new Error('Failed to fetch results')
      }
      
      const data = await response.json()
      console.log('Fetched results:', data) // Debug
      
      // ✅ FIX: Structure results for display - check multiple locations for each field
      const structuredResults = {
        // Model evaluation metrics - check top-level first, then nested
        model_evaluation: {
          evaluation_metrics: data.evaluation_metrics || data.results?.model_evaluation?.evaluation_metrics || data.results?.evaluation_metrics || {}
        },
        // EDA plots - check multiple locations
        eda_analysis: {
          plots: data.results?.eda_analysis?.plots || data.results?.eda_plots || data.results?.plots || data.eda_plots || []
        },
        // Feature importance - check multiple locations
        feature_importance: data.feature_importance_model || data.results?.feature_importance_model || data.results?.model_evaluation?.feature_importance || data.results?.feature_importance || {},
        // Dataset info
        dataset_info: data.results?.dataset_info || data.dataset_info || {},
        // Downloadable files - check top-level first, then nested
        downloadable_files: data.downloadable_files || data.results?.downloadable_files || data.downloads?.downloadable_files || [],
        // Execution info for notification
        execution_info: data.execution_info || {},
        // ✅ ADD: Workflow summary
        workflow_summary: data.workflow_summary || ""
      }
      
      console.log('Structured results:', structuredResults) // Debug
      
      // Store workflow ID in results for use in ResultsView
      ;(structuredResults as any).workflow_id = wfId
      
      setResults(structuredResults)
      setActiveView('results')
      
      // ✅ FIX: Show single, informative completion notification (only once)
      if (!workflowCompletedNotified) {
        setWorkflowCompletedNotified(true)
        const completedAgentsCount = data.execution_info?.completed_agents?.length || 7
        const metrics = data.evaluation_metrics || data.results?.model_evaluation?.evaluation_metrics || {}
        const accuracy = metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'
        
        toast.success(
          `🎉 Analysis Complete! ${completedAgentsCount} agents finished successfully. Model accuracy: ${accuracy}`,
          {
            duration: 5000,
            icon: '🎉',
            style: {
              background: '#10b981',
              color: '#fff',
              fontSize: '14px',
              padding: '16px',
              borderRadius: '8px',
            }
          }
        )
      }
    } catch (error) {
      console.error('Error fetching results:', error)
      toast.error('Failed to fetch results')
    }
  }

  // ✅ Handle approval gate responses
  const handleApprovalResponse = async (action: 'approve' | 'reject' | 'modify', comment?: string) => {
    if (!workflowId) {
      console.error('❌ No workflowId available for approval')
      toast.error('Workflow ID not found. Please refresh the page.')
      return
    }
    
    console.log(`🔄 Handling approval: ${action} for workflow ${workflowId}`)
    
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
      
      console.log(`✅ Approval response status: ${response.status}`)
      
      if (response.ok) {
        const data = await response.json()
        console.log('✅ Approval successful:', data)
        toast.success(`Approval ${action}d successfully`)
        setPendingApproval(false)
        // Refresh workflow status after approval
        if (workflowId) {
          pollWorkflowStatus(workflowId)
        }
      } else {
        const errorText = await response.text()
        console.error('❌ Approval failed:', errorText)
        toast.error(`Failed to ${action} workflow: ${response.status}`)
      }
    } catch (error: any) {
      console.error('❌ Error handling approval:', error)
      toast.error(`Failed to process approval: ${error.message || error}`)
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
          description={description}
          setDescription={setDescription}
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
          workflowStatus={workflowStatus}
          workflowId={workflowId}
          onApprovalResponse={handleApprovalResponse}
          onPMQuestion={handlePMQuestion}
          onCancelWorkflow={async () => {
            if (workflowId && (workflowStatus === 'running' || workflowStatus === 'paused')) {
              try {
                const response = await fetch(`http://localhost:8000/api/workflow/${workflowId}`, {
                  method: 'DELETE'
                })
                if (response.ok) {
                  toast.success('Workflow cancelled')
                  // ✅ FIX: Reset all state on cancel
                  resetAppState()
                } else {
                  toast.error('Failed to cancel workflow')
                }
              } catch (error) {
                console.error('Error cancelling workflow:', error)
                toast.error('Failed to cancel workflow')
                // Still reset state even if cancel request fails
                resetAppState()
              }
            }
          }}
        />
      )}
      
      {activeView === 'results' && <ResultsView results={results} workflowId={workflowId} />}
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

function UploadView({ file, handleFileChange, targetColumn, setTargetColumn, description, setDescription, columnOptions, apiKey, setApiKey, onStart }: any) {
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

        {/* Dataset Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Dataset Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe your dataset and what you want to predict (e.g., 'Customer churn prediction dataset with demographic and usage features')"
            rows={3}
            className="w-full border border-gray-300 rounded-lg px-4 py-3 bg-white text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all resize-none"
          />
          <p className="text-xs text-gray-500 mt-1">
            This description helps the AI understand your dataset better and generate more accurate analysis.
          </p>
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
          type="button"
          onClick={async (e) => {
            e.preventDefault()
            e.stopPropagation()
            console.log('🔵🔵🔵 Button clicked!', {
              file: !!file,
              targetColumn: !!targetColumn,
              description: !!(description && description.trim()),
              apiKey: !!(apiKey && apiKey.trim()),
              onStartType: typeof onStart,
              onStartExists: !!onStart
            })
            
            // Validate fields
            if (!file || !targetColumn || !(description && description.trim()) || !(apiKey && apiKey.trim())) {
              console.error('❌ Validation failed')
              toast.error('Please fill in all required fields')
              return
            }
            
            // Call onStart function directly
            console.log('✅ About to call onStart function...')
            if (onStart && typeof onStart === 'function') {
              console.log('✅ Calling onStart function...')
              try {
                const result = await onStart()
                console.log('✅ onStart completed:', result)
              } catch (error: any) {
                console.error('❌ Error in onStart:', error)
                toast.error(`Failed to start workflow: ${error.message || error}`)
              }
            } else {
              console.error('❌ onStart is not a function:', typeof onStart, onStart)
              toast.error('Workflow start function not available. Please refresh the page.')
            }
          }}
          disabled={!file || !targetColumn || !(description && description.trim()) || !(apiKey && apiKey.trim())}
          style={{ pointerEvents: 'auto', cursor: 'pointer', zIndex: 10 }}
          className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 rounded-lg font-semibold text-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
        >
          <Play className="w-6 h-6" />
          <span>Start Analysis</span>
        </button>
        
        {/* Debug info */}
        {process.env.NODE_ENV === 'development' && (
          <div className="text-xs text-gray-400 mt-2">
            Debug: File={file ? '✓' : '✗'} | Target={targetColumn ? '✓' : '✗'} | Desc={description?.trim() ? '✓' : '✗'} | API={apiKey?.trim() ? '✓' : '✗'} | Button Disabled={(!file || !targetColumn || !description || !description.trim() || !apiKey || !apiKey.trim()) ? 'YES' : 'NO'} | onStart={typeof onStart}
          </div>
        )}
      </div>
    </div>
  )
}

function WorkflowView({ agents, pmExpanded, setPmExpanded, pendingApproval, setPendingApproval, pmMessages, sandboxMetrics, workflowStatus, workflowId, onApprovalResponse, onPMQuestion, onCancelWorkflow }: any) {
  // Pass workflowId to CompletedAgent components
  // Map agent IDs to icons and labels
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
  
  const agentLabels: any = {
    discovery: 'Data Discovery',
    eda: 'EDA Analysis',
    cleaning: 'Data Cleaning',
    feature: 'Feature Engineering',
    model: 'Model Building',
    eval: 'Model Evaluation',
    report: 'Technical Reporting',
    pm: 'Project Manager'
  }

  // Add icons to agents
  const agentsWithIcons = agents.map((agent: any) => ({
    ...agent,
    icon: iconMap[agent.id] || Circle,
    label: agentLabels[agent.id] || agent.label
  }))
  
  // Find active agent
  const activeAgent = agentsWithIcons.find((a: any) => a.status === 'active' || a.status === 'running')
  const completedAgents = agentsWithIcons.filter((a: any) => a.status === 'complete' || a.status === 'completed')
  
  // Auto-scroll PM chat to bottom
  const pmMessagesEndRef = React.useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (pmMessagesEndRef.current) {
      pmMessagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [pmMessages])

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Main Content */}
      <div className={`flex-1 flex flex-col transition-all ${pmExpanded ? 'mr-96' : 'mr-0'}`}>
        {/* Workflow Header with Cancel Button */}
        {(workflowStatus === 'running' || workflowStatus === 'paused') && workflowId && (
          <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm">
            <div className="flex items-center space-x-4">
              <div className={`w-3 h-3 rounded-full ${workflowStatus === 'running' ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></div>
              <span className="text-sm font-medium text-gray-700">
                {workflowStatus === 'running' ? 'Workflow Running' : 'Workflow Paused'}
              </span>
            </div>
            {onCancelWorkflow && (
              <button
                onClick={onCancelWorkflow}
                className="px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors flex items-center space-x-2"
              >
                <span>✕</span>
                <span>Cancel Workflow</span>
              </button>
            )}
          </div>
        )}
        {/* Timeline */}
        <div className="bg-white border-b border-gray-200 px-6 py-6 overflow-x-auto shadow-sm">
          <div className="flex items-center justify-between min-w-max max-w-6xl mx-auto">
            {agentsWithIcons.map((agent: any, idx: number) => (
              <React.Fragment key={agent.id}>
                <AgentStep {...agent} />
                {idx < agentsWithIcons.length - 1 && (
                  <div className={`w-12 h-1 ${agent.status === 'complete' || agent.status === 'completed' ? 'bg-blue-600' : 'bg-gray-300'}`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Agent Activity */}
        <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
          <div className="max-w-5xl mx-auto space-y-6">
            {/* Active Agent Card - Dynamic */}
            {activeAgent ? (
              <ActiveAgentCard agent={activeAgent} sandboxMetrics={sandboxMetrics} />
            ) : workflowStatus === 'completed' ? (
              <div className="bg-green-50 rounded-xl shadow-md border border-green-200 p-6">
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-8 h-8 text-green-600" />
                  <div>
                    <h3 className="text-xl font-bold text-green-900">Workflow Completed!</h3>
                    <p className="text-sm text-green-700">All agents have finished successfully. Check the Results tab.</p>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Sandbox Monitor - Only show when active */}
            {(activeAgent && (sandboxMetrics.cpu > 0 || sandboxMetrics.memory > 0 || sandboxMetrics.time > 0)) && (
              <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
                <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Code className="w-5 h-5 text-gray-700" />
                  <span>Sandbox Execution Monitor</span>
                </h4>
                <div className="grid grid-cols-3 gap-6">
                  <MetricBar label="CPU Usage" value={Math.min(100, Math.max(0, sandboxMetrics.cpu))} color="green" />
                  <MetricBar label="Memory" value={Math.min(100, Math.max(0, sandboxMetrics.memory))} color="blue" />
                  <div>
                    <p className="text-xs text-gray-500 mb-2">Time Remaining</p>
                    <p className="text-lg font-bold text-gray-900">~{Math.max(0, sandboxMetrics.time)}s</p>
                  </div>
                </div>
              </div>
            )}

            {/* Completed Agents */}
            {completedAgents.map((agent: any) => (
              <CompletedAgent 
                key={agent.id}
                icon={agent.icon} 
                name={agent.label || agent.name} 
                time={agent.time || 'Completed'}
                agentId={agent.id}
                workflowId={workflowId}
              />
            ))}
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
            {pmMessages.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <MessageSquare className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p className="text-sm">No messages yet. Ask me anything!</p>
              </div>
            ) : (
              <>
                {pmMessages.map((message: any, index: number) => (
                  <PMMessage 
                    key={index}
                    agent={message.agent || 'System'} 
                    time={new Date(message.timestamp).toLocaleTimeString()} 
                    message={message.content}
                    type={message.type}
                  />
                ))}
                <div ref={pmMessagesEndRef} />
              </>
            )}

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

function ResultsView({ results, workflowId }: any) {
  // Get workflowId from props or from results
  const currentWorkflowId = workflowId || results?.workflow_id || null
  
  // ✅ ADD: Full-screen image viewer state
  const [fullScreenImage, setFullScreenImage] = useState<string | null>(null)
  
  // ✅ ADD: PM chatbot state for results page
  const [pmExpanded, setPmExpanded] = useState(true)
  const [pmMessages, setPmMessages] = useState<any[]>([])
  const [pmQuestion, setPmQuestion] = useState('')
  
  // ✅ ADD: Fetch PM messages for results page
  useEffect(() => {
    if (currentWorkflowId) {
      const fetchPMMessages = async () => {
        try {
          const response = await fetch(`http://localhost:8000/api/workflow/status/${currentWorkflowId}`)
          if (response.ok) {
            const data = await response.json()
            setPmMessages(data.pm_messages || [])
          }
        } catch (error) {
          console.error('Error fetching PM messages:', error)
        }
      }
      fetchPMMessages()
    }
  }, [currentWorkflowId])
  
  // ✅ ADD: Handle PM questions on results page
  const handlePMQuestion = async (question: string) => {
    if (!currentWorkflowId || !question.trim()) return
    
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/${currentWorkflowId}/pm/question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      
      if (response.ok) {
        const data = await response.json()
        setPmMessages(prev => [...prev, {
          type: 'question',
          content: question,
          timestamp: new Date().toISOString(),
          agent: 'You'
        }, {
          type: 'answer',
          content: data.answer,
          timestamp: new Date().toISOString(),
          agent: 'Project Manager'
        }])
        setPmQuestion('')
      }
    } catch (error) {
      console.error('Error asking PM question:', error)
    }
  }
  
  // Auto-scroll PM chat to bottom
  const pmMessagesEndRef = React.useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (pmMessagesEndRef.current) {
      pmMessagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [pmMessages])
  
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
    <div className="flex-1 flex overflow-hidden">
      {/* Main Results Content */}
      <div className="flex-1 overflow-y-auto p-8 bg-gray-50">
        <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <h2 className="text-4xl font-bold text-gray-900">Analysis Complete! 🎉</h2>
          <button className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-all shadow-lg">
            <Download className="w-5 h-5" />
            <span>Download All</span>
          </button>
        </div>

        {/* Metrics - ✅ FIX: Check multiple locations for evaluation_metrics */}
        <div className="grid grid-cols-4 gap-6">
          <MetricCard 
            label="Accuracy" 
            value={(() => {
              const metrics = results?.model_evaluation?.evaluation_metrics || results?.evaluation_metrics || {}
              return metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'
            })()} 
            gradient="from-blue-500 to-blue-600" 
          />
          <MetricCard 
            label="F1 Score" 
            value={(() => {
              const metrics = results?.model_evaluation?.evaluation_metrics || results?.evaluation_metrics || {}
              // Try multiple key variations
              const f1 = metrics.f1_weighted || metrics.f1_score || metrics.f1 || metrics['f1-weighted'] || 0
              return f1 ? f1.toFixed(3) : 'N/A'
            })()} 
            gradient="from-green-500 to-green-600" 
          />
          <MetricCard 
            label="Precision" 
            value={(() => {
              const metrics = results?.model_evaluation?.evaluation_metrics || results?.evaluation_metrics || {}
              // Try multiple key variations
              const precision = metrics.precision_weighted || metrics.precision_score || metrics.precision || metrics['precision-weighted'] || 0
              return precision ? `${(precision * 100).toFixed(1)}%` : 'N/A'
            })()} 
            gradient="from-purple-500 to-purple-600" 
          />
          <MetricCard 
            label="Recall" 
            value={(() => {
              const metrics = results?.model_evaluation?.evaluation_metrics || results?.evaluation_metrics || {}
              // Try multiple key variations
              const recall = metrics.recall_weighted || metrics.recall_score || metrics.recall || metrics['recall-weighted'] || 0
              return recall ? `${(recall * 100).toFixed(1)}%` : 'N/A'
            })()} 
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
              results.eda_analysis.plots.map((plot: any, idx: number) => {
                const plotUrl = `http://localhost:8000${plot.path || plot.url}`
                return (
                  <div key={idx} className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200 p-4 cursor-pointer hover:shadow-lg transition-shadow" onClick={() => setFullScreenImage(plotUrl)}>
                    <p className="font-medium text-sm mb-2">{plot.title || plot.name || `Plot ${idx + 1}`}</p>
                    <img 
                      src={plotUrl} 
                      alt={plot.title || plot.name || `Plot ${idx + 1}`}
                      className="w-full h-auto rounded"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5QbG90IG5vdCBhdmFpbGFibGU8L3RleHQ+PC9zdmc+'
                      }}
                    />
                  </div>
                )
              })
            ) : (
              <div className="col-span-2 text-center py-8 text-gray-500">
                <p>No EDA plots generated. Check backend logs for EDA agent execution.</p>
              </div>
            )}
          </div>
          
          {/* ✅ ADD: Full-screen image viewer */}
          {fullScreenImage && (
            <div 
              className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4"
              onClick={() => setFullScreenImage(null)}
            >
              <div className="relative max-w-7xl max-h-full">
                <button
                  onClick={() => setFullScreenImage(null)}
                  className="absolute top-4 right-4 bg-white rounded-full p-2 hover:bg-gray-200 transition-colors z-10"
                  aria-label="Close"
                >
                  <X className="w-6 h-6 text-gray-800" />
                </button>
                <img 
                  src={fullScreenImage} 
                  alt="Full screen plot"
                  className="max-w-full max-h-[90vh] object-contain rounded-lg"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
            </div>
          )}
        </div>

        {/* Summary - ✅ ADD: Comprehensive workflow summary */}
        {results?.workflow_summary && (
          <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-3">
              <FileText className="w-7 h-7 text-blue-600" />
              <span>Summary</span>
            </h3>
            <div className="prose prose-sm max-w-none">
              <MarkdownContent content={results.workflow_summary} />
            </div>
          </div>
        )}

        {/* Feature Importance */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Feature Importance</h3>
          <div className="space-y-4">
            {results?.feature_importance && Object.keys(results.feature_importance).length > 0 ? (
              Object.entries(results.feature_importance)
                .sort(([, a]: any, [, b]: any) => b - a)
                .slice(0, 10)
                .map(([feature, importance]: any) => (
                  <FeatureBar 
                    key={feature} 
                    label={feature} 
                    value={Math.round((importance || 0) * 100)} 
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
            {results?.downloadable_files && results.downloadable_files.length > 0 ? (
              results.downloadable_files.map((file: any, idx: number) => (
                <DeliverableItem 
                  key={idx}
                  name={file.name || `file_${idx + 1}`} 
                  size={file.size || 'Unknown'} 
                  downloadUrl={file.path || file.url}
                  workflowId={currentWorkflowId || undefined}
                />
              ))
            ) : (
              <>
                <DeliverableItem name="cleaned_dataset.csv" size="Processing..." workflowId={currentWorkflowId || undefined} />
                <DeliverableItem name="trained_model.joblib" size="Processing..." workflowId={currentWorkflowId || undefined} />
                <DeliverableItem name="analysis_notebook.ipynb" size="Processing..." workflowId={currentWorkflowId || undefined} />
              </>
            )}
          </div>
        </div>
        </div>
      </div>
      
      {/* ✅ ADD: PM Chatbot Panel (similar to WorkflowView) */}
      {pmExpanded && (
        <div className="w-96 bg-white border-l border-gray-200 flex flex-col shadow-xl">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MessageSquare className="w-6 h-6" />
              <h3 className="font-bold text-lg">Project Manager</h3>
            </div>
            <button
              onClick={() => setPmExpanded(false)}
              className="text-white hover:text-gray-200 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-gray-50">
            {pmMessages.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <MessageSquare className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p className="text-sm">No messages yet. Ask me anything about your results!</p>
              </div>
            ) : (
              <>
                {pmMessages.map((message: any, index: number) => (
                  <PMMessage 
                    key={index}
                    agent={message.agent || 'System'} 
                    time={new Date(message.timestamp).toLocaleTimeString()} 
                    message={message.content}
                    type={message.type}
                  />
                ))}
                <div ref={pmMessagesEndRef} />
              </>
            )}
          </div>

          <div className="p-4 border-t border-gray-200 bg-white">
            <div className="flex space-x-2">
              <input
                type="text"
                value={pmQuestion}
                onChange={(e) => setPmQuestion(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && pmQuestion.trim()) {
                    handlePMQuestion(pmQuestion)
                  }
                }}
                placeholder="Ask about your results..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <button
                onClick={() => pmQuestion.trim() && handlePMQuestion(pmQuestion)}
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
              >
                Send
              </button>
            </div>
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

// ========== HELPER COMPONENTS ==========

// ✅ ADD: Markdown content renderer for Summary box
function MarkdownContent({ content }: { content: string }) {
  if (!content) return null
  
  // Enhanced markdown parser with better table support
  const parseMarkdown = (text: string) => {
    let html = text
    
    // Process tables first (before other replacements)
    const tableRegex = /(\|.+\|\n\|[-:|\s]+\|\n(?:\|.+\|\n?)+)/g
    html = html.replace(tableRegex, (match) => {
      const lines = match.trim().split('\n').filter(l => l.trim())
      if (lines.length < 2) return match
      
      // Skip separator line
      const dataLines = lines.filter(l => !l.match(/^\|[-:\s|]+\|$/))
      
      let tableHtml = '<div class="overflow-x-auto my-4"><table class="min-w-full border-collapse border border-gray-300 bg-white"><thead><tr class="bg-gray-50">'
      
      // Header row
      const headerCells = dataLines[0].split('|').filter(c => c.trim())
      headerCells.forEach(cell => {
        tableHtml += `<th class="px-4 py-3 border border-gray-300 text-left font-semibold text-gray-900">${cell.trim()}</th>`
      })
      tableHtml += '</tr></thead><tbody>'
      
      // Data rows
      for (let i = 1; i < dataLines.length; i++) {
        const cells = dataLines[i].split('|').filter(c => c.trim())
        tableHtml += '<tr class="hover:bg-gray-50">'
        cells.forEach(cell => {
          tableHtml += `<td class="px-4 py-2 border border-gray-300 text-gray-700">${cell.trim()}</td>`
        })
        tableHtml += '</tr>'
      }
      
      tableHtml += '</tbody></table></div>'
      return tableHtml
    })
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold mt-6 mb-3 text-gray-900">$1</h3>')
    html = html.replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold mt-8 mb-4 text-gray-900">$1</h2>')
    html = html.replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mt-8 mb-4 text-gray-900">$1</h1>')
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em class="italic text-gray-700">$1</em>')
    
    // Bullet points and lists
    const lines = html.split('\n')
    let inList = false
    let listItems: string[] = []
    let processedLines: string[] = []
    
    lines.forEach((line, idx) => {
      const trimmed = line.trim()
      const isListItem = trimmed.match(/^[•\-\*]\s+(.+)$/) || trimmed.match(/^\d+\.\s+(.+)$/)
      
      if (isListItem) {
        if (!inList) {
          inList = true
          listItems = []
        }
        const content = isListItem[1] || trimmed.replace(/^[•\-\*]\s+/, '').replace(/^\d+\.\s+/, '')
        listItems.push(`<li class="ml-4 mb-2 text-gray-700">${content}</li>`)
      } else {
        if (inList && listItems.length > 0) {
          processedLines.push(`<ul class="list-disc space-y-1 mb-4 ml-6">${listItems.join('')}</ul>`)
          listItems = []
          inList = false
        }
        if (trimmed && !trimmed.startsWith('<')) {
          processedLines.push(`<p class="mb-4 text-gray-700 leading-relaxed">${trimmed}</p>`)
        } else if (trimmed) {
          processedLines.push(line)
        }
      }
    })
    
    if (inList && listItems.length > 0) {
      processedLines.push(`<ul class="list-disc space-y-1 mb-4 ml-6">${listItems.join('')}</ul>`)
    }
    
    html = processedLines.join('\n')
    
    // Clean up multiple consecutive <p> tags
    html = html.replace(/<\/p>\n<p class="mb-4 text-gray-700 leading-relaxed">/g, '<br />')
    
    return html
  }
  
  const htmlContent = parseMarkdown(content)
  
  return (
    <div 
      className="markdown-content text-gray-800 prose prose-sm max-w-none"
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  )
}

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

function CompletedAgent({ icon: Icon, name, time, agentId, workflowId }: any) {
  const [showDetails, setShowDetails] = useState(false)
  const [summary, setSummary] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  
  const fetchSummary = async () => {
    if (!workflowId || !agentId || summary !== null) return
    
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/api/workflow/${workflowId}/agent/${agentId}/summary`)
      if (response.ok) {
        const data = await response.json()
        setSummary(data.summary || "Agent execution completed successfully.")
      } else {
        setSummary("Agent execution completed successfully. Details will be available in the final report.")
      }
    } catch (error) {
      console.error('Error fetching agent summary:', error)
      setSummary("Agent execution completed successfully. Details will be available in the final report.")
    } finally {
      setLoading(false)
    }
  }
  
  const handleToggleDetails = () => {
    if (!showDetails && summary === null) {
      fetchSummary()
    }
    setShowDetails(!showDetails)
  }
  
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 hover:shadow-md transition-all">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Icon className="w-6 h-6 text-green-600" />
          <div>
            <h4 className="font-semibold text-gray-900">{name}</h4>
            <p className="text-sm text-gray-500">{time}</p>
          </div>
        </div>
        <button 
          onClick={handleToggleDetails}
          className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
        >
          {showDetails ? 'Hide Details' : 'View Details'}
        </button>
      </div>
      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          {loading ? (
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Loader className="w-4 h-4 animate-spin" />
              <span>Loading summary...</span>
            </div>
          ) : summary ? (
            <div className="text-sm text-gray-700 whitespace-pre-line space-y-2">
              {summary.split('\n').map((line, idx) => {
                // Format markdown-style headers
                if (line.startsWith('**') && line.endsWith('**')) {
                  return <h5 key={idx} className="font-bold text-gray-900 mt-3 mb-1">{line.replace(/\*\*/g, '')}</h5>
                }
                // Format bullet points
                if (line.startsWith('- ')) {
                  return <div key={idx} className="ml-4">{line}</div>
                }
                // Regular text
                return <p key={idx}>{line}</p>
              })}
            </div>
          ) : (
            <p className="text-sm text-gray-600">
              Agent execution completed successfully. Details will be available in the final report.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

// Active Agent Card Component
function ActiveAgentCard({ agent, sandboxMetrics }: any) {
  const hasLayer2 = agent.layer === 'Layer 2' || agent.time?.includes('Layer 2')
  
  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
            <agent.icon className="w-7 h-7 text-blue-600" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-gray-900">{agent.label || agent.name}</h3>
            <p className="text-sm text-gray-600">Processing your dataset...</p>
          </div>
        </div>
        <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold flex items-center space-x-2">
          <Loader className="w-4 h-4 animate-spin" />
          <span>Running</span>
        </span>
      </div>

      <div className="space-y-4">
        {/* Layer 1 - Always shown */}
        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
          <div className="flex items-center space-x-2 mb-3">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <span className="font-semibold text-green-900">Layer 1: Analysis Complete</span>
          </div>
          <p className="text-sm text-gray-700 ml-7">
            Hardcoded analysis completed. Reliable baseline results generated.
          </p>
        </div>

        {/* Layer 2 - Show if agent is using Layer 2 */}
        {hasLayer2 && (
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
            <div className="flex items-center space-x-2 mb-3">
              <Loader className="w-5 h-5 text-blue-600 animate-spin" />
              <span className="font-semibold text-blue-900">Layer 2: LLM Code Generation & Execution</span>
            </div>
            <div className="ml-7">
              <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-blue-500 h-3 rounded-full transition-all animate-pulse" style={{width: '65%'}} />
              </div>
              <p className="text-xs text-gray-600 mt-2">Generating adaptive code with LLM and executing in Docker sandbox...</p>
            </div>
          </div>
        )}
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
  
  // ✅ ENHANCED: Render markdown properly (tables, headers, lists, bold, etc.)
  const renderMarkdown = (msg: string) => {
    if (!msg) return ''
    
    let html = msg
    
    // Process tables first (before other markdown)
    const tableRegex = /(\|.+\|\n\|[-:|\s]+\|\n(?:\|.+\|\n?)+)/g
    html = html.replace(tableRegex, (match) => {
      const rows = match.trim().split('\n').filter(r => r.trim())
      if (rows.length < 2) return match
      
      let tableHtml = '<div class="overflow-x-auto my-4"><table class="min-w-full border-collapse border border-gray-300 bg-white"><thead><tr class="bg-gray-50">'
      
      // Header row
      const headerRow = rows[0].split('|').filter(c => c.trim())
      headerRow.forEach(cell => {
        tableHtml += `<th class="border border-gray-300 px-4 py-2 text-left font-semibold text-gray-900">${cell.trim()}</th>`
      })
      tableHtml += '</tr></thead><tbody>'
      
      // Data rows (skip separator row)
      for (let i = 2; i < rows.length; i++) {
        const cells = rows[i].split('|').filter(c => c.trim())
        tableHtml += '<tr>'
        cells.forEach(cell => {
          tableHtml += `<td class="border border-gray-300 px-4 py-2 text-gray-700">${cell.trim()}</td>`
        })
        tableHtml += '</tr>'
      }
      
      tableHtml += '</tbody></table></div>'
      return tableHtml
    })
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold mt-4 mb-2 text-gray-900">$1</h3>')
    html = html.replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold mt-6 mb-3 text-gray-900">$1</h2>')
    html = html.replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mt-6 mb-4 text-gray-900">$1</h1>')
    
    // Bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
    html = html.replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
    
    // Lists
    html = html.replace(/^\* (.*$)/gim, '<li class="ml-4">$1</li>')
    html = html.replace(/^- (.*$)/gim, '<li class="ml-4">$1</li>')
    html = html.replace(/(<li.*<\/li>)/s, '<ul class="list-disc space-y-1 mb-4 ml-6">$1</ul>')
    
    // Paragraphs (lines that don't start with HTML tags)
    const lines = html.split('\n')
    const processedLines: string[] = []
    let inList = false
    let listItems: string[] = []
    
    lines.forEach(line => {
      const trimmed = line.trim()
      if (trimmed.startsWith('<li')) {
        inList = true
        listItems.push(trimmed)
      } else if (trimmed.startsWith('</ul>')) {
        if (inList && listItems.length > 0) {
          processedLines.push(`<ul class="list-disc space-y-1 mb-4 ml-6">${listItems.join('')}</ul>`)
          listItems = []
        }
        inList = false
        processedLines.push(trimmed)
      } else {
        if (inList && listItems.length > 0) {
          processedLines.push(`<ul class="list-disc space-y-1 mb-4 ml-6">${listItems.join('')}</ul>`)
          listItems = []
        }
        inList = false
        if (trimmed && !trimmed.startsWith('<') && !trimmed.match(/^#+\s/)) {
          processedLines.push(`<p class="mb-2 text-gray-700 leading-relaxed">${trimmed}</p>`)
        } else if (trimmed) {
          processedLines.push(trimmed)
        }
      }
    })
    
    if (inList && listItems.length > 0) {
      processedLines.push(`<ul class="list-disc space-y-1 mb-4 ml-6">${listItems.join('')}</ul>`)
    }
    
    html = processedLines.join('\n')
    
    // Clean up multiple consecutive <p> tags
    html = html.replace(/<\/p>\n<p class="mb-2 text-gray-700 leading-relaxed">/g, '<br />')
    
    return html
  }
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-purple-600">{agent}</span>
        <span className="text-xs text-gray-400">{time}</span>
      </div>
      <div className={`rounded-lg p-4 text-sm leading-relaxed shadow-sm border ${getMessageStyle()}`}>
        <div 
          className="prose prose-sm max-w-none"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(message) }}
        />
      </div>
    </div>
  )
}

// ✅ Approval Gate Component
function ApprovalGate({ onApprovalResponse }: any) {
  const [isProcessing, setIsProcessing] = useState(false)
  
  const handleApproval = async (action: 'approve' | 'reject' | 'modify') => {
    if (isProcessing) return // Prevent double clicks
    
    setIsProcessing(true)
    console.log(`🔄 Processing approval: ${action}`)
    
    try {
      if (onApprovalResponse && typeof onApprovalResponse === 'function') {
        await onApprovalResponse(action)
      } else {
        console.error('onApprovalResponse is not a function:', typeof onApprovalResponse)
      }
    } catch (error) {
      console.error('Error in approval handler:', error)
    } finally {
      // Reset after a short delay to allow UI update
      setTimeout(() => setIsProcessing(false), 1000)
    }
  }
  
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
              type="button"
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                handleApproval('approve')
              }}
              disabled={isProcessing}
              style={{ pointerEvents: isProcessing ? 'none' : 'auto', cursor: isProcessing ? 'wait' : 'pointer' }}
              className="bg-green-600 text-white py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Processing...' : '✓ Approve'}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                handleApproval('modify')
              }}
              disabled={isProcessing}
              style={{ pointerEvents: isProcessing ? 'none' : 'auto', cursor: isProcessing ? 'wait' : 'pointer' }}
              className="bg-gray-200 text-gray-700 py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Modify
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                handleApproval('reject')
              }}
              disabled={isProcessing}
              style={{ pointerEvents: isProcessing ? 'none' : 'auto', cursor: isProcessing ? 'wait' : 'pointer' }}
              className="bg-red-100 text-red-700 py-2.5 px-4 rounded-lg text-sm font-semibold hover:bg-red-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
  // ✅ FIX: Cap value at 100% to prevent overflow
  const cappedValue = Math.min(Math.max(value, 0), 100)
  const displayValue = Math.round(cappedValue)
  
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-2">
        <span className="font-semibold text-gray-900 truncate pr-2" title={label}>{label}</span>
        <span className="text-gray-500 whitespace-nowrap">{displayValue}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div 
          className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full transition-all" 
          style={{width: `${cappedValue}%`, maxWidth: '100%'}} 
        />
      </div>
    </div>
  )
}

function DeliverableItem({ name, size, downloadUrl, workflowId }: any) {
  const handleDownload = async () => {
    if (!downloadUrl && !workflowId) {
      console.error('No download URL or workflow ID provided')
      return
    }
    
    try {
      // If downloadUrl is a full path, extract file type
      let downloadPath = downloadUrl
      if (downloadUrl && workflowId) {
        // Determine file type from name
        let fileType = 'model'
        if (name.includes('dataset') || name.includes('.csv')) fileType = 'cleaned_dataset'
        else if (name.includes('notebook') || name.includes('.ipynb')) fileType = 'notebook'
        else if (name.includes('report') || name.includes('.md')) fileType = 'report'
        
        // Use download endpoint
        downloadPath = `http://localhost:8000/api/workflow/download/${workflowId}/${fileType}`
      } else if (downloadUrl && !downloadUrl.startsWith('http')) {
        // Relative path - prepend backend URL
        downloadPath = `http://localhost:8000${downloadUrl}`
      }
      
      // Open download link
      window.open(downloadPath, '_blank')
    } catch (error) {
      console.error('Download error:', error)
    }
  }
  
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
        {downloadUrl || workflowId ? (
          <button 
            onClick={handleDownload}
            className="text-sm text-blue-600 hover:text-blue-700 font-medium cursor-pointer"
          >
            Download
          </button>
        ) : (
          <span className="text-sm text-gray-400">Processing...</span>
        )}
      </div>
    </div>
  )
}

