'use client'

import React, { useState, useEffect } from 'react'
import ApprovalGate from './ApprovalGate'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { 
  AlertCircle, 
  CheckCircle2, 
  Clock, 
  RefreshCw,
  Play,
  Pause
} from 'lucide-react'
import { toast } from 'react-hot-toast'

interface ApprovalGateManagerProps {
  workflowId: string
  onWorkflowResume?: () => void
  onWorkflowPause?: () => void
}

interface ApprovalGate {
  gate_id: string
  gate_type: string
  title: string
  description: string
  proposal: Record<string, any>
  educational_explanation: string
  status: string
  created_at: string
  approved_at?: string
  user_comments?: string
  modifications?: Record<string, any>
}

export default function ApprovalGateManager({ 
  workflowId, 
  onWorkflowResume,
  onWorkflowPause 
}: ApprovalGateManagerProps) {
  const [gates, setGates] = useState<ApprovalGate[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [workflowPaused, setWorkflowPaused] = useState(false)

  // Fetch approval gates
  const fetchGates = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`/api/workflow/${workflowId}/approval-gates`)
      if (response.ok) {
        const data = await response.json()
        setGates(data.gates || [])
        setWorkflowPaused(data.workflow_paused || false)
      }
    } catch (error) {
      console.error('Error fetching approval gates:', error)
      toast.error('Failed to fetch approval gates')
    } finally {
      setIsLoading(false)
    }
  }

  // Handle gate approval
  const handleApprove = async (gateId: string, comments?: string) => {
    try {
      const response = await fetch(`/api/workflow/${workflowId}/approval-gates/${gateId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ comments }),
      })

      if (response.ok) {
        await fetchGates() // Refresh gates
        toast.success('Proposal approved successfully!')
      } else {
        throw new Error('Failed to approve proposal')
      }
    } catch (error) {
      console.error('Error approving gate:', error)
      toast.error('Failed to approve proposal')
    }
  }

  // Handle gate rejection
  const handleReject = async (gateId: string, comments: string) => {
    try {
      const response = await fetch(`/api/workflow/${workflowId}/approval-gates/${gateId}/reject`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ comments }),
      })

      if (response.ok) {
        await fetchGates() // Refresh gates
        toast.success('Proposal rejected')
      } else {
        throw new Error('Failed to reject proposal')
      }
    } catch (error) {
      console.error('Error rejecting gate:', error)
      toast.error('Failed to reject proposal')
    }
  }

  // Handle gate modification
  const handleModify = async (gateId: string, modifications: Record<string, any>, comments: string) => {
    try {
      const response = await fetch(`/api/workflow/${workflowId}/approval-gates/${gateId}/modify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ modifications, comments }),
      })

      if (response.ok) {
        await fetchGates() // Refresh gates
        toast.success('Proposal modified and approved!')
      } else {
        throw new Error('Failed to modify proposal')
      }
    } catch (error) {
      console.error('Error modifying gate:', error)
      toast.error('Failed to modify proposal')
    }
  }

  // Resume workflow
  const handleResumeWorkflow = async () => {
    try {
      const response = await fetch(`/api/workflow/${workflowId}/resume`, {
        method: 'POST',
      })

      if (response.ok) {
        setWorkflowPaused(false)
        onWorkflowResume?.()
        toast.success('Workflow resumed')
      } else {
        throw new Error('Failed to resume workflow')
      }
    } catch (error) {
      console.error('Error resuming workflow:', error)
      toast.error('Failed to resume workflow')
    }
  }

  // Pause workflow
  const handlePauseWorkflow = async () => {
    try {
      const response = await fetch(`/api/workflow/${workflowId}/pause`, {
        method: 'POST',
      })

      if (response.ok) {
        setWorkflowPaused(true)
        onWorkflowPause?.()
        toast.success('Workflow paused')
      } else {
        throw new Error('Failed to pause workflow')
      }
    } catch (error) {
      console.error('Error pausing workflow:', error)
      toast.error('Failed to pause workflow')
    }
  }

  // Load gates on mount
  useEffect(() => {
    fetchGates()
  }, [workflowId])

  // Auto-refresh every 5 seconds if workflow is paused
  useEffect(() => {
    if (workflowPaused) {
      const interval = setInterval(fetchGates, 5000)
      return () => clearInterval(interval)
    }
  }, [workflowPaused])

  const pendingGates = gates.filter(gate => gate.status === 'pending')
  const completedGates = gates.filter(gate => gate.status !== 'pending')

  const getStatusSummary = () => {
    const total = gates.length
    const pending = pendingGates.length
    const approved = gates.filter(g => g.status === 'approved').length
    const rejected = gates.filter(g => g.status === 'rejected').length
    const modified = gates.filter(g => g.status === 'modified').length

    return { total, pending, approved, rejected, modified }
  }

  const statusSummary = getStatusSummary()

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">Approval Gates</CardTitle>
              <CardDescription>
                Review and approve key decisions in the machine learning workflow
              </CardDescription>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={fetchGates}
                disabled={isLoading}
                variant="outline"
                size="sm"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              
              {workflowPaused ? (
                <Button
                  onClick={handleResumeWorkflow}
                  className="bg-green-600 hover:bg-green-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Resume Workflow
                </Button>
              ) : (
                <Button
                  onClick={handlePauseWorkflow}
                  variant="outline"
                >
                  <Pause className="w-4 h-4 mr-2" />
                  Pause Workflow
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{statusSummary.total}</div>
              <div className="text-sm text-gray-600">Total Gates</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{statusSummary.pending}</div>
              <div className="text-sm text-gray-600">Pending</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{statusSummary.approved}</div>
              <div className="text-sm text-gray-600">Approved</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{statusSummary.modified}</div>
              <div className="text-sm text-gray-600">Modified</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{statusSummary.rejected}</div>
              <div className="text-sm text-gray-600">Rejected</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Workflow Status */}
      {workflowPaused && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
              <div>
                <h3 className="font-semibold text-yellow-900">Workflow Paused</h3>
                <p className="text-sm text-yellow-800">
                  The workflow is waiting for your approval on pending gates. 
                  Please review and approve the proposals to continue.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pending Gates */}
      {pendingGates.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-yellow-600" />
            <h2 className="text-xl font-semibold">Pending Approvals</h2>
            <Badge variant="outline" className="text-yellow-600 border-yellow-600">
              {pendingGates.length}
            </Badge>
          </div>
          
          <div className="space-y-4">
            {pendingGates.map((gate) => (
              <ApprovalGate
                key={gate.gate_id}
                gate={gate}
                onApprove={handleApprove}
                onReject={handleReject}
                onModify={handleModify}
                disabled={isLoading}
              />
            ))}
          </div>
        </div>
      )}

      {/* Completed Gates */}
      {completedGates.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-green-600" />
            <h2 className="text-xl font-semibold">Completed Approvals</h2>
            <Badge variant="outline" className="text-green-600 border-green-600">
              {completedGates.length}
            </Badge>
          </div>
          
          <div className="space-y-4">
            {completedGates.map((gate) => (
              <ApprovalGate
                key={gate.gate_id}
                gate={gate}
                onApprove={handleApprove}
                onReject={handleReject}
                onModify={handleModify}
                disabled={true}
              />
            ))}
          </div>
        </div>
      )}

      {/* No Gates Message */}
      {gates.length === 0 && !isLoading && (
        <Card>
          <CardContent className="pt-6 text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Approval Gates</h3>
            <p className="text-gray-600">
              No approval gates are currently active for this workflow.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
