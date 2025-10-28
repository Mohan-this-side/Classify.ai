'use client'

import React, { useState } from 'react'
import { Button } from './ui/button'
import { Textarea } from './ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { 
  CheckCircle, 
  XCircle, 
  Edit3, 
  Info, 
  Clock, 
  AlertCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { toast } from 'react-hot-toast'

interface ApprovalGateProps {
  gate: {
    gate_id: string
    gate_type: string
    title: string
    description: string
    proposal: Record<string, any>
    educational_explanation: string
    status: string
    created_at: string
  }
  onApprove: (gateId: string, comments?: string) => void
  onReject: (gateId: string, comments: string) => void
  onModify: (gateId: string, modifications: Record<string, any>, comments: string) => void
  disabled?: boolean
}

export default function ApprovalGate({ 
  gate, 
  onApprove, 
  onReject, 
  onModify, 
  disabled = false 
}: ApprovalGateProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showModifyForm, setShowModifyForm] = useState(false)
  const [comments, setComments] = useState('')
  const [modifications, setModifications] = useState<Record<string, any>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleApprove = async () => {
    if (isSubmitting) return
    setIsSubmitting(true)
    try {
      await onApprove(gate.gate_id, comments || undefined)
      toast.success('Proposal approved successfully!')
    } catch (error) {
      toast.error('Failed to approve proposal')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleReject = async () => {
    if (isSubmitting || !comments.trim()) {
      if (!comments.trim()) {
        toast.error('Please provide a reason for rejection')
      }
      return
    }
    setIsSubmitting(true)
    try {
      await onReject(gate.gate_id, comments)
      toast.success('Proposal rejected')
    } catch (error) {
      toast.error('Failed to reject proposal')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleModify = async () => {
    if (isSubmitting) return
    setIsSubmitting(true)
    try {
      await onModify(gate.gate_id, modifications, comments)
      toast.success('Proposal modified and approved!')
    } catch (error) {
      toast.error('Failed to modify proposal')
    } finally {
      setIsSubmitting(false)
    }
  }

  const getStatusIcon = () => {
    switch (gate.status) {
      case 'approved':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'rejected':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'modified':
        return <Edit3 className="w-5 h-5 text-blue-500" />
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />
    }
  }

  const getStatusColor = () => {
    switch (gate.status) {
      case 'approved':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'rejected':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'modified':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const formatProposalValue = (value: any): string => {
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2)
    }
    return String(value)
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <CardTitle className="text-xl">{gate.title}</CardTitle>
              <CardDescription className="text-sm text-gray-600">
                {gate.description}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={getStatusColor()}>
              {gate.status.toUpperCase()}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="space-y-6">
          {/* Educational Explanation */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-4 h-4 text-blue-600" />
              <h4 className="font-semibold text-blue-900">Educational Explanation</h4>
            </div>
            <div className="text-sm text-blue-800 whitespace-pre-line">
              {gate.educational_explanation}
            </div>
          </div>

          {/* Proposal Details */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Proposal Details</h4>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="grid gap-3">
                {Object.entries(gate.proposal).map(([key, value]) => (
                  <div key={key} className="flex flex-col sm:flex-row sm:items-center gap-2">
                    <span className="font-medium text-gray-700 min-w-0 sm:w-1/3">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                    </span>
                    <span className="text-sm text-gray-600 font-mono bg-white px-2 py-1 rounded border flex-1">
                      {formatProposalValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action Form */}
          {gate.status === 'pending' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Comments (Optional)
                </label>
                <Textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  placeholder="Add any comments or feedback..."
                  className="w-full"
                  rows={3}
                />
              </div>

              {showModifyForm && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <h5 className="font-semibold text-yellow-900 mb-2">Modify Proposal</h5>
                  <div className="space-y-3">
                    {Object.entries(gate.proposal).map(([key, value]) => (
                      <div key={key}>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </label>
                        <Textarea
                          value={modifications[key] || formatProposalValue(value)}
                          onChange={(e) => setModifications(prev => ({
                            ...prev,
                            [key]: e.target.value
                          }))}
                          className="w-full"
                          rows={2}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-3">
                <Button
                  onClick={handleApprove}
                  disabled={disabled || isSubmitting}
                  className="bg-green-600 hover:bg-green-700 text-white"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  {isSubmitting ? 'Approving...' : 'Approve'}
                </Button>

                <Button
                  onClick={handleReject}
                  disabled={disabled || isSubmitting}
                  variant="destructive"
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  {isSubmitting ? 'Rejecting...' : 'Reject'}
                </Button>

                <Button
                  onClick={() => setShowModifyForm(!showModifyForm)}
                  disabled={disabled || isSubmitting}
                  variant="outline"
                >
                  <Edit3 className="w-4 h-4 mr-2" />
                  {showModifyForm ? 'Cancel Modify' : 'Modify'}
                </Button>

                {showModifyForm && (
                  <Button
                    onClick={handleModify}
                    disabled={disabled || isSubmitting}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <Edit3 className="w-4 h-4 mr-2" />
                    {isSubmitting ? 'Modifying...' : 'Modify & Approve'}
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Gate Metadata */}
          <div className="text-xs text-gray-500 border-t pt-3">
            <div className="flex justify-between">
              <span>Gate ID: {gate.gate_id}</span>
              <span>Created: {new Date(gate.created_at).toLocaleString()}</span>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  )
}
