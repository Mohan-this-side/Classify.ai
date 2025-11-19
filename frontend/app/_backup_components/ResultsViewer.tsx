'use client'

import React, { useState } from 'react'
import { 
  Download, 
  FileText, 
  Code, 
  BarChart3, 
  Database, 
  Eye,
  ChevronDown,
  ChevronRight
} from 'lucide-react'

interface ResultsViewerProps {
  results: {
    workflowId: string
    status: string
    datasetInfo?: {
      filename: string
      shape: [number, number]
      targetColumn: string
      columns: string[]
    }
    modelResults?: {
      bestModel: string
      accuracy: number
      precision: number
      recall: number
      f1Score: number
    }
    artifacts?: {
      modelPath?: string
      notebookPath?: string
      reportPath?: string
    }
    executionTime?: number
  }
  onDownload: (type: string) => void
}

export default function ResultsViewer({ results, onDownload }: ResultsViewerProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']))

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(section)) {
      newExpanded.delete(section)
    } else {
      newExpanded.add(section)
    }
    setExpandedSections(newExpanded)
  }

  const SectionHeader = ({ 
    title, 
    icon: Icon, 
    section, 
    badge 
  }: { 
    title: string
    icon: any
    section: string
    badge?: string
  }) => (
    <button
      onClick={() => toggleSection(section)}
      className="w-full flex items-center justify-between p-4 text-left hover:bg-night-800/50 transition-colors"
    >
      <div className="flex items-center space-x-3">
        <Icon className="w-5 h-5 text-neon-400" />
        <span className="font-medium text-night-100">{title}</span>
        {badge && (
          <span className="badge bg-neon-900/30 text-neon-300">
            {badge}
          </span>
        )}
      </div>
      {expandedSections.has(section) ? (
        <ChevronDown className="w-5 h-5 text-night-400" />
      ) : (
        <ChevronRight className="w-5 h-5 text-night-400" />
      )}
    </button>
  )

  return (
    <div className="space-y-4">
      {/* Summary Section */}
      <div className="card">
        <SectionHeader
          title="Workflow Summary"
          icon={BarChart3}
          section="summary"
          badge={results.status}
        />
        {expandedSections.has('summary') && (
          <div className="p-4 pt-0 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-neon-400">
                  {results.datasetInfo?.shape[0] || 'N/A'}
                </div>
                <div className="text-sm text-night-300">Rows</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neon-400">
                  {results.datasetInfo?.shape[1] || 'N/A'}
                </div>
                <div className="text-sm text-night-300">Columns</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {results.modelResults?.accuracy ? `${(results.modelResults.accuracy * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div className="text-sm text-night-300">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neon-400">
                  {results.executionTime ? `${results.executionTime.toFixed(1)}s` : 'N/A'}
                </div>
                <div className="text-sm text-night-300">Duration</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Information */}
      {results.datasetInfo && (
        <div className="card">
          <SectionHeader
            title="Dataset Information"
            icon={Database}
            section="dataset"
          />
          {expandedSections.has('dataset') && (
            <div className="p-4 pt-0 space-y-3">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-night-300">Filename</label>
                  <div className="text-night-100 font-medium">{results.datasetInfo.filename}</div>
                </div>
                <div>
                  <label className="text-sm text-night-300">Target Column</label>
                  <div className="text-night-100 font-medium">{results.datasetInfo.targetColumn}</div>
                </div>
              </div>
              <div>
                <label className="text-sm text-night-300">Columns</label>
                <div className="flex flex-wrap gap-2 mt-1">
                  {results.datasetInfo.columns.map((col, index) => (
                    <span key={index} className="badge bg-night-700 text-night-200">
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Results */}
      {results.modelResults && (
        <div className="card">
          <SectionHeader
            title="Model Performance"
            icon={BarChart3}
            section="model"
            badge={results.modelResults.bestModel}
          />
          {expandedSections.has('model') && (
            <div className="p-4 pt-0 space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-xl font-bold text-green-400">
                    {(results.modelResults.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-night-300">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-neon-400">
                    {(results.modelResults.precision * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-night-300">Precision</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-neon-400">
                    {(results.modelResults.recall * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-night-300">Recall</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-neon-400">
                    {(results.modelResults.f1Score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-night-300">F1 Score</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Download Section */}
      <div className="card">
        <SectionHeader
          title="Download Results"
          icon={Download}
          section="downloads"
        />
        {expandedSections.has('downloads') && (
          <div className="p-4 pt-0">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => onDownload('model')}
                className="btn-primary flex items-center justify-center space-x-2"
                disabled={!results.artifacts?.modelPath}
              >
                <Database className="w-4 h-4" />
                <span>Download Model</span>
              </button>
              
              <button
                onClick={() => onDownload('notebook')}
                className="btn-primary flex items-center justify-center space-x-2"
                disabled={!results.artifacts?.notebookPath}
              >
                <FileText className="w-4 h-4" />
                <span>Download Notebook</span>
              </button>
              
              <button
                onClick={() => onDownload('report')}
                className="btn-primary flex items-center justify-center space-x-2"
                disabled={!results.artifacts?.reportPath}
              >
                <Code className="w-4 h-4" />
                <span>Download Report</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
