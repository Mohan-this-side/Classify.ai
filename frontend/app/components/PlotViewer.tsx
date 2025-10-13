'use client'

import React, { useState, useEffect } from 'react'
import { BarChart3, Eye, Download, ExternalLink, RefreshCw } from 'lucide-react'

interface Plot {
  id: string
  name: string
  path: string
  description?: string
  agent: string
  timestamp: Date
}

interface PlotViewerProps {
  plots: Plot[]
  isGenerating: boolean
  onRefresh?: () => void
}

export default function PlotViewer({ plots, isGenerating, onRefresh }: PlotViewerProps) {
  const [selectedPlot, setSelectedPlot] = useState<Plot | null>(null)
  const [plotImages, setPlotImages] = useState<Map<string, string>>(new Map())

  // Load plot images
  useEffect(() => {
    const loadPlotImages = async () => {
      const newImages = new Map<string, string>()
      
      for (const plot of plots) {
        try {
          // Convert backend path to frontend accessible URL
          const plotUrl = `http://localhost:8000/api/workflow/plot/${plot.path}`
          const response = await fetch(plotUrl)
          
          if (response.ok) {
            const blob = await response.blob()
            const imageUrl = URL.createObjectURL(blob)
            newImages.set(plot.id, imageUrl)
          }
        } catch (error) {
          console.error(`Failed to load plot ${plot.name}:`, error)
        }
      }
      
      setPlotImages(newImages)
    }

    if (plots.length > 0) {
      loadPlotImages()
    }
  }, [plots])

  const getAgentIcon = (agent: string) => {
    switch (agent) {
      case 'eda_analysis': return BarChart3
      case 'ml_building': return BarChart3
      case 'model_evaluation': return BarChart3
      default: return Eye
    }
  }

  const getAgentColor = (agent: string) => {
    switch (agent) {
      case 'eda_analysis': return 'text-yellow-400'
      case 'ml_building': return 'text-red-400'
      case 'model_evaluation': return 'text-indigo-400'
      default: return 'text-blue-400'
    }
  }

  const downloadPlot = async (plot: Plot) => {
    try {
      const plotUrl = `http://localhost:8000/api/workflow/plot/${plot.path}`
      const response = await fetch(plotUrl)
      
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${plot.name}.png`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Failed to download plot:', error)
    }
  }

  if (plots.length === 0 && !isGenerating) {
    return null
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-neon-500/20 rounded-lg">
            <BarChart3 className="w-6 h-6 text-neon-400" />
          </div>
          <h2 className="text-2xl font-semibold text-white">
            Generated Visualizations
          </h2>
          {isGenerating && (
            <div className="flex items-center gap-2 text-sm text-neon-400">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Generating...</span>
            </div>
          )}
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="btn-secondary text-sm"
            disabled={isGenerating}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
        )}
      </div>

      {/* Plot Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {plots.map((plot) => {
          const Icon = getAgentIcon(plot.agent)
          const imageUrl = plotImages.get(plot.id)
          
          return (
            <div
              key={plot.id}
              className="group relative bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden hover:border-neon-500/50 transition-all duration-300 cursor-pointer"
              onClick={() => setSelectedPlot(plot)}
            >
              {/* Plot Preview */}
              <div className="aspect-video bg-gray-900/50 flex items-center justify-center relative overflow-hidden">
                {imageUrl ? (
                  <img
                    src={imageUrl}
                    alt={plot.name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                ) : (
                  <div className="text-center text-gray-400">
                    <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Loading...</p>
                  </div>
                )}
                
                {/* Overlay */}
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <div className="flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedPlot(plot)
                      }}
                      className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
                    >
                      <Eye className="w-4 h-4 text-white" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        downloadPlot(plot)
                      }}
                      className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
                    >
                      <Download className="w-4 h-4 text-white" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Plot Info */}
              <div className="p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Icon className={`w-4 h-4 ${getAgentColor(plot.agent)}`} />
                  <span className="text-sm font-medium text-white truncate">
                    {plot.name}
                  </span>
                </div>
                {plot.description && (
                  <p className="text-xs text-gray-400 line-clamp-2">
                    {plot.description}
                  </p>
                )}
                <div className="flex items-center justify-between mt-2">
                  <span className="text-xs text-gray-500">
                    {plot.timestamp.toLocaleTimeString()}
                  </span>
                  <span className="text-xs text-gray-500 capitalize">
                    {plot.agent.replace('_', ' ')}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Plot Modal */}
      {selectedPlot && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <div className="flex items-center gap-3">
                {React.createElement(getAgentIcon(selectedPlot.agent), {
                  className: `w-5 h-5 ${getAgentColor(selectedPlot.agent)}`
                })}
                <h3 className="text-lg font-semibold text-white">
                  {selectedPlot.name}
                </h3>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => downloadPlot(selectedPlot)}
                  className="btn-secondary text-sm"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </button>
                <button
                  onClick={() => setSelectedPlot(null)}
                  className="btn-secondary text-sm"
                >
                  Close
                </button>
              </div>
            </div>
            
            <div className="p-4 max-h-[70vh] overflow-auto">
              {plotImages.get(selectedPlot.id) ? (
                <img
                  src={plotImages.get(selectedPlot.id)}
                  alt={selectedPlot.name}
                  className="w-full h-auto rounded-lg"
                />
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-400">
                  <div className="text-center">
                    <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Loading plot...</p>
                  </div>
                </div>
              )}
              
              {selectedPlot.description && (
                <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
                  <p className="text-sm text-gray-300">
                    {selectedPlot.description}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {plots.length === 0 && isGenerating && (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 bg-neon-500/10 rounded-full flex items-center justify-center">
            <RefreshCw className="w-8 h-8 text-neon-400 animate-spin" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">
            Generating Visualizations
          </h3>
          <p className="text-gray-400">
            Our agents are creating insightful plots and charts...
          </p>
        </div>
      )}
    </div>
  )
}
