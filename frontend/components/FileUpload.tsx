'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, X, AlertCircle, FileSpreadsheet, Database, Zap, CheckCircle, Lightbulb } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface FileUploadProps {
  onFileSelect: (file: File) => void
  onTargetColumnChange: (column: string) => void
  onDescriptionChange: (description: string) => void
  targetColumn: string
  description: string
  disabled?: boolean
}

export default function FileUpload({ 
  onFileSelect, 
  onTargetColumnChange, 
  onDescriptionChange, 
  targetColumn, 
  description, 
  disabled = false 
}: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [fileInfo, setFileInfo] = useState<{rows?: number, columns?: string[]} | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (disabled) return
    
    const droppedFile = acceptedFiles[0]
    if (droppedFile) {
      setFile(droppedFile)
      onFileSelect(droppedFile)
      
      // Simulate file analysis (in real app, you'd parse the file)
      setTimeout(() => {
        setFileInfo({
          rows: Math.floor(Math.random() * 10000) + 100,
          columns: ['id', 'feature_1', 'feature_2', 'target']
        })
      }, 500)
      
      toast.success(`📊 ${droppedFile.name} loaded successfully!`)
    }
  }, [onFileSelect, disabled])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
    disabled,
    onDragEnter: () => setIsDragOver(true),
    onDragLeave: () => setIsDragOver(false),
    onDropAccepted: () => setIsDragOver(false)
  })

  const removeFile = () => {
    setFile(null)
    setFileInfo(null)
    onFileSelect(null as any)
  }

  const getFileIcon = (fileName: string) => {
    if (fileName.endsWith('.csv')) return FileSpreadsheet
    if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) return Database
    return File
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      {/* File Upload Area */}
      <div>
        <label className="label mb-4 block">📂 Dataset File</label>
        <div
          {...getRootProps()}
          className={`file-upload-area transition-all duration-300 ${
            isDragActive || isDragOver ? 'border-neon-500 bg-neon-500/10 shadow-neon' : ''
          } ${file ? 'border-cyber-500/50 bg-cyber-500/5' : ''} ${
            disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
          }`}
        >
          <input {...getInputProps()} disabled={disabled} />
          
          {file ? (
            <div className="space-y-4">
              <div className="flex items-center justify-center gap-4">
                <div className="p-3 bg-cyber-500/20 rounded-xl">
                  {(() => {
                    const IconComponent = getFileIcon(file.name)
                    return <IconComponent className="w-8 h-8 text-cyber-400" />
                  })()}
                </div>
                <div className="flex-1 text-left">
                  <p className="font-medium text-white">{file.name}</p>
                  <p className="text-sm text-gray-400">{formatFileSize(file.size)}</p>
                  {fileInfo && (
                    <p className="text-xs text-cyber-400 mt-1">
                      ~{fileInfo.rows?.toLocaleString()} rows • {fileInfo.columns?.length} columns
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={removeFile}
                  className="p-2 text-gray-400 hover:text-red-400 transition-colors rounded-lg hover:bg-red-500/10"
                  disabled={disabled}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              {fileInfo && (
                <div className="bg-dark-800/50 rounded-lg p-4 border border-cyber-500/20">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-4 h-4 text-cyber-400" />
                    <span className="text-sm font-medium text-cyber-400">File Analysis Complete</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <span className="text-gray-500">Estimated Rows:</span>
                      <span className="ml-2 text-white font-mono">{fileInfo.rows?.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Columns:</span>
                      <span className="ml-2 text-white font-mono">{fileInfo.columns?.length}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="relative mb-6">
                <div className="w-16 h-16 mx-auto bg-neon-500/10 rounded-full flex items-center justify-center">
                  <Upload className={`w-8 h-8 transition-colors duration-300 ${
                    isDragActive ? 'text-neon-400' : 'text-gray-400'
                  }`} />
                </div>
                {(isDragActive || isDragOver) && (
                  <div className="absolute inset-0 bg-neon-500/20 rounded-full animate-ping" />
                )}
              </div>
              
              <div className="space-y-2">
                <p className={`text-lg font-medium transition-colors duration-300 ${
                  isDragActive ? 'text-neon-400' : 'text-white'
                }`}>
                  {isDragActive ? '⚡ Drop your dataset here!' : '🎯 Upload Your Dataset'}
                </p>
                <p className="text-sm text-gray-400">
                  Drag & drop or click to browse • CSV, Excel files up to 100MB
                </p>
              </div>
              
              <div className="flex items-center justify-center gap-4 mt-6 text-xs text-gray-500">
                <div className="flex items-center gap-1">
                  <FileSpreadsheet className="w-3 h-3" />
                  <span>CSV</span>
                </div>
                <div className="flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  <span>Excel</span>
                </div>
                <div className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  <span>Auto-Analysis</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Target Column */}
      <div>
        <label htmlFor="targetColumn" className="label">
          🎯 Target Column
        </label>
        <input
          type="text"
          id="targetColumn"
          value={targetColumn}
          onChange={(e) => onTargetColumnChange(e.target.value)}
          placeholder="e.g., species, category, target, label"
          className="input"
          disabled={disabled}
        />
        <p className="mt-2 text-sm text-gray-400 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span>The column you want to predict/classify</span>
        </p>
      </div>

      {/* Description */}
      <div>
        <label htmlFor="description" className="label">
          📝 Task Description
        </label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder="Describe your machine learning task in detail. For example: 'This dataset contains customer transaction data. I want to predict which customers are likely to churn based on their purchase history, demographics, and engagement metrics.'"
          className="input h-32 resize-none"
          disabled={disabled}
        />
        <p className="mt-2 text-sm text-gray-400 flex items-center gap-2">
          <Lightbulb className="w-4 h-4" />
          <span>Help our AI agents understand your data and goals</span>
        </p>
      </div>

      {/* Features Preview */}
      {file && (
        <div className="bg-dark-800/30 rounded-2xl p-6 border border-neon-500/20">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-electric-400" />
            <h3 className="text-lg font-semibold text-white">What Our AI Agents Will Do</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { icon: '🧹', title: 'Data Cleaning', desc: 'Handle missing values, outliers, and inconsistencies' },
              { icon: '🔍', title: 'Data Discovery', desc: 'Research domain knowledge and similar datasets' },
              { icon: '📊', title: 'EDA Analysis', desc: 'Generate comprehensive statistical insights' },
              { icon: '⚙️', title: 'Feature Engineering', desc: 'Create and select optimal features automatically' },
              { icon: '🤖', title: 'Model Training', desc: 'Train and optimize multiple ML algorithms' },
              { icon: '📈', title: 'Evaluation', desc: 'Assess performance with detailed metrics' }
            ].map((step, index) => (
              <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-dark-900/50">
                <span className="text-lg">{step.icon}</span>
                <div>
                  <p className="font-medium text-white text-sm">{step.title}</p>
                  <p className="text-xs text-gray-400">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
