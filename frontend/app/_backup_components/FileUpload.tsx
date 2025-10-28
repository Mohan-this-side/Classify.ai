'use client'

import React, { useCallback, useState } from 'react'
import { Upload, FileText, X } from 'lucide-react'

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
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (disabled) return
    
    const files = e.dataTransfer.files
    if (files && files[0]) {
      const file = files[0]
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file)
        onFileSelect(file)
      }
    }
  }, [disabled, onFileSelect])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    
    const files = e.target.files
    if (files && files[0]) {
      const file = files[0]
      setSelectedFile(file)
      onFileSelect(file)
    }
  }, [disabled, onFileSelect])

  const removeFile = useCallback(() => {
    setSelectedFile(null)
  }, [])

  return (
    <div className="space-y-6">
      {/* File Upload Area */}
      <div
        className={`file-upload-area ${dragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileInput}
          disabled={disabled}
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload" className="cursor-pointer block w-full h-full">
          <div className="text-center">
            <Upload className="w-12 h-12 mx-auto mb-4 text-neon-400" />
            <h3 className="text-xl font-semibold mb-2 text-night-100">
              {selectedFile ? 'File Selected' : 'Upload Dataset'}
            </h3>
            <p className="text-night-300 mb-4">
              {selectedFile 
                ? selectedFile.name 
                : 'Drag and drop your CSV file here, or click to browse'
              }
            </p>
            {selectedFile && (
              <button
                type="button"
                onClick={removeFile}
                className="btn-secondary text-sm"
              >
                <X className="w-4 h-4 mr-2" />
                Remove File
              </button>
            )}
          </div>
        </label>
      </div>

      {/* Configuration Form */}
      <div className="space-y-4">
        <div>
          <label htmlFor="target-column" className="label">
            Target Column
          </label>
          <input
            id="target-column"
            type="text"
            value={targetColumn}
            onChange={(e) => onTargetColumnChange(e.target.value)}
            placeholder="e.g., species, category, label"
            className="input"
            disabled={disabled}
          />
        </div>

        <div>
          <label htmlFor="description" className="label">
            Project Description
          </label>
          <textarea
            id="description"
            value={description}
            onChange={(e) => onDescriptionChange(e.target.value)}
            placeholder="Describe your classification task..."
            className="input min-h-[100px] resize-none"
            disabled={disabled}
          />
        </div>
      </div>

      {/* File Info */}
      {selectedFile && (
        <div className="card">
          <div className="flex items-center space-x-3">
            <FileText className="w-8 h-8 text-neon-400" />
            <div>
              <h4 className="font-medium text-night-100">{selectedFile.name}</h4>
              <p className="text-sm text-night-300">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
