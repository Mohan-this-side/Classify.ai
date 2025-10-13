'use client'

import { useState } from 'react'
import { 
  Download, FileText, Brain, BarChart3, Database, Eye, Code, FileSpreadsheet,
  TrendingUp, Zap, CheckCircle, AlertCircle, Star, Award, Target,
  Cpu, Activity, PieChart, LineChart, DownloadCloud
} from 'lucide-react'

interface ResultsViewerProps {
  results: any
  onDownload: (type: string) => void
}

export default function ResultsViewer({ results, onDownload }: ResultsViewerProps) {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', name: 'Overview', icon: Eye, count: null },
    { id: 'metrics', name: 'Performance', icon: TrendingUp, count: null },
    { id: 'model', name: 'Model', icon: Brain, count: null },
    { id: 'data', name: 'Data', icon: Database, count: null },
    { id: 'downloads', name: 'Downloads', icon: DownloadCloud, count: 4 }
  ]

  // Mock data - in real app this would come from results prop
  const mockResults = {
    model: {
      name: 'Random Forest Classifier',
      accuracy: 0.942,
      f1_score: 0.92,
      precision: 0.91,
      recall: 0.93,
      parameters: {
        n_estimators: 100,
        max_depth: 10,
        min_samples_split: 5
      }
    },
    dataset: {
      original_rows: 1500,
      cleaned_rows: 1485,
      features: 8,
      target_classes: 3
    },
    timing: {
      total: '4m 32s',
      cleaning: '45s',
      training: '2m 15s',
      evaluation: '1m 12s'
    },
    features: [
      { name: 'petal_length', importance: 0.35 },
      { name: 'petal_width', importance: 0.28 },
      { name: 'sepal_length', importance: 0.22 },
      { name: 'sepal_width', importance: 0.15 }
    ]
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Classification Results</h2>
            <p className="text-gray-600">Your model has been trained and evaluated successfully</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={() => onDownload('model')}
              className="btn-outline flex items-center space-x-2"
            >
              <Brain className="w-4 h-4" />
              <span>Download Model</span>
            </button>
            <button
              onClick={() => onDownload('report')}
              className="btn-primary flex items-center space-x-2"
            >
              <FileText className="w-4 h-4" />
              <span>Download Report</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const IconComponent = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <IconComponent className="w-4 h-4" />
                <span>{tab.name}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Accuracy</span>
                  <span className="font-semibold text-success-600">94.2%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">F1 Score</span>
                  <span className="font-semibold text-success-600">0.92</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Precision</span>
                  <span className="font-semibold text-success-600">0.91</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Recall</span>
                  <span className="font-semibold text-success-600">0.93</span>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Info</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Original Rows</span>
                  <span className="font-semibold">1,500</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Cleaned Rows</span>
                  <span className="font-semibold">1,485</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Features</span>
                  <span className="font-semibold">8</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Target Classes</span>
                  <span className="font-semibold">3</span>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Processing Time</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Time</span>
                  <span className="font-semibold">4m 32s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Data Cleaning</span>
                  <span className="font-semibold">45s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Training</span>
                  <span className="font-semibold">2m 15s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Evaluation</span>
                  <span className="font-semibold">1m 12s</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Confusion Matrix</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-center text-sm text-gray-600">
                  Confusion Matrix visualization would be displayed here
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">ROC Curve</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-center text-sm text-gray-600">
                    ROC Curve visualization would be displayed here
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Precision-Recall Curve</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-center text-sm text-gray-600">
                    Precision-Recall Curve visualization would be displayed here
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'model' && (
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Model Type</h4>
                  <p className="text-gray-600">Random Forest Classifier</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Best Parameters</h4>
                  <div className="text-sm text-gray-600">
                    <p>n_estimators: 100</p>
                    <p>max_depth: 10</p>
                    <p>min_samples_split: 5</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
              <div className="space-y-2">
                {[
                  { name: 'petal_length', importance: 0.35 },
                  { name: 'petal_width', importance: 0.28 },
                  { name: 'sepal_length', importance: 0.22 },
                  { name: 'sepal_width', importance: 0.15 }
                ].map((feature, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <span className="text-sm font-medium text-gray-700 w-24">{feature.name}</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full"
                        style={{ width: `${feature.importance * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 w-12">{(feature.importance * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'data' && (
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Quality Report</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-success-50 rounded-lg">
                  <span className="text-sm font-medium text-success-800">Missing Values</span>
                  <span className="text-sm text-success-600">0 (0%)</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-success-50 rounded-lg">
                  <span className="text-sm font-medium text-success-800">Duplicates</span>
                  <span className="text-sm text-success-600">0 (0%)</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-success-50 rounded-lg">
                  <span className="text-sm font-medium text-success-800">Data Types</span>
                  <span className="text-sm text-success-600">All optimized</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Distribution</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-center text-sm text-gray-600">
                    Data distribution plots would be displayed here
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Correlation Matrix</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-center text-sm text-gray-600">
                    Correlation matrix heatmap would be displayed here
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reports' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="card">
                <div className="flex items-center space-x-3 mb-4">
                  <FileText className="w-8 h-8 text-primary-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Executive Summary</h3>
                    <p className="text-sm text-gray-600">High-level overview</p>
                  </div>
                </div>
                <button
                  onClick={() => onDownload('executive_summary')}
                  className="btn-outline w-full"
                >
                  Download PDF
                </button>
              </div>

              <div className="card">
                <div className="flex items-center space-x-3 mb-4">
                  <BarChart3 className="w-8 h-8 text-success-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Technical Report</h3>
                    <p className="text-sm text-gray-600">Detailed analysis</p>
                  </div>
                </div>
                <button
                  onClick={() => onDownload('technical_report')}
                  className="btn-outline w-full"
                >
                  Download PDF
                </button>
              </div>

              <div className="card">
                <div className="flex items-center space-x-3 mb-4">
                  <Database className="w-8 h-8 text-warning-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Data Report</h3>
                    <p className="text-sm text-gray-600">Data quality analysis</p>
                  </div>
                </div>
                <button
                  onClick={() => onDownload('data_report')}
                  className="btn-outline w-full"
                >
                  Download PDF
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="card">
                <div className="flex items-center space-x-3 mb-4">
                  <Code className="w-8 h-8 text-primary-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Jupyter Notebook</h3>
                    <p className="text-sm text-gray-600">Complete analysis notebook</p>
                  </div>
                </div>
                <button
                  onClick={() => onDownload('notebook')}
                  className="btn-outline w-full"
                >
                  Download .ipynb
                </button>
              </div>

              <div className="card">
                <div className="flex items-center space-x-3 mb-4">
                  <FileSpreadsheet className="w-8 h-8 text-success-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">Cleaned Dataset</h3>
                    <p className="text-sm text-gray-600">Preprocessed data</p>
                  </div>
                </div>
                <button
                  onClick={() => onDownload('dataset')}
                  className="btn-outline w-full"
                >
                  Download CSV
                </button>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Generated Code</h3>
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm text-gray-100">
{`# Data Cleaning Code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and clean data
df = pd.read_csv('dataset.csv')
df = df.dropna()
df = df.drop_duplicates()

# Feature engineering
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))`}
                </pre>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'downloads' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="card text-center">
                <div className="flex justify-center mb-4">
                  <Brain className="w-12 h-12 text-primary-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Trained Model</h3>
                <p className="text-sm text-gray-600 mb-4">Random Forest Classifier (.pkl)</p>
                <button
                  onClick={() => onDownload('model')}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
              </div>

              <div className="card text-center">
                <div className="flex justify-center mb-4">
                  <FileText className="w-12 h-12 text-success-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Full Report</h3>
                <p className="text-sm text-gray-600 mb-4">Comprehensive analysis (.pdf)</p>
                <button
                  onClick={() => onDownload('report')}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
              </div>

              <div className="card text-center">
                <div className="flex justify-center mb-4">
                  <Code className="w-12 h-12 text-warning-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Jupyter Notebook</h3>
                <p className="text-sm text-gray-600 mb-4">Complete analysis (.ipynb)</p>
                <button
                  onClick={() => onDownload('notebook')}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
              </div>

              <div className="card text-center">
                <div className="flex justify-center mb-4">
                  <Database className="w-12 h-12 text-gray-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Cleaned Data</h3>
                <p className="text-sm text-gray-600 mb-4">Processed dataset (.csv)</p>
                <button
                  onClick={() => onDownload('dataset')}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Download All</h3>
              <p className="text-gray-600 mb-4">
                Get all generated files in a single zip archive including the trained model, reports, notebook, and cleaned dataset.
              </p>
              <button
                onClick={() => onDownload('all')}
                className="btn-primary flex items-center space-x-2"
              >
                <DownloadCloud className="w-5 h-5" />
                <span>Download All Files (.zip)</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
