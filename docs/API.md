# API Documentation

This document provides comprehensive API documentation for the DS Capstone Multi-Agent Classification System.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The API is built with FastAPI and provides RESTful endpoints for:

- Workflow management
- File upload and processing
- Real-time progress tracking
- Result retrieval
- Approval gate management

**Base URL**: `http://localhost:8000` (development)  
**API Version**: v1  
**Content Type**: `application/json`

## Authentication

Currently, the API does not require authentication for development. In production, implement proper authentication using JWT tokens or API keys.

## Endpoints

### Workflow Management

#### Start Workflow

**POST** `/api/workflow/start`

Start a new classification workflow.

**Request Body**:
```json
{
  "file": "multipart/form-data",
  "target_column": "string",
  "user_id": "string (optional)"
}
```

**Response**:
```json
{
  "workflow_id": "string",
  "status": "started",
  "message": "Workflow started successfully"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/workflow/start" \
  -F "file=@dataset.csv" \
  -F "target_column=target" \
  -F "user_id=user123"
```

#### Get Workflow Status

**GET** `/api/workflow/status/{workflow_id}`

Get the current status of a workflow.

**Response**:
```json
{
  "workflow_id": "string",
  "status": "running|completed|failed|paused",
  "progress": 75.5,
  "current_phase": "Model Training",
  "agent_status": {
    "data_cleaning": "completed",
    "data_discovery": "completed",
    "eda_analysis": "running",
    "feature_engineering": "pending",
    "ml_building": "pending",
    "model_evaluation": "pending",
    "technical_reporter": "pending",
    "project_manager": "completed"
  },
  "completed_agents": ["data_cleaning", "data_discovery"],
  "errors": [],
  "estimated_completion": "2024-01-15T10:30:00Z"
}
```

#### Get Workflow Results

**GET** `/api/workflow/results/{workflow_id}`

Get the results of a completed workflow.

**Response**:
```json
{
  "workflow_id": "string",
  "status": "completed",
  "results": {
    "cleaned_dataset": {
      "path": "results/workflow_id/cleaned_dataset.csv",
      "shape": [1000, 5],
      "quality_score": 0.95
    },
    "model": {
      "path": "results/workflow_id/model.joblib",
      "type": "RandomForestClassifier",
      "accuracy": 0.92,
      "precision": 0.91,
      "recall": 0.89,
      "f1_score": 0.90
    },
    "notebook": {
      "path": "results/workflow_id/notebook.ipynb",
      "size": "2.5MB"
    },
    "report": {
      "path": "results/workflow_id/report.md",
      "size": "1.2MB"
    },
    "plots": [
      "results/workflow_id/plots/confusion_matrix.png",
      "results/workflow_id/plots/roc_curve.png",
      "results/workflow_id/plots/feature_importance.png"
    ]
  },
  "metadata": {
    "execution_time": "00:05:23",
    "dataset_info": {
      "original_shape": [1000, 6],
      "cleaned_shape": [1000, 5],
      "missing_values_handled": 45,
      "outliers_removed": 12
    },
    "model_info": {
      "best_model": "RandomForestClassifier",
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
      },
      "cross_validation_score": 0.89
    }
  }
}
```

#### List Workflows

**GET** `/api/workflow/list`

List all workflows (with optional filtering).

**Query Parameters**:
- `user_id` (optional): Filter by user ID
- `status` (optional): Filter by status
- `limit` (optional): Maximum number of results (default: 50)
- `offset` (optional): Number of results to skip (default: 0)

**Response**:
```json
{
  "workflows": [
    {
      "workflow_id": "string",
      "user_id": "string",
      "status": "completed",
      "created_at": "2024-01-15T09:00:00Z",
      "completed_at": "2024-01-15T09:05:23Z",
      "dataset_name": "dataset.csv",
      "target_column": "target"
    }
  ],
  "total": 25,
  "limit": 50,
  "offset": 0
}
```

### File Management

#### Download File

**GET** `/api/workflow/download/{workflow_id}/{file_type}`

Download a specific file from a workflow.

**Path Parameters**:
- `workflow_id`: Workflow identifier
- `file_type`: Type of file (`cleaned_dataset`, `model`, `notebook`, `report`, `plots`)

**Response**: File download

**Example**:
```bash
curl -O "http://localhost:8000/api/workflow/download/workflow123/cleaned_dataset"
curl -O "http://localhost:8000/api/workflow/download/workflow123/model"
curl -O "http://localhost:8000/api/workflow/download/workflow123/notebook"
curl -O "http://localhost:8000/api/workflow/download/workflow123/report"
curl -O "http://localhost:8000/api/workflow/download/workflow123/plots"
```

#### Get Plot

**GET** `/api/workflow/plot/{workflow_id}/{plot_name}`

Get a specific plot from a workflow.

**Path Parameters**:
- `workflow_id`: Workflow identifier
- `plot_name`: Name of the plot file

**Response**: Image file (PNG)

### Approval Gates

#### Get Approval Gates

**GET** `/api/workflow/approval-gates/{workflow_id}`

Get all approval gates for a workflow.

**Response**:
```json
{
  "workflow_id": "string",
  "gates": [
    {
      "gate_id": "string",
      "gate_type": "data_cleaning_approval",
      "title": "Data Cleaning Approval",
      "description": "Review and approve data cleaning actions",
      "proposal": {
        "actions_taken": [
          "Removed 45 missing values",
          "Standardized categorical values",
          "Detected and handled 12 outliers"
        ],
        "quality_improvement": 0.15
      },
      "educational_explanation": "Data cleaning is crucial for model accuracy...",
      "status": "pending",
      "created_at": "2024-01-15T09:02:00Z"
    }
  ]
}
```

#### Approve Gate

**POST** `/api/workflow/approval-gates/{gate_id}/approve`

Approve an approval gate.

**Request Body**:
```json
{
  "user_comments": "string (optional)",
  "modifications": {
    "key": "value"
  }
}
```

**Response**:
```json
{
  "gate_id": "string",
  "status": "approved",
  "approved_at": "2024-01-15T09:05:00Z",
  "user_comments": "string",
  "workflow_resumed": true
}
```

#### Reject Gate

**POST** `/api/workflow/approval-gates/{gate_id}/reject`

Reject an approval gate.

**Request Body**:
```json
{
  "reason": "string",
  "suggestions": "string (optional)"
}
```

**Response**:
```json
{
  "gate_id": "string",
  "status": "rejected",
  "rejected_at": "2024-01-15T09:05:00Z",
  "reason": "string",
  "workflow_paused": true
}
```

### Health and Status

#### Health Check

**GET** `/health`

Check the health of the API service.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T09:00:00Z",
  "version": "1.0.0",
  "uptime": "2d 5h 30m"
}
```

#### Database Health

**GET** `/health/database`

Check database connectivity.

**Response**:
```json
{
  "status": "healthy",
  "database": "postgresql",
  "connection_time": "0.05s"
}
```

#### Redis Health

**GET** `/health/redis`

Check Redis connectivity.

**Response**:
```json
{
  "status": "healthy",
  "redis": "connected",
  "ping_time": "0.01s"
}
```

## WebSocket API

### Connection

**WebSocket** `/ws/{workflow_id}`

Connect to real-time updates for a specific workflow.

**Connection URL**: `ws://localhost:8000/ws/{workflow_id}`

### Message Types

#### Workflow Update

```json
{
  "type": "workflow_update",
  "data": {
    "workflow_id": "string",
    "status": "running",
    "progress": 45.5,
    "current_phase": "Data Analysis"
  }
}
```

#### Agent Started

```json
{
  "type": "agent_started",
  "agent": "data_cleaning",
  "message": "Starting data cleaning process...",
  "timestamp": "2024-01-15T09:00:00Z"
}
```

#### Agent Completed

```json
{
  "type": "agent_completed",
  "agent": "data_cleaning",
  "message": "Data cleaning completed successfully",
  "results": {
    "quality_score": 0.95,
    "issues_found": 12,
    "actions_taken": ["missing_value_imputation", "outlier_removal"]
  },
  "timestamp": "2024-01-15T09:02:30Z"
}
```

#### Agent Failed

```json
{
  "type": "agent_failed",
  "agent": "data_cleaning",
  "error": "Invalid data format detected",
  "timestamp": "2024-01-15T09:02:30Z"
}
```

#### Approval Gate Triggered

```json
{
  "type": "approval_gate_triggered",
  "gate_id": "string",
  "gate_type": "data_cleaning_approval",
  "title": "Data Cleaning Approval",
  "description": "Review and approve data cleaning actions",
  "timestamp": "2024-01-15T09:02:30Z"
}
```

#### Workflow Completed

```json
{
  "type": "workflow_completed",
  "workflow_id": "string",
  "results": {
    "model_accuracy": 0.92,
    "execution_time": "00:05:23"
  },
  "timestamp": "2024-01-15T09:05:23Z"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "string",
  "error_id": "string",
  "error_type": "string",
  "severity": "low|medium|high|critical",
  "category": "validation|network|timeout|resource|data|model|agent|workflow|unknown",
  "timestamp": "2024-01-15T09:00:00Z",
  "retryable": true,
  "details": {
    "field": "additional error details"
  }
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Types

#### Validation Errors

```json
{
  "error": "Invalid target column specified",
  "error_type": "ValueError",
  "severity": "medium",
  "category": "validation",
  "retryable": false
}
```

#### Network Errors

```json
{
  "error": "Database connection timeout",
  "error_type": "ConnectionError",
  "severity": "high",
  "category": "network",
  "retryable": true
}
```

#### Resource Errors

```json
{
  "error": "Insufficient memory for model training",
  "error_type": "MemoryError",
  "severity": "critical",
  "category": "resource",
  "retryable": false
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Workflow Creation**: 10 requests per minute per IP
- **File Upload**: 5 requests per minute per IP
- **General API**: 100 requests per minute per IP

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Examples

### Complete Workflow Example

```python
import requests
import websocket
import json
import time

# 1. Start workflow
def start_workflow(file_path, target_column):
    url = "http://localhost:8000/api/workflow/start"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'target_column': target_column}
        response = requests.post(url, files=files, data=data)
    
    return response.json()

# 2. Monitor progress via WebSocket
def monitor_workflow(workflow_id):
    def on_message(ws, message):
        data = json.loads(message)
        print(f"Update: {data}")
        
        if data['type'] == 'workflow_completed':
            print("Workflow completed!")
            ws.close()
    
    ws_url = f"ws://localhost:8000/ws/{workflow_id}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()

# 3. Get results
def get_results(workflow_id):
    url = f"http://localhost:8000/api/workflow/results/{workflow_id}"
    response = requests.get(url)
    return response.json()

# 4. Download files
def download_file(workflow_id, file_type):
    url = f"http://localhost:8000/api/workflow/download/{workflow_id}/{file_type}"
    response = requests.get(url)
    
    with open(f"{file_type}_{workflow_id}.csv", 'wb') as f:
        f.write(response.content)

# Usage
workflow = start_workflow("dataset.csv", "target")
workflow_id = workflow["workflow_id"]

# Monitor in background
import threading
monitor_thread = threading.Thread(target=monitor_workflow, args=(workflow_id,))
monitor_thread.start()

# Wait for completion
time.sleep(300)  # 5 minutes

# Get results
results = get_results(workflow_id)
print(f"Model accuracy: {results['results']['model']['accuracy']}")

# Download files
download_file(workflow_id, "cleaned_dataset")
download_file(workflow_id, "model")
download_file(workflow_id, "notebook")
download_file(workflow_id, "report")
```

### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class WorkflowClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async startWorkflow(filePath, targetColumn) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));
        form.append('target_column', targetColumn);
        
        const response = await axios.post(`${this.baseUrl}/api/workflow/start`, form, {
            headers: form.getHeaders()
        });
        
        return response.data;
    }
    
    monitorWorkflow(workflowId, onUpdate) {
        const ws = new WebSocket(`ws://localhost:8000/ws/${workflowId}`);
        
        ws.on('message', (data) => {
            const update = JSON.parse(data);
            onUpdate(update);
        });
        
        return ws;
    }
    
    async getResults(workflowId) {
        const response = await axios.get(`${this.baseUrl}/api/workflow/results/${workflowId}`);
        return response.data;
    }
    
    async downloadFile(workflowId, fileType) {
        const response = await axios.get(`${this.baseUrl}/api/workflow/download/${workflowId}/${fileType}`, {
            responseType: 'stream'
        });
        
        return response.data;
    }
}

// Usage
const client = new WorkflowClient();

async function runWorkflow() {
    try {
        // Start workflow
        const workflow = await client.startWorkflow('./dataset.csv', 'target');
        console.log('Workflow started:', workflow.workflow_id);
        
        // Monitor progress
        const ws = client.monitorWorkflow(workflow.workflow_id, (update) => {
            console.log('Update:', update);
        });
        
        // Wait for completion (in real app, handle this properly)
        setTimeout(async () => {
            const results = await client.getResults(workflow.workflow_id);
            console.log('Results:', results);
            ws.close();
        }, 300000); // 5 minutes
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

runWorkflow();
```

## SDKs and Libraries

### Python SDK

```python
from ds_capstone_sdk import WorkflowClient

client = WorkflowClient(api_key="your_api_key")

# Start workflow
workflow = client.start_workflow("dataset.csv", "target")

# Monitor progress
for update in client.monitor_workflow(workflow.workflow_id):
    print(update)

# Get results
results = client.get_results(workflow.workflow_id)
```

### JavaScript SDK

```javascript
import { WorkflowClient } from '@ds-capstone/sdk';

const client = new WorkflowClient({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.example.com'
});

// Start workflow
const workflow = await client.startWorkflow('dataset.csv', 'target');

// Monitor progress
client.monitorWorkflow(workflow.workflowId, (update) => {
    console.log(update);
});

// Get results
const results = await client.getResults(workflow.workflowId);
```

## Support

For API support:

1. **Documentation**: Check this file and the main README
2. **Interactive API**: Visit `/docs` for Swagger UI
3. **Issues**: Create an issue in the repository
4. **Email**: Contact the development team

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- Workflow management endpoints
- Real-time WebSocket updates
- File download functionality
- Approval gate system
