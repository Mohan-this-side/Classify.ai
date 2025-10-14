"""
Simple test backend to verify basic functionality
"""

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid
import json
import asyncio
from typing import Dict, Any, Optional

app = FastAPI(title="Test Backend")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Test backend is running"}

@app.get("/")
async def root():
    return {"message": "Test backend is working"}

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, session_id)
    except WebSocketDisconnect:
        manager.disconnect(session_id)

# Mock workflow endpoints for testing
@app.post("/api/workflow/start")
async def start_workflow(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    description: str = Form(...),
    api_key: str = Form("dummy_key"),
    user_id: Optional[str] = Form("web_user")
) -> Dict[str, Any]:
    """Mock workflow start endpoint"""
    workflow_id = str(uuid.uuid4())
    
    # Simulate file processing
    content = await file.read()
    filename = file.filename
    
    # Start background task to simulate workflow progress
    background_tasks.add_task(simulate_workflow_progress, workflow_id)
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Workflow started successfully",
        "dataset_info": {
            "filename": filename,
            "size": len(content),
            "target_column": target_column,
            "description": description
        }
    }

async def simulate_workflow_progress(workflow_id: str):
    """Simulate workflow progress with WebSocket events"""
    agents = [
        "data_cleaning", "data_discovery", "eda_analysis", 
        "feature_engineering", "ml_building", "model_evaluation", 
        "technical_reporter", "project_manager"
    ]
    
    for i, agent in enumerate(agents):
        # Emit agent started event
        await manager.send_personal_message({
            "type": "agent_started",
            "agent": agent,
            "workflow_id": workflow_id,
            "message": f"Starting {agent} agent...",
            "progress": (i / len(agents)) * 100
        }, workflow_id)
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Emit agent completed event
        await manager.send_personal_message({
            "type": "agent_completed",
            "agent": agent,
            "workflow_id": workflow_id,
            "message": f"Completed {agent} agent",
            "progress": ((i + 1) / len(agents)) * 100
        }, workflow_id)
    
    # Emit workflow completed event
    await manager.send_personal_message({
        "type": "workflow_completed",
        "workflow_id": workflow_id,
        "message": "Workflow completed successfully",
        "progress": 100
    }, workflow_id)

@app.get("/api/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Mock workflow status endpoint"""
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "progress": 100.0,
        "current_phase": "completed",
        "message": "Workflow completed successfully"
    }

@app.get("/api/workflow/results/{workflow_id}")
async def get_workflow_results(workflow_id: str) -> Dict[str, Any]:
    """Mock workflow results endpoint"""
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "results": {
            "model_accuracy": 0.85,
            "plots": ["plot1.png", "plot2.png"],
            "notebook": "analysis.ipynb",
            "report": "report.pdf"
        }
    }

@app.get("/api/workflow/plot/{plot_path}")
async def get_plot_image(plot_path: str):
    """Serve plot images"""
    # For now, return a placeholder response
    # In a real implementation, this would serve the actual plot file
    return JSONResponse({
        "message": f"Plot {plot_path} would be served here",
        "plot_path": plot_path
    })

@app.get("/api/workflow/download/{workflow_id}/{file_type}")
async def download_workflow_file(workflow_id: str, file_type: str):
    """Download workflow files (model, notebook, report)"""
    # For now, return a placeholder response
    # In a real implementation, this would serve the actual file
    return JSONResponse({
        "message": f"Download {file_type} for workflow {workflow_id}",
        "workflow_id": workflow_id,
        "file_type": file_type
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
