"""
🚀 DS Capstone Project - Main FastAPI Application

This is the main entry point for the multi-agent classification system backend.
It provides REST API endpoints and WebSocket connections for real-time communication.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
import asyncio
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .api.workflow_routes import router as workflow_router
from .api.approval_routes import router as approval_router
from .services import realtime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DS Capstone Multi-Agent Classification System",
    description="AI-powered multi-agent system for automated ML classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "status": "connected",
            "created_at": datetime.now(),
            "agents": {},
            "progress": 0.0
        }
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")

    async def broadcast(self, message: dict):
        for session_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")

# Include API routers (lazy initialization)
app.include_router(workflow_router)
app.include_router(approval_router, prefix="/api/workflow")

# Initialize connection manager
manager = ConnectionManager()

# Provide the manager to realtime service so other modules can emit events
realtime.set_connection_manager(manager)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, session_id)
            elif message.get("type") == "get_status":
                status = manager.session_data.get(session_id, {})
                await manager.send_personal_message({"type": "status", "data": status}, session_id)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)

# File upload endpoint
@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    description: str = Form(...),
    api_key: str = Form(...)
):
    """Upload dataset and start classification workflow"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Validate file
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Store file temporarily (in production, use proper file storage)
        file_path = f"temp/{session_id}_{file.filename}"
        
        # TODO: Implement actual file processing and workflow initiation
        # For now, return a mock response
        
        return JSONResponse({
            "session_id": session_id,
            "status": "uploaded",
            "message": "Dataset uploaded successfully. Starting classification workflow...",
            "file_info": {
                "filename": file.filename,
                "size": file.size,
                "target_column": target_column,
                "description": description
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get workflow status
@app.get("/workflow/{session_id}/status")
async def get_workflow_status(session_id: str):
    """Get current workflow status for a session"""
    if session_id not in manager.session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return manager.session_data[session_id]

# Get workflow results
@app.get("/workflow/{session_id}/results")
async def get_workflow_results(session_id: str):
    """Get workflow results for a session"""
    if session_id not in manager.session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Implement actual results retrieval
    return {
        "session_id": session_id,
        "status": "completed",
        "results": {
            "cleaned_dataset": "path/to/cleaned_dataset.csv",
            "trained_model": "path/to/model.pkl",
            "notebook": "path/to/notebook.ipynb",
            "report": "path/to/report.pdf"
        }
    }

# Download results
@app.get("/workflow/{session_id}/download/{file_type}")
async def download_result(session_id: str, file_type: str):
    """Download specific result file"""
    if session_id not in manager.session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Implement actual file download
    return {"message": f"Download {file_type} for session {session_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
