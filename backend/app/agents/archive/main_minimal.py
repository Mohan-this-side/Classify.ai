"""
🚀 DS Capstone Project - Minimal FastAPI Application

This is a minimal version that doesn't initialize the workflow until needed.
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
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
        self.session_data[session_id] = {"connected_at": datetime.now()}
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

# Initialize connection manager
manager = ConnectionManager()

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
            
            # Echo back the message
            await manager.send_personal_message({
                "type": "echo",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }, session_id)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)

# Simple test endpoint
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify backend is working"""
    return {"message": "Backend is working!", "timestamp": datetime.now()}

# Basic workflow endpoint (without full workflow initialization)
@app.post("/api/workflow/start")
async def start_workflow_minimal(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    description: str = Form(...),
    api_key: str = Form("dummy_key"),
    user_id: Optional[str] = Form("web_user")
):
    """Minimal workflow start endpoint"""
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Create a simple response
        workflow_id = str(uuid.uuid4())
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow started successfully",
            "file_name": file.filename,
            "target_column": target_column,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
