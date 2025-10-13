#!/bin/bash

# 🚀 DS Capstone Multi-Agent System Startup Script

echo "🚀 Starting DS Capstone Multi-Agent System"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Working directory: $SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Function to kill existing processes
kill_existing_processes() {
    echo "🔍 Checking for existing processes..."
    
    # Kill existing uvicorn processes
    UVICORN_PIDS=$(pgrep -f "uvicorn app.main:app")
    if [ ! -z "$UVICORN_PIDS" ]; then
        echo "🛑 Killing existing uvicorn processes: $UVICORN_PIDS"
        kill $UVICORN_PIDS 2>/dev/null
        sleep 2
    fi
    
    # Kill existing Next.js processes
    NEXTJS_PIDS=$(pgrep -f "next dev")
    if [ ! -z "$NEXTJS_PIDS" ]; then
        echo "🛑 Killing existing Next.js processes: $NEXTJS_PIDS"
        kill $NEXTJS_PIDS 2>/dev/null
        sleep 2
    fi
    
    echo "✅ Existing processes cleaned up"
}

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend (FastAPI)..."
    cd "$SCRIPT_DIR/backend" || { echo "❌ Failed to change to backend directory"; exit 1; }
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔧 Activating Python virtual environment..."
    source venv/bin/activate
    
    # Verify activation
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Test agent imports
    echo "🧪 Testing agent imports..."
    python3 -c "from app.agents.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent; print('✅ Agents imported successfully')" || {
        echo "❌ Agent import test failed"
        exit 1
    }
    
    # Start backend server
    echo "🚀 Starting FastAPI server on http://localhost:8000"
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    cd "$SCRIPT_DIR"
    echo "✅ Backend started with PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo "🔧 Starting Frontend (Next.js)..."
    cd "$SCRIPT_DIR/frontend" || { echo "❌ Failed to change to frontend directory"; exit 1; }
    
    # Install dependencies
    echo "📦 Installing Node.js dependencies..."
    npm install
    
    # Start frontend server
    echo "🚀 Starting Next.js server on http://localhost:3000"
    npm run dev &
    FRONTEND_PID=$!
    
    cd "$SCRIPT_DIR"
    echo "✅ Frontend started with PID: $FRONTEND_PID"
}

# Function to wait for services
wait_for_services() {
    echo "⏳ Waiting for services to start..."
    
    # Wait for backend
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ Backend is ready!"
            break
        fi
        echo "  Waiting for backend... ($i/30)"
        sleep 2
    done
    
    # Wait for frontend
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null; then
            echo "✅ Frontend is ready!"
            break
        fi
        echo "  Waiting for frontend... ($i/30)"
        sleep 2
    done
}

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Clean up existing processes
kill_existing_processes

# Start services
start_backend
start_frontend

# Wait for services to be ready
wait_for_services

echo ""
echo "🎉 DS Capstone Multi-Agent System is running!"
echo "============================================="
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running
wait
