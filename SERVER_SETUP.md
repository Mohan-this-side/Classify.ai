# 🚀 DS Capstone Multi-Agent System - Server Setup Guide

## Overview
This guide will help you set up and run both the backend (FastAPI + LangGraph) and frontend (Next.js) servers for the multi-agent data science system.

## Prerequisites

### Required Software
- **Python 3.9+** (with pip)
- **Node.js 18+** (with npm)
- **Git** (for version control)

### Optional but Recommended
- **PostgreSQL** (for production database)
- **Redis** (for caching and WebSocket support)

## Quick Start (Development Mode)

### 1. Clone and Navigate to Project
```bash
cd "/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"
```

### 2. Backend Setup (FastAPI + LangGraph)

#### Create Python Virtual Environment
```bash
cd backend
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Set Environment Variables
Create a `.env` file in the backend directory:
```bash
# Copy example environment file
cp ../env.example backend/.env

# Edit with your settings
nano backend/.env
```

Required environment variables:
```env
# Google AI API Key (Required)
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# Database (Optional - uses SQLite by default)
DATABASE_URL=sqlite:///./app.db

# Redis (Optional - for WebSocket and caching)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

#### Start Backend Server
```bash
# From backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: **http://localhost:8000**
- API Documentation: **http://localhost:8000/docs**
- Health Check: **http://localhost:8000/health**

### 3. Frontend Setup (Next.js + TypeScript)

Open a new terminal window:

```bash
cd "/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project/frontend"
```

#### Install Dependencies
```bash
npm install
```

#### Start Frontend Development Server
```bash
npm run dev
```

The frontend will be available at: **http://localhost:3000**

## 🎯 Testing the Application

### 1. Access the Application
Open your browser and navigate to: **http://localhost:3000**

### 2. Upload a Dataset
- Click on the file upload area or drag & drop a CSV/Excel file
- Fill in the target column name (e.g., "species", "category")
- Provide a description of your machine learning task
- Click "Start AI Analysis"

### 3. Monitor Real-time Progress
- Watch the AI agents work in real-time
- See progress bars and status updates
- Download results when complete

## 🛠 Advanced Setup

### Using Docker (Alternative)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

### Database Setup (Production)
```bash
# Install PostgreSQL
brew install postgresql  # macOS
# or use your package manager

# Create database
createdb ds_capstone

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://username:password@localhost:5432/ds_capstone
```

### Redis Setup (For WebSockets)
```bash
# Install Redis
brew install redis  # macOS

# Start Redis server
redis-server

# Update REDIS_URL in .env
REDIS_URL=redis://localhost:6379
```

## 🔧 Development Commands

### Backend Commands
```bash
# Start server
uvicorn app.main:app --reload

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy .
```

### Frontend Commands
```bash
# Development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Linting
npm run lint

# Type checking
npm run type-check
```

## 📊 Features to Test

### 1. **Agent Workflow**
- Data Cleaning Agent: Handles missing values and outliers
- Data Discovery Agent: Researches similar datasets
- EDA Agent: Performs exploratory data analysis
- Feature Engineering Agent: Creates optimal features
- ML Builder Agent: Trains and optimizes models
- Model Evaluation Agent: Assesses performance
- Technical Reporter Agent: Generates reports

### 2. **Real-time Updates**
- WebSocket connection for live progress
- Agent status indicators
- Progress bars and animations

### 3. **Results Download**
- Cleaned dataset (CSV)
- Trained model (pickle file)
- Python notebook (Jupyter)
- Technical report (PDF/HTML)

## 🐛 Troubleshooting

### Common Issues

#### Backend Issues
```bash
# Port already in use
lsof -ti:8000 | xargs kill

# Dependencies issue
pip install --upgrade -r requirements.txt

# Database connection
# Check DATABASE_URL in .env file
```

#### Frontend Issues
```bash
# Port already in use
lsof -ti:3000 | xargs kill

# Node modules issue
rm -rf node_modules package-lock.json
npm install

# Build issues
npm run build
```

#### API Key Issues
- Get Google AI API Key: https://ai.google.dev/
- Add to backend/.env file
- Restart backend server

### Environment Variables
Make sure all required environment variables are set in `backend/.env`:
```env
GOOGLE_AI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key
CORS_ORIGINS=["http://localhost:3000"]
```

## 📝 Logs and Debugging

### Backend Logs
```bash
# View logs in terminal where uvicorn is running
# Or check log files if configured
tail -f logs/app.log
```

### Frontend Logs
```bash
# Browser console for frontend issues
# Terminal for build/server issues
```

## 🚀 Production Deployment

### Backend (FastAPI)
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend (Next.js)
```bash
# Build for production
npm run build

# Start production server
npm start
```

## 💡 Tips for Best Experience

1. **Use Latest Browsers**: Chrome, Firefox, Safari latest versions
2. **File Size Limits**: Keep datasets under 100MB for optimal performance
3. **Internet Connection**: Required for AI agent interactions
4. **System Resources**: Ensure adequate RAM for large datasets

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify environment variables are set
4. Check that both servers are running on correct ports

---

🎉 **You're all set!** Your modern, futuristic multi-agent data science system is ready to transform your data into actionable insights.
