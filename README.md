# 🚀 DS Capstone Project: Multi-Agent Classification System

## 📋 Project Overview

An AI-powered multi-agent system that automates the entire machine learning classification pipeline from data cleaning to model deployment. The system uses multiple specialized agents working together to solve data science classification problems.

## 🏗️ Architecture

### Backend (FastAPI + LangGraph)
- **FastAPI**: REST API and WebSocket endpoints
- **LangGraph**: Multi-agent workflow orchestration
- **Celery**: Async task processing
- **PostgreSQL**: Data persistence
- **Redis**: Caching and message broker

### Frontend (Next.js 14 + TypeScript)
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Styling and responsive design
- **WebSocket**: Real-time communication

## 🤖 Agents

1. **Data Cleaning Agent**: Cleans and preprocesses datasets
2. **Data Discovery Agent**: Researches similar datasets and approaches
3. **EDA Agent**: Performs exploratory data analysis
4. **Feature Engineering Agent**: Creates and selects features
5. **ML Model Builder Agent**: Trains and optimizes models
6. **Model Evaluation Agent**: Evaluates model performance
7. **Technical Reporter Agent**: Generates comprehensive reports
8. **Project Manager Agent**: Orchestrates the entire workflow

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL
- Redis

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
ds-capstone-project/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── agents/         # Individual agent modules
│   │   ├── workflows/      # LangGraph workflows
│   │   ├── ml/            # ML pipeline components
│   │   ├── services/      # Business logic
│   │   └── api/           # API endpoints
│   └── tests/             # Backend tests
├── frontend/               # Next.js frontend
│   ├── app/               # App Router pages
│   ├── components/        # React components
│   └── lib/              # Utilities and API client
├── docker/                # Containerization
├── infrastructure/        # Deployment configs
└── docs/                 # Documentation
```

## 🔧 Development Status

- [x] Project structure setup
- [ ] Backend foundation
- [ ] Agent development
- [ ] Frontend development
- [ ] Integration
- [ ] Testing
- [ ] Deployment

## 📚 Documentation

- [API Documentation](docs/api/)
- [Agent Documentation](docs/agents/)
- [Deployment Guide](docs/deployment/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details
