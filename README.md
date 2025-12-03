# Classify AI - Multi-Agent Classification System

A comprehensive machine learning pipeline system that uses multiple AI agents to perform end-to-end classification tasks, from data discovery to model evaluation and reporting.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized AI agents for each stage of the ML pipeline
- **Double-Layer Execution**: Layer 1 (hardcoded) + Layer 2 (LLM-generated code in Docker sandbox)
- **Real-time Progress Tracking**: WebSocket-based progress updates
- **Comprehensive Analysis**: Automated EDA, feature engineering, model building, and evaluation
- **Interactive Project Manager**: AI-powered chatbot for answering questions about the workflow
- **Secure Code Execution**: Docker sandbox for safe execution of LLM-generated code

## 🎥 Demo Walkthrough

Watch a complete demonstration of the Classify AI system in action:

[![Classify AI Demo Walkthrough](https://img.youtube.com/vi/pHOhsWIxguo/maxresdefault.jpg)](https://www.youtube.com/watch?v=pHOhsWIxguo&t=1s)

**Click the image above to watch the full demo video on YouTube**

The demo covers:
- Dataset upload and configuration
- Real-time workflow execution
- Interactive Project Manager
- Model training and evaluation
- Results visualization and download

## 📁 Project Structure

```
ds-capstone-project/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── agents/         # AI agents (data discovery, EDA, cleaning, etc.)
│   │   ├── api/            # API routes
│   │   ├── services/       # Core services (LLM, sandbox, validation)
│   │   └── workflows/      # Workflow orchestration
│   └── tests/              # Test suites
├── frontend/                # Next.js frontend
│   └── app/                # React components
├── docs/                    # Documentation
│   ├── important/          # Critical documentation
│   ├── fixes/              # Bug fixes and improvements
│   ├── guides/             # User guides
│   ├── architecture/       # Technical architecture
│   └── test-results/       # Test results
├── docker/                 # Docker configurations
├── config/                 # Configuration files
└── test_data/             # Sample datasets
```

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop
- API Key (Gemini, OpenAI, or Anthropic)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ds-capstone-project
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Environment Configuration**
   ```bash
   cp config/env.example .env
   # Edit .env with your API keys
   ```

## 🚀 Running the Application

### Start Backend
```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Documentation

All documentation is organized in the `docs/` directory:

### Documentation Structure

- **`docs/important/`** - Critical documentation and important notes
  - See `docs/important/README.md` for overview
  - Contains links to all critical fixes and improvements

- **`docs/fixes/`** - Bug fixes, root cause analyses, and technical improvements
  - See `docs/fixes/README.md` for complete list
  - 15+ fix documents covering Layer 2, Docker, code validation, and more

- **`docs/changelog/`** - Project status updates and changelog
  - See `docs/changelog/README.md` for status history

- **`docs/guides/`** - User guides and quick start documentation
  - Quick Start Guide
  - Docker workflow requirements
  - Deployment guides

- **`docs/architecture/`** - Technical architecture documentation
  - System architecture
  - Double-layer architecture details

- **`docs/test-results/`** - Test results and evaluation reports

### Key Documents

- **Quick Start**: `docs/guides/QUICK_START_GUIDE.md`
- **Architecture**: `docs/architecture/TECHNICAL_ARCHITECTURE.md`
- **API Documentation**: `docs/API.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Important Fixes**: `docs/important/README.md`

## 🤖 Agents

The system uses specialized AI agents:

1. **Data Discovery** - Analyzes dataset structure and characteristics
2. **EDA Analysis** - Performs exploratory data analysis
3. **Data Cleaning** - Cleans and preprocesses data
4. **Feature Engineering** - Creates and selects features
5. **ML Builder** - Builds and trains classification models
6. **Model Evaluation** - Evaluates model performance
7. **Technical Reporter** - Generates technical documentation
8. **Project Manager** - Coordinates workflow and answers questions

## 🔒 Security

- LLM-generated code runs in isolated Docker containers
- No network access from sandbox containers
- Resource limits and timeouts enforced
- Code validation before execution

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

This is a capstone project for Northeastern University. For questions or issues, please contact the development team.

## 📧 Contact

Classify AI Team
