# 🚀 Classify AI: Multi-Agent Machine Learning Classification System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

An AI-powered multi-agent system that automates the entire machine learning classification pipeline from data upload to model deployment. Built with FastAPI, Next.js, and LangGraph, it features human-in-the-loop approval gates and secure sandboxed code execution.

## ✨ Key Features

- 🤖 **7 Specialized AI Agents** working in orchestrated workflow
- 🔒 **Secure Sandbox Execution** with Docker for AI-generated code
- 👥 **Human-in-the-Loop** approval gates for critical decisions
- 📊 **Real-time Progress Tracking** via WebSocket connections
- 📈 **Interactive Visualizations** with Plotly and Matplotlib
- 📝 **Comprehensive Reporting** with Jupyter notebook generation
- 🎯 **Educational Explanations** for non-technical users
- 🔄 **Self-Healing Code Generation** with iterative refinement

## 🏗️ System Architecture

```
Frontend (Next.js) 
    ↓ WebSocket/REST API
Backend (FastAPI) 
    ↓ LangGraph Workflow
AI Agents (Domain-Organized)
    ↓ Sandbox Execution
Docker Sandbox (Secure Code Execution)
```

### Backend Stack
- **FastAPI**: High-performance REST API and WebSocket endpoints
- **LangGraph**: Multi-agent workflow orchestration with state management
- **PostgreSQL**: Data persistence and workflow state storage
- **Redis**: Caching and Celery message broker
- **Docker**: Secure sandbox for AI-generated code execution
- **Google Gemini Pro**: AI model for code generation and analysis

### Frontend Stack
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling and responsive design
- **WebSocket**: Real-time communication with backend
- **React Hot Toast**: User notifications and feedback

## 🤖 AI Agents (Domain-Organized)

### 🧹 Data Cleaning Domain
- **Enhanced Data Cleaning Agent**: Main orchestrator with intelligent prompt engineering
- **Missing Value Analyzer**: Analyzes missing data patterns (MCAR, MAR, MNAR)
- **Missing Value Imputer**: Handles various imputation strategies
- **Outlier Detector**: Identifies outliers using multiple methods (IQR, Z-score, Isolation Forest)
- **Data Type Validator**: Validates and converts data types automatically
- **Educational Explainer**: Generates clear explanations for data cleaning decisions

### 📊 Data Analysis Domain
- **Data Discovery Agent**: Comprehensive data profiling and intelligent feature recommendations
- **EDA Agent**: Advanced exploratory data analysis with interactive Plotly visualizations

### 🤖 ML Pipeline Domain
- **Feature Engineering Agent**: Creates and selects optimal features
- **ML Builder Agent**: Hybrid architecture combining hardcoded analysis with LLM code generation
- **Model Evaluation Agent**: Comprehensive performance assessment with multiple metrics

### 📝 Reporting Domain
- **Technical Reporter Agent**: Generates comprehensive reports and reproducible Jupyter notebooks

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (optional, can use SQLite for development)
- Redis (optional, can use in-memory for development)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/classify-ai.git
cd classify-ai
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up environment variables
cp ../env.example .env
# Edit .env with your API keys (Google Gemini Pro, etc.)

# Start the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
ds-capstone-project/
├── 📁 backend/                          # Backend API & Core Logic
│   ├── 📁 app/                          # Main application
│   │   ├── 📁 agents/                   # AI Agents (organized by domain)
│   │   │   ├── 📁 data_cleaning/        # Data Cleaning Agent & Components
│   │   │   │   ├── enhanced_data_cleaning_agent.py
│   │   │   │   ├── missing_value_analyzer.py
│   │   │   │   ├── missing_value_imputer.py
│   │   │   │   ├── outlier_detector.py
│   │   │   │   ├── data_type_validator.py
│   │   │   │   ├── educational_explainer.py
│   │   │   │   └── __init__.py
│   │   │   ├── 📁 data_analysis/        # Discovery & EDA Agents
│   │   │   │   ├── data_discovery_agent.py
│   │   │   │   ├── eda_agent.py
│   │   │   │   └── __init__.py
│   │   │   ├── 📁 ml_pipeline/          # ML Building & Evaluation
│   │   │   │   ├── feature_engineering_agent.py
│   │   │   │   ├── ml_builder_agent.py
│   │   │   │   ├── model_evaluation_agent.py
│   │   │   │   └── __init__.py
│   │   │   ├── 📁 reporting/            # Technical Reporting
│   │   │   │   ├── technical_reporter_agent.py
│   │   │   │   └── __init__.py
│   │   │   ├── 📁 archive/              # Old/Unused Agent Files
│   │   │   ├── base_agent.py
│   │   │   └── __init__.py
│   │   ├── 📁 api/                      # API Routes
│   │   │   ├── workflow_routes.py
│   │   │   └── approval_routes.py
│   │   ├── 📁 workflows/                # Workflow Management
│   │   │   ├── classification_workflow.py
│   │   │   ├── state_management.py
│   │   │   └── approval_gates.py
│   │   ├── 📁 services/                 # Core Services
│   │   │   ├── realtime.py
│   │   │   ├── sandbox_executor.py
│   │   │   ├── tasks.py
│   │   │   └── celery_app.py
│   │   ├── 📁 models/                   # Data Models
│   │   │   └── database.py
│   │   ├── 📁 config/                   # Configuration
│   │   │   └── settings.py
│   │   ├── main.py                      # FastAPI App Entry Point
│   │   └── __init__.py
│   ├── 📁 tests/                        # Backend Tests
│   ├── 📁 models/                       # Generated Models (Runtime)
│   ├── 📁 notebooks/                    # Generated Notebooks (Runtime)
│   ├── 📁 plots/                        # Generated Plots (Runtime)
│   └── requirements.txt
├── 📁 frontend/                         # Next.js Frontend
│   ├── 📁 app/                          # Next.js 13+ App Router
│   │   ├── 📁 components/               # React Components
│   │   │   ├── AgentStatus.tsx
│   │   │   ├── ApiKeySettings.tsx
│   │   │   ├── ApprovalGate.tsx
│   │   │   ├── ApprovalGateManager.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   ├── PlotViewer.tsx
│   │   │   ├── ProgressTracker.tsx
│   │   │   ├── RealtimeInsights.tsx
│   │   │   └── ResultsViewer.tsx
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── 📁 lib/                          # Utility Libraries
│   ├── 📁 public/                       # Static Assets
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── next-env.d.ts
├── 📁 docker/                           # Docker Configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.sandbox
│   └── sandbox-seccomp.json
├── 📁 infrastructure/                   # Infrastructure as Code
│   ├── 📁 kubernetes/
│   ├── 📁 monitoring/
│   └── 📁 terraform/
├── 📁 docs/                             # Documentation
│   ├── 📁 agents/
│   ├── 📁 api/
│   └── 📁 deployment/
├── 📁 scripts/                          # Utility Scripts
│   └── PRD.txt
├── 📁 archive/                          # Archived Files
│   ├── 📁 test_files/                   # Test & Demo Files
│   └── 📁 old_components/               # Duplicate Frontend Components
├── 📁 test_data/                        # Test Datasets
├── 📁 models/                           # Current Generated Models
├── 📁 notebooks/                        # Current Generated Notebooks
├── 📁 plots/                            # Current Generated Plots
├── README.md
├── TECHNICAL_ARCHITECTURE.md
├── SERVER_SETUP.md
└── env.example
```

## 🔄 Workflow Process

1. **Data Upload**: User uploads CSV/Excel file with target column specification
2. **Data Cleaning**: Automated cleaning with missing value analysis, outlier detection, and type validation
3. **Data Discovery**: Comprehensive profiling and feature recommendations
4. **EDA Analysis**: Interactive visualizations and statistical analysis
5. **Feature Engineering**: Automated feature creation and selection
6. **Model Building**: Hybrid approach with hardcoded analysis + LLM code generation
7. **Model Evaluation**: Comprehensive performance assessment with multiple metrics
8. **Technical Reporting**: Generation of detailed reports and Jupyter notebooks
9. **Human Approval Gates**: Critical decision points requiring user approval

## 🔒 Security Features

- **Docker Sandbox**: All AI-generated code runs in isolated containers
- **Resource Limits**: CPU, memory, and execution time constraints
- **Network Isolation**: Prevents unauthorized external access
- **Seccomp Profiles**: Additional security hardening
- **Input Validation**: Comprehensive data validation and sanitization

## 🎯 Use Cases

- **Data Scientists**: Accelerate model development and experimentation
- **Business Analysts**: Get insights without deep technical knowledge
- **Students**: Learn data science through guided, educational workflows
- **Enterprises**: Standardize and automate ML workflows across teams

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test

# Integration tests
python test_system.py
```

## 🚀 Deployment

### Docker Compose
```bash
docker-compose up -d
```

### Manual Deployment
See [SERVER_SETUP.md](SERVER_SETUP.md) for detailed deployment instructions.

## 📊 Performance Metrics

- **Data Processing**: Handles datasets up to 100K+ rows
- **Model Training**: Supports scikit-learn, XGBoost, LightGBM models
- **Response Time**: Real-time updates via WebSocket
- **Security**: Zero-trust sandbox execution environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [Technical Architecture](TECHNICAL_ARCHITECTURE.md)
- [API Documentation](http://localhost:8000/docs)
- [Agent Documentation](docs/agents/)
- [Deployment Guide](docs/deployment/)

## 🐛 Known Issues

- Large datasets (>1M rows) may require additional memory configuration
- Some complex feature engineering operations may timeout in sandbox
- WebSocket connections may drop on unstable networks

## 🔮 Roadmap

- [ ] Support for additional ML frameworks (PyTorch, TensorFlow)
- [ ] Advanced visualization types (3D plots, interactive dashboards)
- [ ] Multi-language support for generated code
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Advanced model interpretability features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API framework
- [Next.js](https://nextjs.org/) for the React framework
- [Google Gemini Pro](https://ai.google.dev/) for AI capabilities

## 📞 Support

For support, email at mohanbhosale6@gmail.con

---

**Made with ❤️ by the Classify AI Team**