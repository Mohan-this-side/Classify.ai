# DS Capstone Project - Folder Structure

## 📁 Project Organization

```
ds-capstone-project/
│
├── backend/                          # Backend API and ML pipeline
│   ├── app/
│   │   ├── agents/                   # Multi-agent system
│   │   │   ├── coordination/         # Project Manager Agent
│   │   │   ├── data_analysis/        # Data Discovery & EDA Agents
│   │   │   ├── data_cleaning/        # Enhanced Data Cleaning Agent
│   │   │   ├── ml_pipeline/          # ML Builder & Evaluation Agents
│   │   │   └── reporting/            # Technical Reporter Agent
│   │   ├── api/                      # FastAPI routes
│   │   │   ├── workflow_routes.py    # Main workflow endpoints
│   │   │   └── approval_routes.py    # Approval gate endpoints
│   │   ├── services/                 # Service layer
│   │   │   ├── llm_service.py        # Multi-provider LLM integration
│   │   │   ├── code_validator.py     # Code security & validation
│   │   │   ├── storage.py            # Results storage service
│   │   │   ├── realtime.py           # WebSocket service
│   │   │   └── celery_app.py         # Background tasks
│   │   ├── workflows/                # Workflow orchestration
│   │   │   ├── classification_workflow.py  # Main LangGraph workflow
│   │   │   ├── state_management.py   # State management
│   │   │   └── approval_gates.py     # HITL approval gates
│   │   ├── models/                   # Database models
│   │   ├── config.py                 # Configuration management
│   │   └── main.py                   # FastAPI application entry
│   │
│   ├── results/                      # Generated workflow results (gitignored)
│   │   └── {workflow-id}/
│   │       ├── cleaned_dataset.csv
│   │       ├── model.joblib
│   │       ├── notebook.ipynb
│   │       ├── report.md
│   │       └── plots/
│   │
│   ├── tests/                        # Backend tests
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # Next.js frontend
│   ├── app/
│   │   ├── components/               # React components
│   │   │   ├── ApprovalGate.tsx
│   │   │   ├── ApprovalGateManager.tsx
│   │   │   └── ui/                   # UI components
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/                   # Shared components
│   ├── lib/                          # Utilities
│   ├── public/                       # Static assets
│   ├── package.json
│   └── next.config.js
│
├── docker/                           # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── docs/                             # Documentation
│   ├── test-results/                 # Test summaries
│   │   ├── CURRENT_SYSTEM_ASSESSMENT.md
│   │   ├── PROGRESS_SUMMARY.md
│   │   ├── LOAN_APPROVAL_TEST_RESULTS.md
│   │   └── FINAL_SUMMARY.md
│   ├── agents/                       # Agent documentation
│   ├── api/                          # API documentation
│   └── deployment/                   # Deployment guides
│
├── test_data/                        # Test datasets
│   ├── README.md
│   ├── Loan Approval Dataset.csv     # Real-world financial data
│   ├── spotify_churn_dataset.csv     # Churn analysis dataset
│   ├── iris_clean.csv                # Clean test data
│   └── messy_*.csv                   # Various data quality test files
│
├── infrastructure/                   # Infrastructure as code
│   ├── kubernetes/
│   ├── monitoring/
│   └── terraform/
│
├── .cursor/                          # Cursor AI settings (gitignored)
├── .taskmaster/                      # Taskmaster AI settings (gitignored)
│
├── .env                              # Environment variables (gitignored)
├── .gitignore                        # Git ignore rules
├── README.md                         # Project overview
├── PROJECT_STRUCTURE.md              # This file
├── setup.sh                          # Project setup script
├── start_system.sh                   # System startup script
└── requirements.txt                  # Root Python dependencies
```

---

## 🗂️ Key Directories Explained

### `/backend/app/agents/`
Contains all AI agents organized by function:
- **coordination**: Project Manager Agent for real-time updates
- **data_analysis**: Data Discovery & EDA for profiling and analysis
- **data_cleaning**: Advanced data cleaning with quality metrics
- **ml_pipeline**: ML Builder & Model Evaluation agents
- **reporting**: Technical Reporter for notebooks and reports

### `/backend/app/services/`
Service layer for cross-cutting concerns:
- **llm_service**: Multi-provider LLM integration (Gemini, OpenAI, Anthropic)
- **code_validator**: Security scanning and code quality validation
- **storage**: Centralized file storage management
- **realtime**: WebSocket communication for live updates

### `/backend/app/workflows/`
LangGraph-based workflow orchestration:
- **classification_workflow**: Main multi-agent pipeline
- **state_management**: Centralized state handling
- **approval_gates**: Human-in-the-loop decision points

### `/backend/results/`
Generated outputs for each workflow run (gitignored):
- Cleaned datasets
- Trained models
- Jupyter notebooks
- Technical reports
- Visualizations

### `/frontend/`
Next.js application with TypeScript and Tailwind CSS:
- React components for UI
- Approval gate interface
- Real-time progress tracking

### `/docs/`
Project documentation:
- **test-results**: Test summaries and assessments
- **agents**: Agent-specific documentation
- **api**: API reference documentation
- **deployment**: Deployment and operations guides

### `/test_data/`
Sample datasets for testing:
- Clean datasets for baseline testing
- Messy datasets for data quality testing
- Real-world datasets (Loan Approval, Spotify Churn)

---

## 🚫 Gitignored Items

The following are excluded from git tracking:

### Generated Files
- `backend/results/` - Workflow outputs
- `backend/models/` - Trained models
- `backend/plots/` - Generated visualizations
- `backend/notebooks/` - Generated notebooks

### Development
- `.cursor/` - Cursor AI settings
- `.taskmaster/` - Taskmaster AI settings
- `venv/` - Python virtual environment
- `node_modules/` - Node dependencies
- `__pycache__/` - Python cache files

### Sensitive
- `.env` - Environment variables
- `*.log` - Log files

### Temporary
- `test_results/` - Test output
- `*.tmp`, `*.temp` - Temporary files

---

## 📝 Important Files

### Configuration
- `.env` - Environment variables (API keys, database URLs)
- `backend/app/config.py` - Application configuration
- `docker/docker-compose.yml` - Docker orchestration

### Setup
- `setup.sh` - Initial project setup
- `start_system.sh` - Start backend + frontend
- `requirements.txt` - Python dependencies
- `frontend/package.json` - Node dependencies

### Documentation
- `README.md` - Project overview
- `docs/test-results/FINAL_SUMMARY.md` - Complete test results
- `docs/test-results/LOAN_APPROVAL_TEST_RESULTS.md` - Real-world test

---

## 🔄 Workflow Data Flow

```
User Upload (CSV)
      ↓
FastAPI Endpoint (/api/workflow/start)
      ↓
Classification Workflow (LangGraph)
      ↓
┌─────────────────────────────────────┐
│ Multi-Agent Pipeline (Sequential)   │
│                                      │
│ 1. Data Cleaning Agent               │
│    └→ Cleaned dataset                │
│                                      │
│ 2. Data Discovery Agent              │
│    └→ Data profiling & insights      │
│                                      │
│ 3. EDA Agent                         │
│    └→ Statistical analysis           │
│                                      │
│ 4. Feature Engineering Agent         │
│    └→ Engineered features            │
│                                      │
│ 5. ML Builder Agent                  │
│    └→ Trained models                 │
│                                      │
│ 6. Model Evaluation Agent            │
│    └→ Performance metrics            │
│                                      │
│ 7. Technical Reporter Agent          │
│    └→ Notebook + Report              │
└─────────────────────────────────────┘
      ↓
Storage Service
      ↓
backend/results/{workflow-id}/
  ├── cleaned_dataset.csv
  ├── model.joblib
  ├── notebook.ipynb
  ├── report.md
  └── plots/
      ↓
Download Endpoints (/api/workflow/{id}/download)
```

---

## 🛠️ Development Workflow

### 1. Setup
```bash
./setup.sh                    # Initial setup
source venv/bin/activate      # Activate Python environment
```

### 2. Development
```bash
./start_system.sh             # Start backend + frontend
```

### 3. Testing
```bash
# Backend is already running at localhost:8000
# Upload dataset via API or frontend
```

### 4. Results
```bash
# Check backend/results/{workflow-id}/ for outputs
```

---

## 📦 Dependencies

### Backend (Python)
- FastAPI - Web framework
- LangGraph - Workflow orchestration
- Pandas, NumPy - Data processing
- Scikit-learn - ML algorithms
- Plotly, Matplotlib - Visualizations
- Google Generative AI - LLM integration

### Frontend (Node.js)
- Next.js - React framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- Lucide React - Icons

### Infrastructure
- Docker - Containerization
- PostgreSQL - Database
- Redis - Caching
- Celery - Background tasks

---

## 🔐 Security Notes

- All API keys stored in `.env` (gitignored)
- Generated code validated before execution
- Sandbox execution for LLM-generated code
- Input validation on all endpoints
- CORS configuration for production

---

**Last Updated**: October 22, 2025

