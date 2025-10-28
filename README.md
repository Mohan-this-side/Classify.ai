# Classify AI: Automated ML Pipeline with Real-Time Education

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/Mohan-this-side/Classify.ai)

## Overview

**Classify AI** is an intelligent, end-to-end machine learning automation system that transforms raw datasets into production-ready classification models through a sophisticated multi-agent architecture. The system automates the entire ML pipeline—from data cleaning to model deployment—while maintaining transparency, security, and educational value throughout the process.

### Key Value Proposition

- **Automated Pipeline**: Upload a dataset, receive a trained model, cleaned data, Jupyter notebook, and comprehensive technical report
- **Double-Layer Architecture**: Combines reliable hardcoded analysis with adaptive LLM-generated code for optimal results
- **Educational Transparency**: Real-time explanations of every decision and transformation through an interactive Project Manager
- **Production Ready**: Generates deployment-ready models with complete documentation
- **Secure Execution**: All LLM-generated code runs in isolated Docker sandboxes with comprehensive monitoring

---

## User Interface

### Upload Page
The clean, intuitive interface makes it easy to get started with your machine learning project:

![Upload Page](https://via.placeholder.com/800x600/4F46E5/FFFFFF?text=Upload+Page+Interface)

**Features:**
- **Drag & Drop Upload**: Simply drag your CSV or Excel file into the upload zone
- **Smart Column Detection**: Automatic detection of available columns for target selection
- **API Key Integration**: Support for Gemini, OpenAI, and Anthropic API keys
- **File Validation**: Automatic validation with size and format constraints (Max 100MB, up to 1M rows)

### Workflow Page
Real-time monitoring of your ML pipeline with interactive Project Manager:

![Workflow Page](https://via.placeholder.com/1200x800/10B981/FFFFFF?text=Workflow+Page+with+Project+Manager)

**Features:**
- **Real-time Progress**: Live tracking of all 8 agents with execution times and Layer 1/Layer 2 indicators
- **Interactive Project Manager**: Educational chat interface with Q&A capabilities
- **Approval Gates**: Human-in-the-loop decision points with educational context
- **Sandbox Monitoring**: Real-time CPU, memory, and execution time monitoring
- **Agent Status**: Detailed status for each agent with completion indicators

---

## System Architecture

### Multi-Agent Workflow System

Our system employs **8 specialized AI agents** working in concert to handle different aspects of the ML pipeline:

```mermaid
graph TB
    subgraph "Agent Workflow"
        DD[Data Discovery Agent<br/>Dataset Profiling]
        EDA[EDA Agent<br/>Exploratory Analysis]
        DC[Data Cleaning Agent<br/>Quality Improvement]
        FE[Feature Engineering Agent<br/>Feature Creation]
        ML[ML Builder Agent<br/>Model Training]
        ME[Model Evaluation Agent<br/>Performance Analysis]
        TR[Technical Reporter Agent<br/>Documentation]
        PM[Project Manager Agent<br/>Education & Guidance]
    end
    
    DD --> EDA
    EDA --> DC
    DC --> FE
    FE --> ML
    ML --> ME
    ME --> TR
    PM -.-> DD
    PM -.-> EDA
    PM -.-> DC
    PM -.-> FE
    PM -.-> ML
    PM -.-> ME
    PM -.-> TR
```

### Double-Layer Architecture

Each agent employs a sophisticated two-layer approach:

**Layer 1 - Hardcoded Analysis** ⚡
- Pre-written, battle-tested Python functions
- Comprehensive data quality assessment
- Statistical profiling and pattern detection
- Reliable baseline analysis that never fails

**Layer 2 - LLM Code Generation** 🐳
- Custom code tailored to specific dataset characteristics
- Adaptive processing based on Layer 1 insights
- Validated and executed in secure Docker sandboxes
- Enhanced analysis with dataset-specific optimizations

---

## Core Features

### 1. Interactive Project Manager
A dedicated AI assistant that provides:
- **Educational Messages**: Real-time explanations of each processing step
- **Q&A System**: Ask questions about the workflow and get intelligent answers
- **Approval Gates**: Interactive decision points with educational context
- **Progress Updates**: Detailed status updates and completion notifications

### 2. Human-in-the-Loop Control
Critical decision points where the workflow pauses for user approval:
- **Data Cleaning Strategies**: Choose between different imputation methods
- **Feature Engineering Approaches**: Approve or modify feature creation
- **Model Selection Decisions**: Review and approve model choices
- **Educational Context**: Learn about each decision with detailed explanations

### 3. Secure Code Execution
All LLM-generated code runs in isolated Docker containers with:
- **No Network Access**: Complete isolation from external systems
- **Resource Limits**: CPU, memory, and execution time constraints
- **Complete Audit Trail**: Full logging of all operations
- **Security Validation**: Multi-layer code validation and security scanning

### 4. Comprehensive Deliverables
Every workflow generates:
- **Cleaned Dataset**: Processed CSV with quality improvements
- **Trained Model**: Serialized `.joblib` model file
- **Jupyter Notebook**: Reproducible analysis notebook
- **Technical Report**: Comprehensive documentation and insights
- **Visualizations**: EDA plots, performance charts, and feature importance
- **Execution Logs**: Complete audit trail of all operations

---

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Workflow Engine**: LangGraph for stateful multi-agent orchestration
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, NumPy, SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **LLM Integration**: Google Gemini, OpenAI GPT, Anthropic Claude

### Frontend
- **Framework**: Next.js 14+ (React)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom components
- **Icons**: Lucide React
- **State Management**: React Hooks with real-time updates
- **UI Components**: Custom-built components with modern design

### Infrastructure
- **Containerization**: Docker with Docker Compose
- **Database**: PostgreSQL for workflow state and metadata
- **Cache**: Redis for task queues and caching
- **Message Queue**: Celery for asynchronous task processing
- **WebSocket**: FastAPI WebSocket for real-time communication

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- Node.js 16 or higher
- Docker and Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mohan-this-side/Classify.ai.git
cd Classify.ai
```

2. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your API keys (optional for basic usage)
```

3. **Start the system**
```bash
# Using Docker Compose (Recommended)
docker-compose -f docker/docker-compose.yml up -d

# Or using the setup script
chmod +x setup.sh
./setup.sh
```

4. **Access the application**
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Manual Setup

**Backend**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

---

## Usage Guide

### Basic Workflow

1. **Prepare Your Dataset**
   - CSV or Excel format
   - Include a target column for classification
   - Any number of features (tested up to 50+ features)

2. **Upload and Configure**
   - Navigate to http://localhost:3001
   - Upload your dataset using drag & drop or file browser
   - Select your target column from the dropdown
   - Provide an LLM API key (optional for Layer 1 functionality)

3. **Monitor Progress**
   - Watch real-time progress updates in the workflow view
   - Read educational explanations from the Project Manager
   - Respond to approval gates when prompted
   - Ask questions using the Q&A interface

4. **Download Results**
   - Cleaned dataset (CSV)
   - Trained model (.joblib)
   - Jupyter notebook (.ipynb)
   - Technical report (.md)
   - Visualizations (.png)

### Example: Loan Approval Prediction

```python
# Example workflow result for a 2,000-row loan dataset:
# - Processing time: ~5 minutes
# - Model accuracy: 100%
# - Deliverables: 5 files (652KB model, 111KB dataset, 30KB notebook, etc.)
# - Educational messages: 12 interactive updates
# - Approval gates: 2 decision points
```

---

## Performance Metrics

Based on comprehensive testing with real-world datasets:

| Dataset | Size | Processing Time | Accuracy | Deliverables | Educational Value |
|---------|------|----------------|----------|--------------|-------------------|
| Loan Approval | 2,000 rows | ~5 minutes | 100% | 5 files (737KB) | 12 messages, 2 gates |
| Iris Dataset | 150 rows | ~2 minutes | 97% | 5 files (42KB) | 8 messages, 1 gate |
| Clean Test | 100 rows | ~1 minute | 100% | 5 files (58KB) | 6 messages, 1 gate |

**System Capabilities**:
- Handles datasets up to 100,000+ rows
- Supports 50+ features
- Trains 9+ ML algorithms in parallel
- Generates complete documentation automatically
- Provides real-time educational guidance

---

## Security Features

### Code Validation
- **AST Parsing**: Syntax validation for all generated code
- **Security Scanning**: Detection of dangerous operations
- **Import Validation**: Whitelist-based import checking
- **Resource Usage**: CPU, memory, and time limit enforcement

### Sandboxed Execution
- **Docker Isolation**: Complete container isolation
- **No Network Access**: Zero external connectivity
- **Resource Limits**: Strict CPU and memory constraints
- **Execution Timeouts**: Automatic termination of long-running code
- **Non-root Execution**: Security-first execution environment

### Audit Trail
- **Complete Logging**: Full audit trail of all operations
- **Generated Code Storage**: All LLM-generated code is stored
- **Execution Results**: Comprehensive tracking of sandbox results
- **User Decisions**: Complete logging of approval gate responses

---

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

### End-to-End Testing
```bash
# Test complete workflow with sample dataset
python test_complete_workflow.py
```

---

## Deployment

### Docker Deployment (Recommended)
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Production Deployment
For production deployment, see our detailed guides:
- **Kubernetes**: [docs/deployment/kubernetes/](docs/deployment/kubernetes/)
- **Cloud Providers**: AWS, GCP, Azure configurations
- **SSL/TLS**: Security configuration
- **Monitoring**: Logging and performance monitoring
- **Scaling**: Horizontal scaling strategies

---

## Roadmap

### Current Features (v1.0)
- Multi-agent classification pipeline
- Real-time progress tracking with Project Manager
- Interactive approval gates with educational context
- Q&A system with LLM integration
- Automated model training and evaluation
- Comprehensive deliverables generation
- Secure sandboxed code execution
- Double-layer architecture (Layer 1 + Layer 2)

### Upcoming Features (v2.0)
- Regression support
- Time series forecasting
- Multi-class classification enhancements
- Advanced feature engineering
- Model ensemble techniques
- A/B testing framework

### Future Enhancements
- Custom agent creation
- Advanced approval gate workflows
- Integration with MLOps platforms
- Model monitoring and drift detection
- Automated retraining pipelines

---

## Troubleshooting

### Common Issues

**Backend won't start**:
```bash
# Check if port 8000 is in use
lsof -ti:8000 | xargs kill -9

# Restart backend
cd backend && source venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

**Frontend won't build**:
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

**Docker issues**:
```bash
# Rebuild containers
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up --build
```

**Project Manager not responding**:
```bash
# Check WebSocket connection
# Ensure backend is running on port 8000
# Check browser console for WebSocket errors
```

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Write tests for new features
- Update documentation as needed
- Follow the existing code style and patterns

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Classify AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **FastAPI** for the excellent web framework
- **LangChain/LangGraph** for workflow orchestration
- **scikit-learn** for ML algorithms
- **Next.js** for the modern frontend framework
- **Tailwind CSS** for beautiful styling
- **The open-source community** for inspiration and support

---

## Contact

- **Project Repository**: [https://github.com/Mohan-this-side/Classify.ai](https://github.com/Mohan-this-side/Classify.ai)
- **Documentation**: [https://classify-ai.readthedocs.io](https://classify-ai.readthedocs.io)
- **Issues**: [https://github.com/Mohan-this-side/Classify.ai/issues](https://github.com/Mohan-this-side/Classify.ai/issues)
- **Discussions**: [https://github.com/Mohan-this-side/Classify.ai/discussions](https://github.com/Mohan-this-side/Classify.ai/discussions)

---

## Citation

If you use Classify AI in your research or project, please cite:

```bibtex
@software{classify_ai_2025,
  title={Classify AI: Automated ML Pipeline with Real-Time Education},
  author={Classify AI Team},
  year={2025},
  url={https://github.com/Mohan-this-side/Classify.ai},
  note={Production-ready automated machine learning system with educational transparency}
}
```

---

## Project Status

**Current Status**: **Production Ready**

- Complete multi-agent workflow implementation
- Interactive Project Manager with Q&A
- Secure sandboxed code execution
- Real-time progress tracking
- Educational transparency and guidance
- Comprehensive testing and validation
- Production-ready deployment configuration

---

**Built with passion for making machine learning accessible to everyone while maintaining transparency and educational value.**