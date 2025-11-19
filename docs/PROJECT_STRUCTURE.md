# Project Structure

This document describes the standard project structure for Classify AI.

## Directory Structure

```
ds-capstone-project/
├── backend/                    # Backend application (FastAPI)
│   ├── app/                    # Application source code
│   │   ├── agents/            # AI agent implementations
│   │   ├── api/                # API route handlers
│   │   ├── services/           # Business logic services
│   │   ├── workflows/          # Workflow orchestration
│   │   ├── models/             # Database models
│   │   ├── utils/              # Utility functions
│   │   ├── config.py           # Configuration
│   │   └── main.py             # Application entry point
│   ├── tests/                  # Backend unit tests
│   ├── requirements.txt         # Python dependencies
│   └── README.md               # Backend documentation
│
├── frontend/                    # Frontend application (Next.js)
│   ├── app/                    # Next.js app directory
│   │   ├── components/         # React components
│   │   └── page.tsx            # Main pages
│   ├── lib/                    # Frontend utilities
│   ├── public/                 # Static assets
│   ├── package.json            # Node dependencies
│   └── README.md               # Frontend documentation
│
├── docker/                      # Docker configurations
│   ├── docker-compose.yml      # Docker Compose setup
│   ├── Dockerfile.backend      # Backend Dockerfile
│   ├── Dockerfile.frontend     # Frontend Dockerfile
│   ├── Dockerfile.sandbox      # Sandbox Dockerfile
│   └── nginx.conf/             # Nginx configuration
│
├── infrastructure/              # Infrastructure as Code
│   ├── kubernetes/             # Kubernetes manifests
│   ├── terraform/              # Terraform configurations
│   └── monitoring/             # Monitoring configurations
│
├── docs/                        # Documentation
│   ├── guides/                 # User guides and tutorials
│   ├── architecture/           # Architecture documentation
│   ├── api/                    # API documentation
│   ├── agents/                 # Agent documentation
│   ├── deployment/             # Deployment guides
│   ├── images/                 # Documentation images
│   └── test-results/           # Test result reports
│
├── scripts/                     # Utility scripts
│   ├── setup.sh                # Project setup script
│   ├── start_system.sh         # Start system script
│   ├── monitor_*.sh            # Monitoring scripts
│   └── watch_*.sh              # Watch scripts
│
├── config/                      # Configuration files
│   ├── .env.example            # Environment variables template
│   └── docker.env.example      # Docker environment template
│
├── tests/                       # Integration tests
│   └── integration/            # End-to-end integration tests
│
├── test_data/                   # Test datasets
│   └── README.md               # Test data documentation
│
├── archive/                     # Archived files
│   ├── generated_artifacts/   # Old generated files
│   ├── old_components/         # Deprecated components
│   └── test_files/            # Old test files
│
├── models/                      # Trained model files (gitignored)
├── notebooks/                   # Jupyter notebooks
├── plots/                       # Generated plots (gitignored)
├── results/                     # Workflow results (gitignored)
│
├── .gitignore                   # Git ignore rules
├── LICENSE                      # License file
├── README.md                    # Main project documentation
└── PRD_GAP_ANALYSIS.md         # PRD gap analysis (gitignored)

```

## Key Directories

### Backend (`backend/`)
- **app/**: Main application code
  - **agents/**: AI agent implementations organized by domain
  - **api/**: REST API route handlers
  - **services/**: Business logic and external service integrations
  - **workflows/**: Workflow orchestration logic
  - **models/**: Database models and schemas
  - **utils/**: Utility functions and helpers

### Frontend (`frontend/`)
- **app/**: Next.js 14+ app directory structure
  - **components/**: Reusable React components
  - **page.tsx**: Page components
- **lib/**: Frontend utility functions
- **public/**: Static assets (images, fonts, etc.)

### Documentation (`docs/`)
- **guides/**: User guides, tutorials, and how-to documents
- **architecture/**: System architecture and design documents
- **api/**: API documentation
- **agents/**: Agent-specific documentation
- **deployment/**: Deployment guides and checklists
- **test-results/**: Test execution reports

### Scripts (`scripts/`)
All utility scripts for development, monitoring, and deployment:
- Setup and initialization scripts
- Monitoring and logging scripts
- Watch scripts for development

### Configuration (`config/`)
Shared configuration files and templates:
- Environment variable templates
- Docker configuration templates

### Tests (`tests/`)
Integration and end-to-end tests that span multiple components.

## File Naming Conventions

- **Python files**: `snake_case.py`
- **TypeScript/React files**: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- **Configuration files**: `kebab-case.yml` or `UPPER_CASE.env`
- **Documentation**: `UPPER_CASE.md` for important docs, `kebab-case.md` for guides

## Ignored Directories

The following directories are gitignored and contain generated/runtime files:
- `node_modules/` - Node.js dependencies
- `venv/` - Python virtual environment
- `__pycache__/` - Python bytecode cache
- `.next/` - Next.js build output
- `models/` - Trained model files
- `plots/` - Generated visualization files
- `results/` - Workflow execution results
- `notebooks/` - Generated Jupyter notebooks

## Best Practices

1. **Keep root directory clean**: Only essential files at the root level
2. **Organize by feature**: Group related files together
3. **Document structure**: Update this file when adding new directories
4. **Use consistent naming**: Follow conventions for file and directory names
5. **Separate concerns**: Keep configuration, scripts, and code separate

