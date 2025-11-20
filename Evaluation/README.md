# Evaluation Framework

Comprehensive evaluation framework for the Classify AI multi-agent ML pipeline system.

## Folder Structure

```
Evaluation/
├── config/                 # Configuration files
│   └── evaluation_config.yaml
├── datasets/               # Test datasets
│   ├── real_world/        # Real-world datasets from Kaggle
│   ├── synthetic/         # Synthetically generated datasets
│   └── metadata/         # Dataset metadata
├── metrics/               # Quality metrics and LLM judge
├── orchestration/         # Workflow evaluation orchestration
├── reports/               # Report generation modules
├── results/               # Evaluation results (empty - results stored in weatherAUS testing/)
├── test_cases/            # Test case definitions
├── visualization/         # Plot and diagram generators
├── weatherAUS testing/   # Current testing framework (see subfolder README)
│   ├── scripts/          # Test scripts
│   ├── results/          # Test results, plots, and tables
│   ├── logs/             # Test execution logs
│   └── docs/             # Documentation
├── archive/               # Archived temporary files
└── logs/                  # General evaluation logs (old)
```

## Current Testing Framework

The main testing framework is located in `weatherAUS testing/`. This framework provides:

- **End-to-end workflow testing**: Tests the complete pipeline from data ingestion to final report generation
- **Agent-level evaluation**: Tests each agent's Layer 1 and Layer 2 execution
- **Docker sandbox integration**: Validates LLM-generated code execution in isolated Docker containers
- **Comprehensive reporting**: Generates detailed reports, plots, and tables
- **Problem-solving visualization**: Tracks how agents address dataset problems step-by-step

### Running the Test

```bash
cd "Evaluation/weatherAUS testing/scripts"
python test_full_workflow_weatherAUS.py
```

Or from the project root:

```bash
python "Evaluation/weatherAUS testing/scripts/test_full_workflow_weatherAUS.py"
```

### Test Outputs

All test results are stored in `weatherAUS testing/results/`:
- `reports/` - JSON reports with detailed execution data
- `plots/` - Visualizations (performance heatmaps, before/after comparisons, etc.)
- `tables/` - Markdown and LaTeX tables summarizing results

See `weatherAUS testing/README.md` for detailed documentation.

## Framework Components

### Test Cases (`test_cases/`)
- `base_test_framework.py`: Base class for all agent tests
- `agent_tests.py`: Specific test suites for each agent

### Metrics (`metrics/`)
- `quality_metrics.py`: Quality score calculations
- `llm_judge.py`: LLM-as-judge evaluation

### Visualization (`visualization/`)
- `plot_generator.py`: Plot generation utilities
- `flowchart_generator.py`: Architecture diagrams

### Reports (`reports/`)
- `report_generator.py`: Markdown report generation
- `scorecard_generator.py`: Agent scorecards
- `table_generator.py`: Table generation (Markdown/LaTeX)

## Configuration

Edit `config/evaluation_config.yaml` to configure:
- Kaggle API credentials
- Dataset specifications
- Quality thresholds
- Test parameters

## Notes

- Old evaluation scripts have been removed to keep the folder clean
- All current testing is done through the `weatherAUS testing/` framework
- Framework components can be reused for testing other datasets
