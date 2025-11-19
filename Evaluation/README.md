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
├── results/               # Evaluation results
│   ├── agent_level/      # Agent-level test results
│   ├── system_level/     # System-level test results
│   ├── tables/           # Generated tables
│   └── visualization/    # Generated plots
├── test_cases/            # Test case definitions
├── visualization/         # Plot and diagram generators
├── weatherAUS testing/    # WeatherAUS-specific testing (see subfolder README)
├── archive/               # Archived temporary files
└── logs/                  # General evaluation logs
```

## Main Evaluation Scripts

### Comprehensive Evaluation
```bash
python run_comprehensive_evaluation.py
```
Runs full evaluation on all datasets with all agents.

### Quick Evaluation
```bash
python run_quick_evaluation.py
```
Fast evaluation for quick feedback.

### Single Dataset Evaluation
```bash
python run_single_dataset_evaluation.py
```
Evaluate on a single dataset.

### Efficient Evaluation
```bash
python run_efficient_evaluation.py
```
Balanced evaluation with dataset sampling.

## Test Framework

### Base Test Framework
`test_cases/base_test_framework.py` - Base class for all agent tests.

### Agent Tests
`test_cases/agent_tests.py` - Specific test suites for each agent.

## Configuration

Edit `config/evaluation_config.yaml` to configure:
- Kaggle API credentials
- Dataset specifications
- Quality thresholds
- Evaluation parameters

## Results

Results are organized by:
- **Agent-level**: Individual agent performance
- **System-level**: End-to-end workflow performance
- **Tables**: LaTeX and Markdown tables
- **Visualization**: Plots and diagrams

## WeatherAUS Testing

See `weatherAUS testing/README.md` for detailed information about weatherAUS-specific testing.

## Documentation

- `README.md` (this file) - Main evaluation framework documentation
- `weatherAUS testing/README.md` - WeatherAUS testing documentation
