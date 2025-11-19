# Comprehensive Evaluation Framework

This evaluation framework provides rigorous, reproducible testing for the Classify AI multi-agent ML automation system. It validates agent correctness, detects "cheating" behaviors, and provides quantitative evidence of intelligent decision-making.

## Overview

The evaluation framework tests all 8 agents at both **agent-level** and **system-level**, using:
- **6 real-world datasets** from Kaggle
- **4 synthetic datasets** with known issues for edge case testing
- **Comprehensive metrics** for each agent
- **LLM-as-judge** evaluation for qualitative assessments
- **Automated visualizations** and reports

## Directory Structure

```
Evaluation/
├── config/
│   └── evaluation_config.yaml      # Central configuration
├── datasets/
│   ├── real_world/                  # Kaggle datasets
│   ├── synthetic/                   # Generated test datasets
│   └── metadata/                    # Dataset metadata JSON files
├── test_cases/
│   ├── base_test_framework.py       # Base test framework
│   └── agent_tests.py               # All agent test suites
├── metrics/
│   ├── quality_metrics.py           # Metric calculators
│   └── llm_judge.py                 # LLM-as-judge evaluator
├── orchestration/
│   ├── workflow_evaluator.py        # System-level tests
│   └── test_runner.py               # Test execution engine
├── visualization/
│   ├── plot_generator.py             # Automated plots
│   └── flowchart_generator.py       # Diagrams and flowcharts
├── reports/
│   ├── report_generator.py           # Report generation
│   └── scorecard_generator.py        # Agent scorecards
├── results/
│   ├── agent_level/                  # Agent test results
│   ├── system_level/                 # Workflow results
│   └── visualization/                # Generated plots/diagrams
├── run_full_evaluation.py            # Main orchestration script
└── compare_evaluations.py            # Comparison tool
```

## Quick Start

### 1. Install Dependencies

```bash
# Install required Python packages
pip install kaggle pandas numpy scikit-learn matplotlib seaborn graphviz pyyaml psutil

# Install graphviz system package (for diagrams)
# macOS:
brew install graphviz
# Ubuntu/Debian:
sudo apt-get install graphviz
```

### 2. Configure Kaggle API

The Kaggle API credentials are already configured in `config/evaluation_config.yaml`. If you need to update them:

1. Get your Kaggle API credentials from https://www.kaggle.com/account
2. Update `config/evaluation_config.yaml` with your username and key

### 3. Run Full Evaluation

```bash
# Run complete evaluation
python Evaluation/run_full_evaluation.py
```

This will:
1. Download 6 real-world datasets from Kaggle
2. Generate 4 synthetic datasets with known issues
3. Generate metadata for all datasets
4. Run agent-level tests on all 8 agents
5. Run system-level workflow tests
6. Generate visualizations (plots and flowcharts)
7. Generate comprehensive reports

### 4. View Results

Results are saved to:
- **Agent Results:** `Evaluation/results/agent_level/all_agent_tests.json`
- **Workflow Results:** `Evaluation/results/system_level/workflow_evaluation.json`
- **Report:** `Evaluation/reports/evaluation_report.md`
- **Visualizations:** `Evaluation/results/visualization/`

## Datasets

### Real-World Datasets (from Kaggle)

1. **Titanic** - Missing values, mixed types, binary classification
2. **Heart Disease UCI** - Medical data, binary classification
3. **Bank Marketing** - Class imbalance (~88:12), many features
4. **Credit Card Fraud** - Severe imbalance (99.8:0.2), PCA features
5. **Adult Income** - Mixed categorical/numeric, binary classification
6. **Wine Quality** - Multiclass, numeric features

### Synthetic Datasets (with Known Issues)

1. **Perfect Leakage** - Target duplicated as feature (tests anti-cheating)
2. **Severe Imbalance** - 99:1 ratio (tests imbalance detection)
3. **Multicollinearity** - High correlation features (tests feature engineering)
4. **High Dimensionality** - 200 features, 100 samples (tests curse of dimensionality)

## Agent Evaluation

Each agent is evaluated on:

### Data Discovery Agent
- Type detection accuracy
- Target suggestion relevance
- Edge cases: mixed types, empty columns

### EDA Agent
- Imbalance detection rate (must detect >70:30 ratio)
- Missing value detection accuracy
- Correlation detection completeness
- **Critical:** Must flag severe imbalance, not celebrate high accuracy

### Data Cleaning Agent
- Data quality improvement
- Zero-variance feature removal
- Imputation appropriateness (LLM-as-judge)

### Feature Engineering Agent
- Feature usefulness (correlation with target)
- Multicollinearity reduction
- Encoding appropriateness (LLM-as-judge)
- Layer 2 success rate

### ML Builder Agent
- Algorithm selection appropriateness (LLM-as-judge)
- Class balancing application (when needed)
- Model diversity
- **Anti-cheating score:** Checks specificity/sensitivity, not just accuracy

### Model Evaluation Agent
- Metric completeness
- Imbalance awareness (flags high accuracy with low minority recall)
- Overfitting detection

### Technical Reporter Agent
- Report completeness
- Visualization quality
- Explanation clarity (LLM-as-judge)

### Project Manager Agent
- Explanation accuracy (LLM-as-judge, target ≥85%)
- Educational effectiveness
- Decision reasoning quality

## Quality Thresholds

Quality thresholds are defined in `config/evaluation_config.yaml`:

- **Layer 2 Success Rate:** ≥80%
- **PM Accuracy:** ≥85% (normalized from 1-5 scale)
- **Imbalance Detection Rate:** ≥95%
- **Type Detection Accuracy:** ≥90%
- And more...

## Key Features

### Anti-Cheating Tests

The framework includes specific tests to detect "cheating" behaviors:

1. **Perfect Leakage Dataset:** Agents must flag suspiciously high accuracy, not celebrate it
2. **Severe Imbalance Dataset:** Must detect imbalance and check minority class recall
3. **99% Accuracy Trap:** Must flag when accuracy is misleading (e.g., always predicting majority class)

### LLM-as-Judge Evaluation

For qualitative assessments, the framework uses LLM-as-judge to evaluate:
- Explanation accuracy
- Educational effectiveness
- Decision reasoning quality
- Imputation/encoding appropriateness

### Reproducibility

- Fixed random seeds
- Versioned results with timestamps
- JSON output for programmatic consumption
- Comparison tool for tracking improvements

## Comparing Evaluations

To compare current results with a previous baseline:

```bash
python Evaluation/compare_evaluations.py <baseline_results_path>
```

This generates a comparison report showing:
- Improvements (agents that got better)
- Regressions (agents that got worse)
- System-level metric changes

## Interpreting Results

### Agent Scorecards

Each agent receives a grade (A/B/C/D/F) based on:
- Pass rate across all datasets
- Key metric performance vs thresholds
- Edge case handling

### System-Level Metrics

- **Success Rate:** Percentage of workflows that completed successfully
- **Layer 2 Success Rate:** Percentage of successful LLM code generation attempts
- **Execution Time:** Average workflow execution time
- **Memory Usage:** Resource consumption

### Visualizations

The framework generates:
- **Agent Scorecard Heatmap:** Pass/fail for each agent per dataset
- **Metric Dashboard:** Key metrics across agents
- **Layer Comparison:** Layer 1 vs Layer 2 success rates
- **Failure Analysis:** Pareto chart of failure modes
- **Execution Timeline:** Workflow execution times
- **Quality Score Distribution:** Histogram of quality scores
- **Workflow State Diagram:** LangGraph state transitions
- **Architecture Diagram:** System components and data flow

## Customization

### Adding New Datasets

1. Add dataset specification to `config/evaluation_config.yaml`:
```yaml
real_world_datasets:
  - name: "new_dataset"
    kaggle_id: "user/dataset-name"
    expected_issues: [...]
    target_column: "target"
```

2. Run evaluation - it will automatically download and test

### Adding New Test Cases

1. Add test function to `test_cases/agent_tests.py`
2. Add metric calculator to `metrics/quality_metrics.py` if needed
3. Update thresholds in `config/evaluation_config.yaml`

### Modifying Thresholds

Edit `config/evaluation_config.yaml` to adjust quality thresholds for each agent.

## Troubleshooting

### Kaggle API Errors

If dataset downloads fail:
1. Verify Kaggle credentials in `config/evaluation_config.yaml`
2. Check internet connection
3. Ensure Kaggle API is installed: `pip install kaggle`

### LLM Service Errors

If LLM-as-judge evaluation fails:
1. Check that LLM service is configured in backend
2. Verify API keys are set
3. The framework will use default scores if LLM is unavailable

### Import Errors

If you get import errors:
1. Ensure you're running from the project root
2. Check that backend is in Python path
3. Install all dependencies: `pip install -r backend/requirements.txt`

## Output Files

After running evaluation, you'll find:

- `results/agent_level/all_agent_tests.json` - All agent test results
- `results/system_level/workflow_evaluation.json` - System-level results
- `reports/evaluation_report.md` - Comprehensive markdown report
- `results/visualization/plots/` - All generated plots (PNG)
- `results/visualization/diagrams/` - Flowcharts and diagrams (PNG)
- `evaluation.log` - Detailed execution log

## Success Criteria

The evaluation framework validates:

✅ All 8 agents have quantifiable quality metrics  
✅ Edge cases (leakage, imbalance, cheating) are detected  
✅ Layer 2 success rate ≥80% is measurable  
✅ PM Accuracy ≥85% is validated via LLM-as-judge  
✅ Anti-cheating tests pass  
✅ Reproducible: Same input = Same output  
✅ Fast: Full evaluation runs in <30 minutes  
✅ Comprehensive: All PRD requirements covered  
✅ Visual: Plots, flowcharts, and reports generated  
✅ Actionable: Clear pass/fail with improvement recommendations  

## Contributing

When adding new features:

1. Follow the existing structure
2. Add tests for new functionality
3. Update this README
4. Ensure reproducibility (fixed seeds, deterministic)

## License

Same as main project.

## Contact

For questions or issues, refer to the main project documentation.

