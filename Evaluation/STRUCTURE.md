# Evaluation Folder Structure

## Overview

The Evaluation folder has been reorganized for clarity and maintainability.

## Main Structure

```
Evaluation/
├── config/                    # Configuration files
├── datasets/                  # Test datasets (real-world + synthetic)
├── metrics/                   # Quality metrics and LLM judge
├── orchestration/             # Workflow evaluation orchestration
├── reports/                   # Report generation modules
├── results/                   # General evaluation results
├── test_cases/                # Test case definitions
├── visualization/             # Plot and diagram generators
├── weatherAUS testing/        # WeatherAUS-specific testing (isolated)
├── archive/                   # Archived temporary/test files
└── logs/                      # General evaluation logs
```

## WeatherAUS Testing Folder

All weatherAUS-specific testing has been moved to `weatherAUS testing/`:

```
weatherAUS testing/
├── scripts/                   # Test scripts
│   ├── test_full_workflow_weatherAUS.py      # Main comprehensive test
│   ├── test_weatherAUS_full_flow.py          # Full flow test
│   ├── test_weatherAUS_with_docker_visibility.py  # Docker visibility test
│   └── monitor_full_workflow_test.sh          # Monitoring script
├── logs/                      # Execution logs
│   ├── full_workflow_test.log
│   ├── full_workflow_test_run2.log
│   └── weatherAUS_*.log
├── results/                   # Test results
│   ├── plots/                 # Generated visualizations
│   ├── tables/                # Generated tables
│   └── reports/               # JSON reports
└── docs/                      # Documentation
    ├── README.md              # Main documentation
    └── FULL_WORKFLOW_TEST_README.md
```

## Cleanup Actions Taken

1. ✅ Created `weatherAUS testing/` folder with organized subfolders
2. ✅ Moved all weatherAUS test scripts to `weatherAUS testing/scripts/`
3. ✅ Moved all weatherAUS logs to `weatherAUS testing/logs/`
4. ✅ Moved all weatherAUS results to `weatherAUS testing/results/`
5. ✅ Moved weatherAUS documentation to `weatherAUS testing/docs/`
6. ✅ Archived temporary/test files to `archive/temp_files/`
7. ✅ Moved old logs to `logs/old/`
8. ✅ Updated script paths in moved files
9. ✅ Created comprehensive README files

## Archived Files

Temporary and test files moved to `archive/temp_files/`:
- `analyze_and_fix_results.py`
- `check_evaluation_status.py`
- `compare_evaluations.py`
- `test_imports.py`
- `test_full_layer2_flow.py`

## Running WeatherAUS Tests

From the Evaluation folder:
```bash
cd Evaluation
source ../venv/bin/activate
python "weatherAUS testing/scripts/test_full_workflow_weatherAUS.py"
```

Or use the monitoring script:
```bash
./weatherAUS\ testing/scripts/monitor_full_workflow_test.sh
```

## General Evaluation Scripts

Main evaluation scripts remain in the root Evaluation folder:
- `run_comprehensive_evaluation.py` - Full evaluation
- `run_quick_evaluation.py` - Quick evaluation
- `run_single_dataset_evaluation.py` - Single dataset
- `run_efficient_evaluation.py` - Efficient evaluation
- `run_final_evaluation.py` - Final evaluation

## Results Organization

- **General results**: `results/` folder
  - `agent_level/` - Agent-level test results
  - `system_level/` - System-level test results
  - `tables/` - Generated tables
  - `visualization/` - Generated plots

- **WeatherAUS results**: `weatherAUS testing/results/` folder
  - `plots/` - WeatherAUS-specific plots
  - `tables/` - WeatherAUS-specific tables
  - `reports/` - WeatherAUS-specific JSON reports

