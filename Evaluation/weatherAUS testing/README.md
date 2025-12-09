# WeatherAUS Dataset Testing

This folder contains all scripts, logs, results, and documentation related to testing the weatherAUS dataset.

## Folder Structure

```
weatherAUS testing/
├── scripts/          # Test scripts for weatherAUS dataset
├── logs/            # Execution logs
├── results/         # Test results
│   ├── plots/      # Generated visualizations
│   ├── tables/     # Generated tables
│   └── reports/    # JSON reports
└── docs/           # Documentation
```

## Test Scripts

### `test_full_workflow_weatherAUS.py`
**Main comprehensive test script** - Tests the complete end-to-end workflow:
- Data ingestion
- All 7 agents in sequence (Data Discovery → EDA → Data Cleaning → Feature Engineering → ML Building → Model Evaluation → Technical Reporter)
- Layer 1 and Layer 2 execution
- Docker sandbox execution
- Information passing between agents
- Final report, plots, and tables generation

**Usage:**
```bash
cd Evaluation
source ../venv/bin/activate
python "weatherAUS testing/scripts/test_full_workflow_weatherAUS.py"
```

### `test_weatherAUS_full_flow.py`
Tests the full Layer 1 → Layer 2 → Docker flow with enhanced visibility.

### `test_weatherAUS_with_docker_visibility.py`
Tests with detailed Docker execution visibility and logging.

### `monitor_full_workflow_test.sh`
Monitoring script to watch test progress in real-time.

**Usage:**
```bash
./weatherAUS\ testing/scripts/monitor_full_workflow_test.sh
```

## Results

### Reports
- `results/reports/full_workflow_weatherAUS_report.json` - Comprehensive test report
- `results/reports/weatherAUS_docker_visibility_results.json` - Docker visibility results
- `results/reports/weatherAUS_full_flow_results.json` - Full flow test results

### Plots
- `results/plots/agent_performance_heatmap.png` - Agent performance heatmap
- `results/plots/execution_timeline.png` - Execution timeline visualization
- `results/plots/layer_comparison.png` - Layer 1 vs Layer 2 comparison

### Tables
- `results/tables/agent_performance_table.md` - Agent performance summary (Markdown)
- `results/tables/agent_performance_table.tex` - Agent performance summary (LaTeX)

## Logs

All execution logs are stored in `logs/` directory:
- `full_workflow_test.log` - Main test execution log
- `weatherAUS_docker_visibility.log` - Docker visibility test log
- `weatherAUS_full_flow.log` - Full flow test log
- And other related logs

## Documentation

- `docs/FULL_WORKFLOW_TEST_README.md` - Detailed documentation about the full workflow test

## Dataset

The weatherAUS dataset is located at:
```
Evaluation/datasets/real_world/weatherAUS.csv
```

**Dataset Info:**
- Target column: `RainTomorrow`
- Original size: 145,460 rows × 23 columns
- Test sampling: 2,000 rows (for faster testing)

## Test Coverage

The comprehensive test validates:
1. ✅ Data ingestion and state initialization
2. ✅ All 7 agents execute in sequence
3. ✅ Layer 1 execution (hardcoded analysis)
4. ✅ Layer 2 execution (LLM code generation → Docker sandbox)
5. ✅ Information passing between agents
6. ✅ Final outputs generation (cleaned dataset, features, model, metrics, report)
7. ✅ Plots and tables generation

## Expected Duration

- **Full workflow test**: ~5-10 minutes for all 7 agents
- Each agent: ~30-60 seconds (Layer 1: ~1s, Layer 2: ~25-50s)

## Success Criteria

✅ All 7 agents execute successfully  
✅ Layer 1 executes for all agents  
✅ Layer 2 executes in Docker for all agents  
✅ Docker containers created and executed successfully  
✅ Information passes correctly between agents  
✅ Final outputs generated (dataset, features, model, metrics, report)  
✅ No critical errors  

## Troubleshooting

### Test Hangs
- Check Docker is running: `docker ps`
- Check LLM API key is set
- Check log file for errors: `tail -f logs/full_workflow_test.log`

### Docker Failures
- Verify Docker daemon: `docker ps`
- Check sandbox volumes: `docker volume ls | grep sandbox`
- Check container logs: `docker logs <container_name>`

### Layer 2 Failures
- Check LLM service is accessible
- Verify code validation passes
- Check sandbox executor configuration

