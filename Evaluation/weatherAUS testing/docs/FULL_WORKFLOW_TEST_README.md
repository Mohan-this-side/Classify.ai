# Full Workflow Test - weatherAUS Dataset

## Overview

This comprehensive test validates the **complete end-to-end workflow** from data ingestion to final model metrics.

## What It Tests

### 1. **Data Ingestion** ✅
- Loads weatherAUS.csv dataset
- Samples to 2000 rows for faster testing
- Validates target column (RainTomorrow)
- Checks data distribution

### 2. **State Initialization** ✅
- Creates ClassificationState object
- Initializes all required state keys
- Sets up dataset references

### 3. **Agent Execution Sequence** (7 Agents)

#### Agent 1: Data Discovery
- **Layer 1**: Statistical profiling, dataset structure analysis
- **Layer 2**: LLM-generated code → Docker execution → Results
- **Output**: Dataset metadata, similar datasets, research insights

#### Agent 2: EDA Analysis
- **Layer 1**: Basic statistical analysis, distributions
- **Layer 2**: LLM-generated EDA code → Docker execution → Results
- **Output**: EDA plots, statistical summary, correlation matrix

#### Agent 3: Data Cleaning
- **Layer 1**: Missing value detection, data quality assessment
- **Layer 2**: LLM-generated cleaning code → Docker execution → Results
- **Output**: Cleaned dataset, cleaning summary, quality score
- **Critical**: Updates `processed_dataset` in state for next agents

#### Agent 4: Feature Engineering
- **Layer 1**: Basic feature transformations
- **Layer 2**: LLM-generated feature engineering code → Docker execution → Results
- **Output**: Engineered features, feature importance, transformations
- **Input**: Uses cleaned dataset from Agent 3

#### Agent 5: ML Building
- **Layer 1**: Model selection logic, hyperparameter defaults
- **Layer 2**: LLM-generated model training code → Docker execution → Results
- **Output**: Best model, training metrics, cross-validation scores
- **Input**: Uses engineered features from Agent 4

#### Agent 6: Model Evaluation
- **Layer 1**: Basic evaluation metrics calculation
- **Layer 2**: LLM-generated evaluation code → Docker execution → Results
- **Output**: Evaluation metrics, confusion matrix, ROC curve, precision-recall
- **Input**: Uses trained model from Agent 5

#### Agent 7: Technical Reporter
- **Layer 1**: Report template generation
- **Layer 2**: LLM-generated comprehensive report → Docker execution → Results
- **Output**: Final report, executive summary, technical documentation
- **Input**: Uses all previous agent outputs

### 4. **Information Flow Verification**
- Tracks state keys added by each agent
- Verifies dataset passing (cleaned_dataset → processed_dataset)
- Checks Layer 2 results extraction from Docker
- Validates state updates between agents

### 5. **Docker Execution Tracking**
- Monitors container creation for each Layer 2 execution
- Tracks execution status (SUCCESS/FAILED)
- Records execution times
- Verifies sandbox results retrieval

### 6. **Final Outputs Verification**
- ✅ Cleaned dataset available
- ✅ EDA results and plots
- ✅ Engineered features
- ✅ Trained model
- ✅ Evaluation metrics
- ✅ Final report
- ✅ Plots generated

### 7. **Final Metrics Extraction**
- Model evaluation metrics (accuracy, precision, recall, F1)
- Training metrics
- Cross-validation scores
- Feature importance
- Data quality score
- Workflow status

## Running the Test

### Start the Test
```bash
cd /Users/mohan/NEU/FALL\ 2025/AGENTS\ V1/ds-capstone-project
source venv/bin/activate
python Evaluation/test_full_workflow_weatherAUS.py
```

### Monitor Progress
```bash
# Watch the log file
tail -f Evaluation/results/full_workflow_test.log

# Or use the monitor script
./Evaluation/monitor_full_workflow_test.sh
```

### Check Results
```bash
# View the comprehensive report
cat Evaluation/results/full_workflow_weatherAUS_report.json | jq .

# Check final summary
tail -100 Evaluation/results/full_workflow_test.log | grep "SUMMARY"
```

## Expected Duration

- **Data Ingestion**: ~1 second
- **State Initialization**: ~1 second
- **Each Agent**: ~30-60 seconds (Layer 1: ~1s, Layer 2: ~25-50s)
- **Total**: ~5-10 minutes for all 7 agents

## Output Files

1. **Log File**: `Evaluation/results/full_workflow_test.log`
   - Detailed execution logs
   - Agent-by-agent progress
   - Docker execution details

2. **Report File**: `Evaluation/results/full_workflow_weatherAUS_report.json`
   - Comprehensive test report
   - Agent execution summaries
   - Docker execution tracking
   - State snapshots
   - Final metrics

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
- Check log file for errors: `tail -f Evaluation/results/full_workflow_test.log`

### Docker Failures
- Verify Docker daemon: `docker ps`
- Check sandbox volumes: `docker volume ls | grep sandbox`
- Check container logs: `docker logs <container_name>`

### Layer 2 Failures
- Check LLM service is accessible
- Verify code validation passes
- Check sandbox executor configuration

## Next Steps

After test completion:
1. Review the comprehensive report
2. Check final metrics in the report
3. Verify all outputs are generated
4. Review any errors or warnings
5. Use results for evaluation documentation

