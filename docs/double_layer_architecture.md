# Double-Layer Architecture Documentation

## Overview

The DS Capstone Multi-Agent System now implements a **Double-Layer Architecture** that combines the reliability of hardcoded logic (Layer 1) with the flexibility of AI-generated code (Layer 2).

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT EXECUTION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  LAYER 1: Hardcoded Analysis (Always Runs)  │          │
│  │  - Reliable, tested components               │          │
│  │  - Fast execution                            │          │
│  │  - Serves as fallback                        │          │
│  └──────────────────────────────────────────────┘          │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────┐          │
│  │  LAYER 2: LLM + Sandbox (Optional)          │          │
│  │  1. Generate code prompt from Layer 1        │          │
│  │  2. Call LLM to generate Python code         │          │
│  │  3. Validate code (security, syntax)         │          │
│  │  4. Execute in Docker sandbox                │          │
│  │  5. Process and validate results             │          │
│  └──────────────────────────────────────────────┘          │
│                        │                                     │
│           ┌────────────┴────────────┐                       │
│           │                         │                        │
│    SUCCESS (use Layer 2)     FAILURE (fallback)            │
│           │                         │                        │
│           └────────────┬────────────┘                       │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────┐          │
│  │        Return Results + Metadata             │          │
│  │  - Which layer was used                      │          │
│  │  - Layer 2 error (if any)                    │          │
│  │  - Execution times                           │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Why Double-Layer Architecture?

### Benefits

1. **Reliability**: Layer 1 always runs and provides a fallback, ensuring the workflow never fails
2. **Flexibility**: Layer 2 can adapt to unique data patterns and generate custom solutions
3. **Performance**: Layer 1 is fast; Layer 2 only runs when enabled
4. **Safety**: Layer 2 code is validated and executed in an isolated sandbox
5. **Educational**: Generated code can be inspected and learned from

### Use Cases

- **Layer 1 Only**: When you need fast, predictable results
- **Layer 1 + 2**: When you want the best possible results and have time for LLM calls
- **Automatic Fallback**: When Layer 2 fails (timeout, invalid code, etc.)

## Implementation Guide

### For Agent Developers

All agents must now inherit from `BaseAgent` and implement three abstract methods:

#### 1. `perform_layer1_analysis(state: ClassificationState) -> Dict[str, Any]`

**Purpose**: Implement reliable, hardcoded analysis that always works.

**Example (Data Cleaning Agent)**:
```python
def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
    """Layer 1: Reliable data cleaning using tested components"""

    df = state["dataset"]

    # Use existing components
    missing_analysis = self.missing_value_analyzer.analyze(df)
    outlier_analysis = self.outlier_detector.detect(df)

    # Perform basic cleaning
    df_cleaned = self.missing_value_imputer.impute(df)

    return {
        "cleaned_data": df_cleaned,
        "missing_values": missing_analysis,
        "outliers": outlier_analysis,
        "data_quality_score": self._calculate_quality_score(df_cleaned)
    }
```

#### 2. `generate_layer2_code(layer1_results: Dict, state: ClassificationState) -> str`

**Purpose**: Generate a detailed prompt for LLM code generation based on Layer 1 insights.

**Example (Data Cleaning Agent)**:
```python
def generate_layer2_code(self, layer1_results: Dict, state: ClassificationState) -> str:
    """Layer 2: Generate prompt for advanced cleaning code"""

    prompt = f"""
    Generate Python code to perform advanced data cleaning based on this analysis:

    ## Layer 1 Analysis Results:
    - Missing values: {layer1_results['missing_values']}
    - Outliers detected: {layer1_results['outliers']}
    - Data quality score: {layer1_results['data_quality_score']}

    ## Requirements:
    1. Use pandas and numpy
    2. Implement advanced imputation strategies (KNN, iterative)
    3. Handle outliers intelligently (don't just drop)
    4. Preserve data relationships
    5. Return cleaned DataFrame

    ## Code Structure:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer

    def clean_data(df):
        # Your advanced cleaning logic here
        return df_cleaned

    # Execute
    result = clean_data(df)
    ```

    Generate the complete code now.
    """

    return prompt
```

#### 3. `process_sandbox_results(sandbox_output: Dict, layer1_results: Dict, state: ClassificationState) -> Dict`

**Purpose**: Validate and process Layer 2 results. Optional - default implementation is provided.

**Example (Data Cleaning Agent)**:
```python
def process_sandbox_results(
    self,
    sandbox_output: Dict,
    layer1_results: Dict,
    state: ClassificationState
) -> Dict:
    """Validate that Layer 2 cleaning is better than Layer 1"""

    # Call parent validation
    results = super().process_sandbox_results(sandbox_output, layer1_results, state)

    # Custom validation
    df_cleaned = results.get("cleaned_data")

    if df_cleaned is None:
        raise ValueError("No cleaned data returned from Layer 2")

    # Calculate Layer 2 quality score
    layer2_quality = self._calculate_quality_score(df_cleaned)
    layer1_quality = layer1_results["data_quality_score"]

    # Layer 2 must be better than Layer 1
    if layer2_quality < layer1_quality:
        raise ValueError(
            f"Layer 2 quality ({layer2_quality}) worse than Layer 1 ({layer1_quality})"
        )

    self.logger.info(f"Layer 2 improved quality by {layer2_quality - layer1_quality:.2f}%")

    return results
```

### BaseAgent Constructor Parameters

```python
BaseAgent(
    agent_name: str,                    # Required: agent name
    agent_version: str = "1.0.0",       # Agent version
    enable_layer2: bool = True,         # Enable/disable Layer 2
    sandbox_timeout: int = 120,         # Sandbox timeout in seconds
    sandbox_memory_limit: str = "2g",   # Memory limit for sandbox
    sandbox_cpu_limit: float = 1.5      # CPU limit for sandbox
)
```

### Example: Complete Agent Implementation

```python
from app.agents.base_agent import BaseAgent
from app.workflows.state_management import ClassificationState

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="custom_agent",
            agent_version="1.0.0",
            enable_layer2=True,
            sandbox_timeout=180  # 3 minutes for complex operations
        )

    def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """Implement reliable Layer 1 logic"""
        # Your hardcoded analysis here
        return {"result": "layer1_analysis"}

    def generate_layer2_code(self, layer1_results: Dict, state: ClassificationState) -> str:
        """Generate LLM prompt"""
        return f"Generate code based on: {layer1_results}"

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "supports_layer2": True
        }
```

## Security Features

### CodeValidator

The `CodeValidator` service scans all generated code before execution:

**Blocked Operations:**
- `os.system`, `subprocess` (system commands)
- `eval`, `exec`, `compile` (dynamic code execution)
- `open`, `file` (unrestricted file access)
- Network operations (`socket`, `urllib`, `requests`)
- Unauthorized imports (only ML/data science libraries allowed)

**Allowed Libraries:**
- Data: `pandas`, `numpy`, `scipy`
- ML: `sklearn`, `xgboost`, `lightgbm`, `catboost`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Utils: `json`, `datetime`, `collections`, `joblib`

### SandboxExecutor

All Layer 2 code runs in an isolated Docker container:

**Security Features:**
- No network access (`--network none`)
- Read-only filesystem
- Memory limits enforced
- CPU limits enforced
- Execution timeout
- No privilege escalation

**Resource Limits (Default):**
- Timeout: 120 seconds
- Memory: 2GB
- CPU: 1.5 cores

## Monitoring and Debugging

### Layer Usage Tracking

The system tracks which layer was used for each agent:

```python
# Check which layer was used
layer_used = state["layer_usage"]["data_cleaning"]  # "layer1" or "layer2"

# Get detailed results
agent_result = state["agent_results"]["data_cleaning"]
print(f"Layer used: {agent_result['layer_used']}")
print(f"Layer 2 attempted: {agent_result['layer2_attempted']}")
print(f"Layer 2 error: {agent_result['layer2_error']}")
```

### Logging

All layer operations are logged:

```
INFO - data_cleaning - Starting double-layer execution for data_cleaning
INFO - data_cleaning - Executing Layer 1 (hardcoded analysis)...
INFO - data_cleaning - PERFORMANCE: Layer 1 execution took 2.35s
INFO - data_cleaning - Attempting Layer 2 (LLM + sandbox)...
INFO - data_cleaning - Generating Layer 2 code prompt...
INFO - data_cleaning - Calling LLM to generate code...
INFO - data_cleaning - LLM generated 1542 characters of code
INFO - data_cleaning - Validating generated code...
INFO - data_cleaning - Executing code in sandbox...
INFO - data_cleaning - Sandbox execution completed: SUCCESS
INFO - data_cleaning - Processing sandbox results...
INFO - data_cleaning - PERFORMANCE: Layer 2 execution took 15.72s
INFO - data_cleaning - Layer 2 execution successful, using Layer 2 results
```

### Performance Metrics

Performance is automatically tracked:

```python
# Access timing information
execution_time = state["agent_execution_times"]["data_cleaning"]
print(f"Total execution time: {execution_time}s")

# Layer-specific metrics are logged
# Check logs for "PERFORMANCE:" messages
```

## Configuration

### Disable Layer 2 Globally

```python
agent = DataCleaningAgent()
agent.enable_layer2 = False  # Use Layer 1 only
```

### Adjust Sandbox Resources

```python
agent = DataCleaningAgent(
    sandbox_timeout=300,        # 5 minutes
    sandbox_memory_limit="4g",  # 4GB RAM
    sandbox_cpu_limit=2.0       # 2 CPU cores
)
```

### Environment Variables

Add to `.env` file:

```bash
# LLM Configuration
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional fallback

# Sandbox Configuration
SANDBOX_TIMEOUT=120
SANDBOX_MEMORY_LIMIT=2g
SANDBOX_CPU_LIMIT=1.5

# Layer 2 Feature Flag
ENABLE_LAYER2=true
```

## Troubleshooting

### Layer 2 Always Falls Back to Layer 1

**Possible causes:**
1. LLM API key not configured
2. Sandbox Docker image not built
3. Code validation failures
4. Sandbox execution timeouts

**Solutions:**
1. Check `.env` file has `GOOGLE_API_KEY`
2. Build sandbox: `docker build -t ds-capstone-ml-sandbox .`
3. Review validation logs for specific errors
4. Increase timeout if operations are complex

### Code Validation Errors

**Common issues:**
- Using unauthorized imports
- Attempting file/network operations
- Syntax errors in generated code

**Solutions:**
- Review `CodeValidator` allowed imports list
- Check LLM prompt is clear about constraints
- Add error handling to LLM prompt template

### Sandbox Timeouts

**Causes:**
- Complex computations (large datasets, deep models)
- Infinite loops in generated code
- Resource-intensive operations

**Solutions:**
- Increase `sandbox_timeout` parameter
- Add complexity hints to LLM prompt
- Implement Layer 1 preprocessing to reduce data size

## Best Practices

### 1. Always Ensure Layer 1 Works Perfectly

Layer 1 is your fallback. It must be reliable, fast, and well-tested.

### 2. Provide Rich Context to Layer 2

The more information you include in `generate_layer2_code()`, the better the LLM-generated code will be.

### 3. Validate Layer 2 Results

Don't blindly trust Layer 2. Implement `process_sandbox_results()` to verify results are valid and better than Layer 1.

### 4. Monitor Layer Usage

Track how often Layer 2 succeeds vs. falls back. High fallback rates indicate prompt or validation issues.

### 5. Optimize for Your Use Case

- Fast operations: Short timeout, low resources
- Complex ML: Long timeout, high resources
- Production: Disable Layer 2 for consistency
- Experimentation: Enable Layer 2 for innovation

## Next Steps

1. **Implement agents**: Use this guide to update existing agents with double-layer architecture
2. **Test thoroughly**: Verify both layers work independently and together
3. **Monitor in production**: Track layer usage and performance
4. **Iterate prompts**: Improve LLM prompts based on Layer 2 success rates
5. **Add features**: Extend with custom validation, caching, or optimization

---

**Version**: 1.0
**Last Updated**: 2025-10-27
**Author**: DS Capstone Team
