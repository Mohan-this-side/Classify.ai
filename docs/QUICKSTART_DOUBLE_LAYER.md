# Double-Layer Architecture Quick Start Guide

## For Agent Developers: 3-Step Implementation

### Step 1: Update Your Agent Constructor

```python
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="my_agent",
            agent_version="1.0.0",
            enable_layer2=True,         # Enable Layer 2
            sandbox_timeout=120,        # 2 minutes timeout
            sandbox_memory_limit="2g",  # 2GB RAM
            sandbox_cpu_limit=1.5       # 1.5 CPU cores
        )
        # Your existing initialization here
```

### Step 2: Implement Layer 1 (Required)

```python
def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
    """
    Layer 1: Reliable, fast, hardcoded analysis
    This is your fallback - must always work!
    """
    # Use your existing tested components
    df = state["dataset"]

    # Perform analysis
    results = {
        "processed_data": df,
        "metrics": {...},
        "issues_found": [...]
    }

    return results
```

### Step 3: Implement Layer 2 Prompt (Required)

```python
def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
    """
    Layer 2: Generate prompt for LLM to create adaptive code
    """
    prompt = f"""
    Generate Python code to improve this analysis:

    ## Layer 1 Results:
    {json.dumps(layer1_results, indent=2)}

    ## Requirements:
    - Use pandas, numpy, sklearn
    - Improve upon Layer 1 results
    - Return results as dictionary

    ## Code Template:
    ```python
    import pandas as pd
    import numpy as np

    def improve_analysis(data):
        # Your generated code here
        return improved_results
    ```

    Generate the complete code now.
    """
    return prompt
```

### Step 4: Validate Results (Optional)

```python
def process_sandbox_results(
    self,
    sandbox_output: Dict[str, Any],
    layer1_results: Dict[str, Any],
    state: ClassificationState
) -> Dict[str, Any]:
    """
    Optional: Validate that Layer 2 is better than Layer 1
    """
    # Call parent validation first
    results = super().process_sandbox_results(sandbox_output, layer1_results, state)

    # Your custom validation
    if results["quality_score"] < layer1_results["quality_score"]:
        raise ValueError("Layer 2 quality worse than Layer 1")

    return results
```

## That's It!

Your agent now supports:
- ✅ Reliable Layer 1 fallback
- ✅ Adaptive Layer 2 LLM generation
- ✅ Automatic fallback on failures
- ✅ Secure sandbox execution
- ✅ Comprehensive logging

## Testing Your Agent

```python
import asyncio
from app.workflows.state_management import create_initial_state

async def test_agent():
    # Create agent
    agent = MyAgent()

    # Create state
    state = create_initial_state(
        session_id="test",
        dataset_id="test_dataset"
    )

    # Execute
    result_state = await agent.execute(state)

    # Check which layer was used
    layer_used = result_state["layer_usage"]["my_agent"]
    print(f"Layer used: {layer_used}")

# Run test
asyncio.run(test_agent())
```

## Troubleshooting

### Layer 2 Always Falls Back to Layer 1?

1. **Check API Key**: Ensure `GOOGLE_API_KEY` is in `.env`
2. **Check Docker**: Ensure Docker is running
3. **Check Logs**: Look for specific error messages
4. **Test Services**:
   ```python
   agent = MyAgent()
   print(f"Layer 2 enabled: {agent.enable_layer2}")
   print(f"Can use Layer 2: {agent._can_use_layer2()}")
   ```

### Code Validation Fails?

Common issues:
- Using unauthorized imports
- Attempting file/network operations
- Syntax errors in generated code

**Fix**: Update your prompt to be more specific about constraints.

### Sandbox Timeout?

**Solutions**:
- Increase timeout: `sandbox_timeout=300` (5 minutes)
- Simplify Layer 2 operations
- Use Layer 1 for preprocessing

## Configuration Examples

### Fast Operations (Simple Analysis)
```python
super().__init__(
    agent_name="fast_agent",
    sandbox_timeout=60,         # 1 minute
    sandbox_memory_limit="1g",  # 1GB
    sandbox_cpu_limit=1.0
)
```

### Heavy Operations (ML Training)
```python
super().__init__(
    agent_name="ml_agent",
    sandbox_timeout=600,        # 10 minutes
    sandbox_memory_limit="8g",  # 8GB
    sandbox_cpu_limit=4.0
)
```

### Production (Layer 1 Only)
```python
super().__init__(
    agent_name="prod_agent",
    enable_layer2=False  # Disable Layer 2 for consistency
)
```

## Need More Details?

See full documentation: `docs/double_layer_architecture.md`

---

**Quick Ref Version**: 1.0
**Last Updated**: 2025-10-27
