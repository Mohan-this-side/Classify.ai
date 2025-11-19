# End-to-End Testing Plan

## Testing Objectives
1. Test with real-world classification dataset
2. Monitor all system components (frontend, backend, Docker)
3. Identify and fix any issues
4. Ensure robustness for various dataset types
5. Document findings in PRD_GAP_ANALYSIS.md

## Pre-Test Checklist

### System Status
- [x] Docker daemon running
- [x] Sandbox image built
- [x] Backend server running (port 8000)
- [x] Frontend server running (port 3001)
- [x] Log monitoring ready

### Resource Limits Check
- Current Docker sandbox limits:
  - Memory: 2GB
  - CPU: 1.5 cores
  - Timeout: 120 seconds
- Recommendation: Monitor during test, increase if needed

## Monitoring Points

### Frontend Monitoring
- File upload success/failure
- API calls and responses
- Error messages displayed to user
- UI state changes

### Backend Monitoring
- API endpoint calls
- Workflow execution
- Agent execution (Layer 1 and Layer 2)
- State management
- Error handling

### Docker Monitoring
- Container creation and execution
- Resource usage (CPU, memory)
- Execution time
- Sandbox errors
- Code execution results

## Expected Issues to Watch For

1. **Large Dataset Handling**
   - Memory limits exceeded
   - Timeout issues
   - File size limits

2. **Data Quality Issues**
   - Missing values
   - Invalid data types
   - Encoding problems
   - Special characters

3. **ML Model Issues**
   - Training failures
   - Memory overflow
   - Convergence problems

4. **Code Generation Issues**
   - LLM generates invalid code
   - Sandbox execution failures
   - Result parsing errors

## Testing Steps

1. User uploads dataset via frontend
2. Monitor backend logs for upload processing
3. Watch workflow execution through all agents
4. Monitor Docker sandbox for Layer 2 execution
5. Check for errors at each stage
6. Document issues in PRD_GAP_ANALYSIS.md
7. Fix issues as they occur
8. Re-test if needed

## Success Criteria

- Dataset uploads successfully
- All agents execute without errors
- Layer 2 (LLM + Docker) executes successfully
- Results are generated and displayed
- No memory/timeout issues
- System handles edge cases gracefully

