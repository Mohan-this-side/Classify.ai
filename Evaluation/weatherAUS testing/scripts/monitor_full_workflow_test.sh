#!/bin/bash
# Monitor the full workflow test progress

LOG_FILE="Evaluation/results/full_workflow_test.log"
REPORT_FILE="Evaluation/results/full_workflow_weatherAUS_report.json"

echo "Monitoring Full Workflow Test..."
echo "Log file: $LOG_FILE"
echo "Report file: $REPORT_FILE"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== Full Workflow Test Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo "--- Last 30 lines of log ---"
        tail -30 "$LOG_FILE"
    else
        echo "Log file not found yet..."
    fi
    
    echo ""
    echo "--- Test Status ---"
    if [ -f "$REPORT_FILE" ]; then
        echo "✓ Report file exists - Test may be complete!"
        echo "Report summary:"
        python3 -c "
import json
try:
    with open('$REPORT_FILE') as f:
        report = json.load(f)
    summary = report.get('summary', {})
    print(f\"  Steps: {summary.get('total_steps', 'N/A')}\")
    print(f\"  Agents: {summary.get('agents_executed', 'N/A')}\")
    print(f\"  Docker Executions: {summary.get('docker_executions', 'N/A')}\")
    print(f\"  Successful Docker: {summary.get('successful_docker', 'N/A')}\")
    print(f\"  Errors: {summary.get('errors', 'N/A')}\")
except Exception as e:
    print(f\"Error reading report: {e}\")
" 2>/dev/null || echo "  Could not parse report"
    else
        echo "⏳ Report file not found - Test still running..."
    fi
    
    echo ""
    echo "--- Docker Containers ---"
    docker ps -a --filter "name=sandbox-" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | head -10
    
    sleep 5
done

