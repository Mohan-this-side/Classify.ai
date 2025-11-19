#!/bin/bash
# Watch workflow execution in real-time

WORKFLOW_ID=$1

if [ -z "$WORKFLOW_ID" ]; then
    echo "Usage: ./watch_workflow.sh <workflow_id>"
    echo "Example: ./watch_workflow.sh abc-123-def-456"
    exit 1
fi

echo "=== Monitoring Workflow: $WORKFLOW_ID ==="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== Workflow Status: $WORKFLOW_ID ==="
    echo "Time: $(date)"
    echo ""
    
    # Get workflow status
    curl -s http://localhost:8000/api/workflow/status/$WORKFLOW_ID | python3 -m json.tool 2>/dev/null | head -30
    
    echo ""
    echo "--- Docker Containers ---"
    docker ps --filter "name=sandbox" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" 2>/dev/null
    
    echo ""
    echo "--- Recent Backend Logs ---"
    tail -5 backend/backend.log 2>/dev/null | sed 's/^/  /'
    
    sleep 3
done
