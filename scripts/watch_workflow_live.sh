#!/bin/bash
# Live Workflow Monitoring - Real-time updates
# Usage: ./watch_workflow_live.sh <workflow_id>

WORKFLOW_ID="${1}"
LOG_DIR="/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"

if [ -z "$WORKFLOW_ID" ]; then
    echo "Usage: ./watch_workflow_live.sh <workflow_id>"
    exit 1
fi

echo "🔍 Live Monitoring for Workflow: ${WORKFLOW_ID}"
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

# Function to get workflow status
get_status() {
    curl -s "http://localhost:8000/api/workflow/status/${WORKFLOW_ID}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"Status: {data.get('status', 'unknown')}\")
    print(f\"Current Agent: {data.get('current_agent', 'none')}\")
    print(f\"Progress: {data.get('progress', 0)}%\")
    agents = data.get('agent_status', {})
    print(f\"\\nAgent Status:\")
    for agent, status in agents.items():
        print(f\"  {agent}: {status}\")
    print(f\"\\nLayer Usage:\")
    layers = data.get('layer_usage', {})
    for agent, layer in layers.items():
        print(f\"  {agent}: {layer}\")
except Exception as e:
    print(f\"Error: {e}\")
" 2>/dev/null || echo "Could not fetch status"
}

# Function to show recent logs
show_logs() {
    echo "📝 Recent Backend Logs:"
    tail -5 "${LOG_DIR}/backend/backend.log" 2>/dev/null | grep -E "agent|layer|workflow|ERROR" || echo "No relevant logs"
}

# Function to show Docker containers
show_docker() {
    echo "🐳 Active Sandbox Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" --filter "name=sandbox" 2>/dev/null || echo "No sandbox containers"
}

# Main monitoring loop
while true; do
    clear
    echo "🔍 Live Monitoring for Workflow: ${WORKFLOW_ID}"
    echo "Last updated: $(date '+%H:%M:%S')"
    echo "================================================"
    echo ""
    
    get_status
    echo ""
    show_docker
    echo ""
    show_logs
    echo ""
    echo "Refreshing in 3 seconds... (Ctrl+C to stop)"
    
    sleep 3
done

