#!/bin/bash
# Comprehensive Real-Time Workflow Monitoring
# Monitors backend logs, Docker containers, and workflow status simultaneously

WORKFLOW_ID="${1}"
LOG_DIR="/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"
BACKEND_LOG="${LOG_DIR}/backend/backend.log"

if [ -z "$WORKFLOW_ID" ]; then
    echo "⚠️  No workflow ID provided. Monitoring general system logs..."
    echo "Usage: ./monitor_comprehensive.sh <workflow_id>"
    echo ""
fi

echo "🔍 COMPREHENSIVE WORKFLOW MONITORING"
echo "======================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$WORKFLOW_ID" ]; then
    echo "Workflow ID: ${WORKFLOW_ID}"
fi
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to get workflow status
get_workflow_status() {
    if [ -z "$WORKFLOW_ID" ]; then
        return
    fi
    
    echo -e "${BLUE}📊 WORKFLOW STATUS:${NC}"
    curl -s "http://localhost:8000/api/workflow/status/${WORKFLOW_ID}" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Status: {data.get('status', 'unknown')}\")
    print(f\"  Current Agent: {data.get('current_agent', 'none')}\")
    print(f\"  Progress: {data.get('progress', 0):.1f}%\")
    agents = data.get('agent_status', {})
    print(f\"  \\n  Agent Status:\")
    for agent, status in agents.items():
        emoji = '✅' if status == 'completed' else '🔄' if status == 'running' else '⏳'
        print(f\"    {emoji} {agent}: {status}\")
except Exception as e:
    print(f\"  Error: {e}\")
" 2>/dev/null || echo "  Could not fetch status"
    echo ""
}

# Function to show recent backend errors
show_backend_errors() {
    echo -e "${RED}⚠️  RECENT ERRORS (last 10):${NC}"
    if [ -f "$BACKEND_LOG" ]; then
        grep -i "error\|exception\|traceback\|failed" "$BACKEND_LOG" 2>/dev/null | tail -10 || echo "  No errors found"
    else
        echo "  Backend log not found"
    fi
    echo ""
}

# Function to show recent agent activity
show_agent_activity() {
    echo -e "${GREEN}🤖 RECENT AGENT ACTIVITY (last 15 lines):${NC}"
    if [ -f "$BACKEND_LOG" ]; then
        grep -E "agent|layer|completed|started" "$BACKEND_LOG" 2>/dev/null | tail -15 || echo "  No agent activity found"
    else
        echo "  Backend log not found"
    fi
    echo ""
}

# Function to show Docker containers
show_docker_status() {
    echo -e "${BLUE}🐳 DOCKER CONTAINERS:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" --filter "name=sandbox" 2>/dev/null | head -5 || echo "  No sandbox containers"
    echo ""
}

# Function to show system resources
show_resources() {
    echo -e "${YELLOW}💻 SYSTEM RESOURCES:${NC}"
    echo "  CPU: $(top -l 1 | grep 'CPU usage' | awk '{print $3}' | cut -d'%' -f1)%"
    echo "  Memory: $(top -l 1 | grep 'PhysMem' | awk '{print $2}')"
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo "🔍 COMPREHENSIVE WORKFLOW MONITORING"
    echo "======================================"
    echo "Last updated: $(date '+%H:%M:%S')"
    if [ -n "$WORKFLOW_ID" ]; then
        echo "Workflow ID: ${WORKFLOW_ID}"
    fi
    echo ""
    
    get_workflow_status
    show_docker_status
    show_agent_activity
    show_backend_errors
    show_resources
    
    echo "Refreshing in 3 seconds... (Ctrl+C to stop)"
    sleep 3
done

