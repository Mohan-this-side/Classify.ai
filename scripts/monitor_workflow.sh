#!/bin/bash
# Comprehensive Workflow Monitoring Script
# Monitors backend, frontend, Docker, and workflow execution

WORKFLOW_ID="${1:-}"
LOG_DIR="/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"

echo "🔍 Starting Comprehensive Workflow Monitoring..."
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check service status
check_service() {
    local service=$1
    local port=$2
    
    if curl -s "http://localhost:${port}" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${service} is running on port ${port}${NC}"
        return 0
    else
        echo -e "${RED}❌ ${service} is NOT running on port ${port}${NC}"
        return 1
    fi
}

# Check services
echo "📊 Service Status Check:"
check_service "Backend" 8000
check_service "Frontend" 3001
echo ""

# Check Docker
echo "🐳 Docker Status:"
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker daemon is running${NC}"
    echo "Active containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -5
else
    echo -e "${RED}❌ Docker daemon is NOT running${NC}"
fi
echo ""

# Monitor backend logs
echo "📝 Backend Logs (last 20 lines):"
echo "--------------------------------"
tail -20 "${LOG_DIR}/backend/backend.log" 2>/dev/null || echo "No backend log file found"
echo ""

# Monitor Docker logs if workflow ID provided
if [ -n "$WORKFLOW_ID" ]; then
    echo "🐳 Docker Sandbox Logs for workflow ${WORKFLOW_ID}:"
    echo "--------------------------------"
    docker logs --tail 20 $(docker ps -q --filter "name=sandbox") 2>/dev/null || echo "No active sandbox containers"
    echo ""
    
    # Check workflow status
    echo "📊 Workflow Status:"
    echo "--------------------------------"
    curl -s "http://localhost:8000/api/workflow/status/${WORKFLOW_ID}" | python3 -m json.tool 2>/dev/null || echo "Could not fetch workflow status"
    echo ""
fi

# Monitor system resources
echo "💻 System Resources:"
echo "--------------------------------"
echo "CPU Usage:"
top -l 1 | grep "CPU usage" | head -1
echo ""
echo "Memory Usage:"
top -l 1 | grep "PhysMem" | head -1
echo ""

# Watch for errors
echo "⚠️  Recent Errors (last 10):"
echo "--------------------------------"
grep -i "error\|exception\|traceback\|failed" "${LOG_DIR}/backend/backend.log" 2>/dev/null | tail -10 || echo "No errors found"
echo ""

echo "✅ Monitoring complete. Run with workflow ID: ./monitor_workflow.sh <workflow_id>"
