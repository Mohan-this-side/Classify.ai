# Workflow Monitoring Commands Reference
# Use these commands to monitor the workflow execution

# 1. Watch backend logs in real-time (color-coded)
# ./watch_backend_logs.sh

# 2. Comprehensive monitoring (workflow status + logs + Docker)
# ./monitor_comprehensive.sh <workflow_id>

# 3. Quick error check
# ./quick_error_check.sh

# 4. Monitor specific workflow
# ./watch_workflow_live.sh <workflow_id>

# 5. Manual log tail
# tail -f backend/backend.log | grep -E "agent|layer|error|completed"

# 6. Check Docker containers
# docker ps -a | grep sandbox

# 7. Check workflow status via API
# curl -s http://localhost:8000/api/workflow/status/<workflow_id> | python3 -m json.tool

# 8. Check workflow results
# curl -s http://localhost:8000/api/workflow/results/<workflow_id> | python3 -m json.tool | head -50

