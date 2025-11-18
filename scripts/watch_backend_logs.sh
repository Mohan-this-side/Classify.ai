#!/bin/bash
# Watch Backend Logs in Real-Time
# Shows all backend activity with color coding

LOG_DIR="/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"
BACKEND_LOG="${LOG_DIR}/backend/backend.log"

if [ ! -f "$BACKEND_LOG" ]; then
    echo "❌ Backend log not found: $BACKEND_LOG"
    exit 1
fi

echo "📝 Watching backend logs in real-time..."
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Use tail -f to follow the log file
tail -f "$BACKEND_LOG" | while read line; do
    # Color code based on content
    if echo "$line" | grep -qi "error\|exception\|traceback\|failed"; then
        echo -e "\033[0;31m$line\033[0m"  # Red for errors
    elif echo "$line" | grep -qi "warning"; then
        echo -e "\033[1;33m$line\033[0m"  # Yellow for warnings
    elif echo "$line" | grep -qi "completed\|success\|✅"; then
        echo -e "\033[0;32m$line\033[0m"  # Green for success
    elif echo "$line" | grep -qi "agent\|layer"; then
        echo -e "\033[0;34m$line\033[0m"  # Blue for agent activity
    else
        echo "$line"  # Default color
    fi
done

