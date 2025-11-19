#!/bin/bash
# Quick Error Checker - Run this to see if there are any errors right now

LOG_DIR="/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project"
BACKEND_LOG="${LOG_DIR}/backend/backend.log"

echo "🔍 QUICK ERROR CHECK"
echo "===================="
echo ""

if [ ! -f "$BACKEND_LOG" ]; then
    echo "❌ Backend log not found at: $BACKEND_LOG"
    exit 1
fi

echo "📝 Last 5 errors/warnings:"
grep -i "error\|exception\|traceback\|warning\|failed" "$BACKEND_LOG" 2>/dev/null | tail -5 || echo "✅ No errors found in recent logs"

echo ""
echo "📊 Last 10 log lines:"
tail -10 "$BACKEND_LOG" 2>/dev/null

echo ""
echo "✅ Error check complete"

