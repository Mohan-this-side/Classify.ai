#!/bin/bash
# Check evaluation test status

echo "=== Evaluation Test Status ==="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "test_full_workflow_weatherAUS.py"; then
    echo "✅ Test is RUNNING"
    ps aux | grep -v grep | grep "test_full_workflow_weatherAUS.py" | awk '{print "   PID:", $2, "| CPU:", $3"%", "| Runtime:", $10}'
    echo ""
    echo "Latest log entries:"
    tail -5 /tmp/eval_full.log 2>/dev/null | grep -E "(Testing Agent|COMPLETED|FAILED)" | tail -3
else
    echo "❌ Test is NOT running"
    echo ""
    echo "Last run status:"
    tail -20 /tmp/eval_full.log 2>/dev/null | grep -E "(COMPLETED|FAILED|SUMMARY|✅|❌)" | tail -5
fi

echo ""
echo "=== Latest Results ==="
LATEST_DIR=$(find results -type d -name "20*" 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_DIR" ]; then
    echo "Latest results folder: $LATEST_DIR"
    echo ""
    echo "Generated files:"
    echo "  Plots: $(find "$LATEST_DIR/plots" -name "*.png" 2>/dev/null | wc -l | xargs) PNG files"
    echo "  Tables: $(find "$LATEST_DIR/tables" -name "*.md" 2>/dev/null | wc -l | xargs) markdown files"
    echo "  Reports: $(find "$LATEST_DIR/reports" -name "*.json" -o -name "*.md" 2>/dev/null | wc -l | xargs) files"
    echo ""
    if [ -f "$LATEST_DIR/comprehensive_evaluation_report.md" ]; then
        echo "✅ Comprehensive evaluation report found!"
    else
        echo "⏳ Comprehensive evaluation report not yet generated (test may still be running)"
    fi
else
    echo "No results found yet"
fi

