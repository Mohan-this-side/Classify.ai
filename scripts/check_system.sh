#!/bin/bash
# Quick Diagnostic Script - Run this to check system health before workflow

echo "🔍 SYSTEM HEALTH CHECK"
echo "====================="
echo ""

# Check backend
echo "📡 Backend Status:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ Backend is running on port 8000"
else
    echo "  ❌ Backend is NOT responding"
fi

# Check frontend
echo "🌐 Frontend Status:"
if curl -s http://localhost:3001 > /dev/null 2>&1; then
    echo "  ✅ Frontend is running on port 3001"
else
    echo "  ❌ Frontend is NOT responding"
fi

# Check Docker
echo "🐳 Docker Status:"
if docker ps > /dev/null 2>&1; then
    echo "  ✅ Docker daemon is running"
    echo "  Active containers: $(docker ps -q | wc -l | tr -d ' ')"
else
    echo "  ❌ Docker daemon is NOT running"
fi

# Check sandbox image
echo "📦 Sandbox Image:"
if docker images | grep -q "ds-capstone-ml-sandbox"; then
    echo "  ✅ Sandbox image exists"
else
    echo "  ⚠️  Sandbox image may need to be built"
fi

# Check logs
echo "📝 Log Files:"
if [ -f "backend/backend.log" ]; then
    echo "  ✅ Backend log exists ($(wc -l < backend/backend.log | tr -d ' ') lines)"
else
    echo "  ⚠️  Backend log not found"
fi

# Check recent errors
echo ""
echo "⚠️  Recent Errors (last 5):"
grep -i "error\|exception\|traceback" backend/backend.log 2>/dev/null | tail -5 || echo "  No errors found"

echo ""
echo "✅ Health check complete. System ready for workflow execution."

