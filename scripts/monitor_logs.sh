#!/bin/bash
# Log monitoring script for comprehensive testing

echo "=== LOG MONITORING STARTED ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Backend logs
echo "--- BACKEND LOGS ---"
tail -f backend/backend.log 2>/dev/null | sed 's/^/[BACKEND] /' &
BACKEND_PID=$!

# Docker logs (sandbox containers)
echo "--- DOCKER SANDBOX LOGS ---"
docker ps --filter "name=sandbox" --format "{{.Names}}" | while read container; do
    docker logs -f "$container" 2>&1 | sed "s/^/[DOCKER-$container] /" &
done

# Frontend logs (if available)
if [ -f frontend/frontend.log ]; then
    echo "--- FRONTEND LOGS ---"
    tail -f frontend/frontend.log 2>&1 | sed 's/^/[FRONTEND] /' &
    FRONTEND_PID=$!
fi

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
