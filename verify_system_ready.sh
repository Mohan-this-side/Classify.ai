#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}DS Capstone System Verification${NC}"
echo -e "${BLUE}================================${NC}\n"

# Track overall status
ALL_CHECKS_PASSED=true

# Function to print status
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        ALL_CHECKS_PASSED=false
    fi
}

# 1. Check Docker Desktop
echo -e "\n${YELLOW}[1/10] Checking Docker Desktop...${NC}"
if docker info > /dev/null 2>&1; then
    check_status 0 "Docker Desktop is running"
    DOCKER_VERSION=$(docker --version)
    echo "   Version: $DOCKER_VERSION"
else
    check_status 1 "Docker Desktop is NOT running"
    echo -e "${YELLOW}   Please start Docker Desktop and run this script again${NC}"
    exit 1
fi

# 2. Check Python virtual environment
echo -e "\n${YELLOW}[2/10] Checking Python environment...${NC}"
if [ -d "venv" ]; then
    check_status 0 "Virtual environment exists"
    if [ -f "venv/bin/python" ]; then
        PYTHON_VERSION=$(venv/bin/python --version 2>&1)
        echo "   $PYTHON_VERSION"
    fi
else
    check_status 1 "Virtual environment not found"
    echo "   Run: python3 -m venv venv"
fi

# 3. Check backend dependencies
echo -e "\n${YELLOW}[3/10] Checking backend dependencies...${NC}"
if [ -f "backend/requirements.txt" ]; then
    check_status 0 "requirements.txt exists"
    # Check if key packages are installed
    if venv/bin/pip show fastapi > /dev/null 2>&1; then
        check_status 0 "FastAPI installed"
    else
        check_status 1 "FastAPI not installed"
        echo "   Run: cd backend && pip install -r requirements.txt"
    fi
else
    check_status 1 "requirements.txt not found"
fi

# 4. Check frontend dependencies
echo -e "\n${YELLOW}[4/10] Checking frontend dependencies...${NC}"
if [ -d "frontend/node_modules" ]; then
    check_status 0 "Node modules installed"
else
    check_status 1 "Node modules not installed"
    echo "   Run: cd frontend && npm install"
fi

# 5. Check Docker Compose file
echo -e "\n${YELLOW}[5/10] Checking Docker Compose configuration...${NC}"
if [ -f "docker/docker-compose.yml" ]; then
    check_status 0 "docker-compose.yml exists"
else
    check_status 1 "docker-compose.yml not found"
fi

# 6. Check Sandbox Dockerfile
echo -e "\n${YELLOW}[6/10] Checking ML Sandbox Dockerfile...${NC}"
if [ -f "docker/Dockerfile.sandbox" ]; then
    check_status 0 "Dockerfile.sandbox exists"
else
    check_status 1 "Dockerfile.sandbox not found"
fi

# 7. Check if ports are available
echo -e "\n${YELLOW}[7/10] Checking port availability...${NC}"
PORT_8000=$(lsof -i :8000 | grep LISTEN | wc -l)
if [ $PORT_8000 -eq 0 ]; then
    check_status 0 "Port 8000 available for backend"
else
    check_status 1 "Port 8000 is in use"
    echo "   Run: kill -9 \$(lsof -t -i:8000)"
fi

PORT_3001=$(lsof -i :3001 | grep LISTEN | wc -l)
if [ $PORT_3001 -eq 0 ]; then
    check_status 0 "Port 3001 available for frontend"
else
    echo -e "${GREEN}✅ Port 3001 in use (frontend running)${NC}"
fi

# 8. Check test data
echo -e "\n${YELLOW}[8/10] Checking test datasets...${NC}"
if [ -f "test_data/Loan Approval Dataset.csv" ]; then
    check_status 0 "Loan Approval Dataset found"
else
    check_status 1 "Loan Approval Dataset not found"
fi

if [ -f "test_data/iris_clean.csv" ]; then
    check_status 0 "Iris dataset found"
else
    check_status 1 "Iris dataset not found"
fi

# 9. Check backend structure
echo -e "\n${YELLOW}[9/10] Checking backend structure...${NC}"
BACKEND_FILES=(
    "backend/app/main.py"
    "backend/app/api/workflow_routes.py"
    "backend/app/agents/base_agent.py"
    "backend/app/services/sandbox_executor.py"
)

for file in "${BACKEND_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_status 0 "$(basename $file) exists"
    else
        check_status 1 "$(basename $file) not found"
    fi
done

# 10. Check frontend files
echo -e "\n${YELLOW}[10/10] Checking frontend structure...${NC}"
FRONTEND_FILES=(
    "frontend/app/page.tsx"
    "frontend/app/globals.css"
    "frontend/postcss.config.js"
    "frontend/tailwind.config.js"
)

for file in "${FRONTEND_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_status 0 "$(basename $file) exists"
    else
        check_status 1 "$(basename $file) not found"
    fi
done

# Summary
echo -e "\n${BLUE}================================${NC}"
if [ "$ALL_CHECKS_PASSED" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo -e "\n${GREEN}System is ready for testing!${NC}\n"
    echo -e "Next steps:"
    echo -e "1. Start backend:  cd backend && source ../venv/bin/activate && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo -e "2. Frontend is already running on port 3001"
    echo -e "3. Open browser: http://localhost:3001"
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo -e "\n${YELLOW}Please fix the issues above before proceeding${NC}"
fi
echo -e "${BLUE}================================${NC}\n"

