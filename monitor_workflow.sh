#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}WORKFLOW MONITORING STARTED${NC}"
echo -e "${CYAN}Watching: backend/backend.log${NC}"
echo -e "${CYAN}Press CTRL+C to stop${NC}"
echo -e "${CYAN}================================${NC}\n"

# Follow the backend log with colored output
tail -f backend/backend.log | while read line; do
    case "$line" in
        *"Started workflow"*)
            echo -e "${GREEN}🚀 $line${NC}"
            ;;
        *"Executing agent"*)
            echo -e "${BLUE}🤖 $line${NC}"
            ;;
        *"Layer 1"*)
            echo -e "${CYAN}📊 $line${NC}"
            ;;
        *"Layer 2"*)
            echo -e "${MAGENTA}🧠 $line${NC}"
            ;;
        *"Sandbox"*|*"sandbox"*)
            echo -e "${YELLOW}🐳 $line${NC}"
            ;;
        *"completed"*|*"success"*)
            echo -e "${GREEN}✅ $line${NC}"
            ;;
        *"ERROR"*|*"error"*|*"failed"*)
            echo -e "${RED}❌ $line${NC}"
            ;;
        *"WARNING"*|*"warning"*)
            echo -e "${YELLOW}⚠️  $line${NC}"
            ;;
        *)
            echo "$line"
            ;;
    esac
done

