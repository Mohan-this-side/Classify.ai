# Quick Start Guide - DS Capstone Multi-Agent System

## 🚀 System is Running!

**Backend:** http://localhost:8000 ✅  
**Frontend:** http://localhost:3000 ✅  
**Docker:** Running ✅  

---

## 📊 Current Status

✅ **PRODUCTION READY** - All systems operational

- **Backend API:** Running on port 8000
- **Frontend UI:** Running on port 3000
- **Docker Sandbox:** Ready and operational
- **Test Workflow:** Completed successfully (93.33% accuracy)
- **All 8 Agents:** Verified working

---

## 🎯 Quick Test

### Access the UI
```bash
open http://localhost:3000
```

### Test Workflow (Already Completed)
- **Workflow ID:** `baa592a5-664a-409a-af92-cfee09941565`
- **Status:** Completed ✅
- **Accuracy:** 93.33%

### View Results
```bash
# Get workflow status
curl http://localhost:8000/api/workflow/status/baa592a5-664a-409a-af92-cfee09941565 | python3 -m json.tool

# Get workflow results
curl http://localhost:8000/api/workflow/results/baa592a5-664a-409a-af92-cfee09941565 | python3 -m json.tool

# View generated files
ls -lh backend/results/baa592a5-664a-409a-af92-cfee09941565/
```

---

## 🔄 Run a New Workflow

### Option 1: Via UI (Recommended)
1. Go to http://localhost:3000
2. Upload a CSV file (min 10 rows)
3. Select target column
4. Add dataset description
5. Enter API key
6. Click "Start Analysis"
7. Approve gates when prompted
8. Download results

### Option 2: Via API
```python
import requests

# Upload and start workflow
files = {'file': open('test_data/iris_clean.csv', 'rb')}
data = {
    'target_column': 'species',
    'description': 'Iris classification dataset',
    'api_key': 'your-gemini-api-key'
}

response = requests.post('http://localhost:8000/api/workflow/start', files=files, data=data)
workflow_id = response.json()['workflow_id']
print(f"Workflow started: {workflow_id}")
```

---

## 🛠️ Monitoring

### Watch Logs
```bash
# Backend logs
tail -f backend/backend.log

# Frontend logs  
tail -f frontend/frontend.log

# Comprehensive monitoring
./scripts/monitor_comprehensive.sh <workflow_id>
```

### Check Status
```bash
# Health check
curl http://localhost:8000/health

# Workflow status
curl http://localhost:8000/api/workflow/status/<workflow_id>
```

---

## 🎓 Approval Gates

Workflows pause at 3 approval gates:
1. **After EDA** - Review data insights
2. **After Data Cleaning** - Confirm data quality
3. **After Feature Engineering** - Ready to train models

### Approve Manually
```python
import requests

workflow_id = "your-workflow-id"
requests.post(
    f"http://localhost:8000/api/workflow/{workflow_id}/pm/approval",
    json={"action": "approve", "feedback": "Looks good!"}
)
```

---

## 🐳 Docker Management

### Check Docker Status
```bash
docker ps
docker images | grep ds-capstone
```

### Restart Docker
```bash
# If Docker is not running
open -a Docker
```

---

## ⚠️ Troubleshooting

### Backend Won't Start
```bash
# Check for port conflicts
lsof -ti:8000 | xargs kill -9

# Restart backend
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Won't Start
```bash
# Check for port conflicts
lsof -ti:3000 | xargs kill -9

# Restart frontend
cd frontend
npm run dev
```

### Docker Issues
```bash
# Ensure Docker Desktop is running
open -a Docker

# Check Docker daemon
docker ps
```

---

## 📚 Documentation

- **Full Status Report:** `PROJECT_STATUS_NOVEMBER_25_2025.md`
- **PRD Gap Analysis:** `PRD_GAP_ANALYSIS.md`
- **Deployment Guide:** `DEPLOYMENT_CHECKLIST.md`
- **Project README:** `README.md`

---

## 🎉 What's Working

✅ Multi-agent workflow (8 agents)  
✅ Approval gates (human-in-the-loop)  
✅ Docker sandbox execution  
✅ Model training & evaluation  
✅ Deliverable generation  
✅ Real-time status updates  
✅ PM educational messages  
✅ Plot generation  
✅ Results API  

## 🟡 Minor Improvements Recommended

- WebSocket instead of polling (works but less efficient)
- Approval gate auto-timeout (works manually)
- Results endpoint path display (plots accessible directly)

---

**System Status:** ✅ **READY FOR USE**

**Last Verified:** November 25, 2025, 4:00 PM EST

