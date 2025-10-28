# Frontend-Backend Integration Plan

## Current Status

### Backend (Port 8000)
- ✅ FastAPI server with REST APIs
- ✅ WebSocket support for real-time updates
- ✅ Multi-agent workflow system
- ✅ Docker sandbox for code execution
- ✅ CORS configured for port 3001

### Frontend (Port 3001)
- ✅ Beautiful UI with Tailwind CSS
- ✅ Three views: Upload, Workflow, Results
- ❌ Not connected to backend yet

## Integration Steps

### 1. API Endpoints to Connect

#### Start Workflow
- **Endpoint**: `POST /api/workflow/start`
- **Payload**: FormData with `file`, `target_column`, `description`, `api_key`
- **Response**: `{workflow_id, status, message}`

#### Get Workflow Status
- **Endpoint**: `GET /api/workflow/status/{workflow_id}`
- **Response**: `{status, progress, current_phase, agent_status}`

#### Get Workflow Results
- **Endpoint**: `GET /api/workflow/results/{workflow_id}`
- **Response**: Complete results including metrics, plots, models

#### WebSocket for Real-Time Updates
- **Endpoint**: `ws://localhost:8000/ws/{session_id}`
- **Events**:
  - `agent.started` - Agent begins execution
  - `agent.progress` - Agent reports progress
  - `agent.completed` - Agent finishes
  - `pm.message` - Project Manager sends message
  - `approval.required` - Approval gate triggered
  - `sandbox.metrics` - CPU/Memory metrics

### 2. Frontend Updates Needed

#### page.tsx
- Add WebSocket connection on workflow start
- Call `/api/workflow/start` when "Start Analysis" clicked
- Poll `/api/workflow/status` or use WebSocket for updates
- Update timeline based on agent statuses
- Display PM messages in real-time
- Show sandbox metrics
- Fetch and display results when complete

#### Remove Unused Components
- Keep: None (all are unused by current page.tsx)
- Delete: All components in `components/` folder (they're from old implementation)

### 3. Data Flow

```
Upload View:
1. User uploads CSV file
2. User selects target column (auto-populated from CSV headers)
3. User enters API key
4. User clicks "Start Analysis"
5. Frontend sends FormData to POST /api/workflow/start
6. Backend returns workflow_id
7. Frontend switches to Workflow View

Workflow View:
1. Frontend opens WebSocket connection to ws://localhost:8000/ws/{workflow_id}
2. Listen for real-time events:
   - agent.started → Update timeline icon to "running"
   - agent.progress → Update progress bar
   - agent.completed → Update timeline icon to "completed"
   - pm.message → Add message to PM panel
   - approval.required → Show approval gate
   - sandbox.metrics → Update CPU/Memory display
3. When all agents complete → Switch to Results View

Results View:
1. Fetch GET /api/workflow/results/{workflow_id}
2. Display metrics, plots, feature importance
3. Provide download links for deliverables
```

### 4. Testing Plan

1. **Upload Test**: Upload Loan Approval Dataset.csv
2. **Target Column**: Select "loan_approved"
3. **API Key**: Enter valid Gemini API key
4. **Verify Timeline**: Check 8 agents progress correctly
5. **Verify PM Messages**: Check messages appear in real-time
6. **Verify Sandbox**: Check CPU/Memory metrics update
7. **Verify Results**: Check results display correctly

## Implementation Priority

1. ✅ Fix CORS (DONE)
2. ✅ Start backend (DONE)
3. 🔄 Connect Start Analysis button
4. 🔄 Add WebSocket integration
5. 🔄 Update agent timeline with real data
6. 🔄 Display PM messages
7. 🔄 Show sandbox metrics
8. 🔄 Display results
9. 🔄 Clean up unused components
10. 🔄 End-to-end testing

