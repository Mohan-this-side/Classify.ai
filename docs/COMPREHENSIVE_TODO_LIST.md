# 📋 COMPREHENSIVE TODO LIST
## Multi-Agent System Fixes & Enhancements

**Created**: October 28, 2025
**Based on**: LangGraph v1.0 best practices and observed workflow issues
**Reference**: [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)

---

## 🎯 PHASE 1: CRITICAL FIXES (Must Complete First)

### ✅ TODO 1.1: Fix Agent Execution Order
**Status**: ✅ **COMPLETED**
**File**: `backend/app/api/workflow_routes.py`
**What was done**: Reordered agents from Cleaning→Discovery→EDA to Discovery→EDA→Cleaning
**Impact**: Agents now execute in logical order

### TODO 1.2: Fix EDA Agent Dataset Access
**Status**: 🔴 **PENDING**
**Priority**: CRITICAL
**File**: `backend/app/agents/data_analysis/eda_agent.py`
**Issue**: EDA agent can't find dataset, returns "No dataset available for EDA"
**Fix Required**:
```python
async def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
    """Layer 1: Hardcoded EDA analysis"""
    
    # Try multiple ways to get dataset
    df = None
    
    # Method 1: Get from processed_dataset
    df = state.get("processed_dataset")
    
    # Method 2: Fallback to original dataset
    if df is None:
        df = state.get("dataset")
    
    # Method 3: Get from state manager using dataset_id
    if df is None:
        from ..workflows.state_management import state_manager
        dataset_id = state.get("dataset_id")
        if dataset_id:
            df = state_manager.get_dataset(dataset_id, "original")
    
    # Method 4: Last resort - log all available state keys for debugging
    if df is None:
        logger.error(f"No dataset found. Available state keys: {list(state.keys())}")
        return {
            "error": "No dataset available for EDA",
            "eda_plots": [],
            "statistical_summary": {},
            "distribution_analysis": {},
            "correlation_matrix": None
        }
    
    logger.info(f"EDA: Found dataset with shape {df.shape}")
    # Continue with EDA analysis...
```

### TODO 1.3: Create Plots Directory and Generate Visualizations
**Status**: 🔴 **PENDING**
**Priority**: CRITICAL
**File**: `backend/app/agents/data_analysis/eda_agent.py`
**Issue**: No plots generated, directory doesn't exist
**Fix Required**:
```python
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Get workflow ID from state
workflow_id = state.get("session_id") or state.get("workflow_id", "default")
plot_dir = f"backend/plots/{workflow_id}"

# Create directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)
logger.info(f"Created plots directory: {plot_dir}")

# Generate and save plots
plot_paths = []

# 1. Correlation Matrix
if len(df.select_dtypes(include=[np.number]).columns) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    corr_path = f"{plot_dir}/correlation_matrix.png"
    plt.savefig(corr_path, bbox_inches='tight', dpi=100)
    plt.close()
    plot_paths.append(f"/api/workflow/plot/{workflow_id}/correlation_matrix.png")
    logger.info(f"Saved correlation matrix to {corr_path}")

# 2. Distribution plots for numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, col in enumerate(numeric_cols):
        if idx < 4:
            df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    plt.tight_layout()
    dist_path = f"{plot_dir}/distributions.png"
    plt.savefig(dist_path, bbox_inches='tight', dpi=100)
    plt.close()
    plot_paths.append(f"/api/workflow/plot/{workflow_id}/distributions.png")
    logger.info(f"Saved distributions to {dist_path}")

# 3. Target distribution (if categorical)
target_col = state.get("target_column")
if target_col and target_col in df.columns:
    plt.figure(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f'Target Distribution: {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    target_path = f"{plot_dir}/target_distribution.png"
    plt.savefig(target_path, bbox_inches='tight', dpi=100)
    plt.close()
    plot_paths.append(f"/api/workflow/plot/{workflow_id}/target_distribution.png")
    logger.info(f"Saved target distribution to {target_path}")

# Store plot paths in state
return {
    "eda_plots": plot_paths,
    "statistical_summary": df.describe().to_dict(),
    ...
}
```

### TODO 1.4: Fix Dataset Passing Between Agents
**Status**: 🔴 **PENDING**
**Priority**: CRITICAL
**Files**: Multiple agent files
**Issue**: Agents not properly passing cleaned/processed datasets to next agents
**Fix Required**: Each agent must update `state["processed_dataset"]`:

**In Data Cleaning Agent**:
```python
# After cleaning operations
cleaned_df = ... # cleaned dataframe
state["processed_dataset"] = cleaned_df
state["dataset"] = cleaned_df  # Also update main dataset reference
logger.info(f"Data cleaning: Updated processed_dataset with shape {cleaned_df.shape}")
```

**In Feature Engineering Agent**:
```python
# Get dataset from previous agent
df = state.get("processed_dataset") or state.get("dataset")
# After feature engineering
engineered_df = ... # with new features
state["processed_dataset"] = engineered_df
logger.info(f"Feature engineering: Updated processed_dataset with {len(new_features)} new features")
```

---

## 🎯 PHASE 2: LANGGRAPH IMPLEMENTATION IMPROVEMENTS

### TODO 2.1: Implement Proper LangGraph State Management
**Status**: 🔴 **PENDING**
**Priority**: HIGH
**Reference**: [LangGraph State Management](https://docs.langchain.com/oss/python/langgraph/overview)
**Files**: `backend/app/workflows/classification_workflow.py`, `backend/app/workflows/state_management.py`
**Issue**: Not using LangGraph's persistence and state properly
**Fix Required**:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# Use proper TypedDict for state
from typing import TypedDict, Annotated
from operator import add

class ClassificationState(TypedDict, total=False):
    """Proper LangGraph state definition"""
    messages: Annotated[list, add]  # For PM agent communication
    dataset: Any
    processed_dataset: Any
    target_column: str
    workflow_id: str
    session_id: str
    agent_statuses: dict
    eda_plots: list
    model_metrics: dict
    approval_gates: Annotated[list, add]  # For approval gates
    pm_messages: Annotated[list, add]  # For PM chat
    current_agent: str
    workflow_status: str
    progress: float

# Use MemorySaver for persistence
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)
```

### TODO 2.2: Implement Project Manager as Coordination Node
**Status**: 🔴 **PENDING**
**Priority**: HIGH
**Reference**: [LangGraph Workflows + Agents](https://docs.langchain.com/oss/python/langgraph/overview)
**File**: `backend/app/agents/coordination/project_manager_agent.py`
**Issue**: PM not coordinating workflow or handling approval gates
**Fix Required**:

```python
class ProjectManagerAgent(BaseAgent):
    """
    Project Manager coordinates workflow and handles human-in-the-loop.
    Based on LangGraph's recommendation for coordination patterns.
    """
    
    async def execute(self, state: ClassificationState) -> ClassificationState:
        """Execute PM coordination"""
        
        # 1. Generate status update message
        current_agent = state.get("current_agent")
        completed_agents = state.get("completed_agents", [])
        
        pm_message = {
            "role": "pm",
            "agent": current_agent,
            "message": self._generate_status_message(state),
            "timestamp": datetime.now().isoformat()
        }
        
        # 2. Add to PM messages (using Annotated[list, add])
        state["pm_messages"] = state.get("pm_messages", []) + [pm_message]
        
        # 3. Check for approval gates
        if self._should_trigger_approval(state):
            approval_gate = self._create_approval_gate(state)
            state["approval_gates"] = state.get("approval_gates", []) + [approval_gate]
            state["workflow_status"] = "paused"
            logger.info(f"PM: Triggered approval gate for {current_agent}")
        
        # 4. Answer user questions (if any pending)
        if "user_question" in state:
            answer = await self._answer_question(state["user_question"], state)
            pm_message["answer"] = answer
        
        return state
    
    def _generate_status_message(self, state: ClassificationState) -> str:
        """Generate educational PM message about current agent"""
        agent = state.get("current_agent")
        
        messages = {
            "data_discovery": "I'm analyzing your dataset structure. Found {} rows and {} columns. The data looks {}!",
            "eda_analysis": "Running exploratory analysis. Discovered interesting correlations and patterns in your data.",
            "data_cleaning": "Cleaning your data based on EDA insights. Found {} missing values and {} outliers.",
            "feature_engineering": "Creating {} new features to improve model performance...",
            "ml_building": "Training multiple classification models. Best performer so far: {}",
            "model_evaluation": "Evaluating model performance. Current accuracy: {}%",
            "technical_reporter": "Generating your final report with all findings and recommendations."
        }
        
        return messages.get(agent, f"Processing {agent}...")
    
    def _should_trigger_approval(self, state: ClassificationState) -> bool:
        """Determine if approval gate needed"""
        agent = state.get("current_agent")
        
        # Trigger approval for critical decisions
        if agent == "data_cleaning":
            # Check if significant data removal
            issues = state.get("cleaning_issues_found", [])
            return len(issues) > 10  # Many issues found
        
        if agent == "feature_engineering":
            # Check if many new features
            new_features = state.get("engineered_features", [])
            return len(new_features) > 5
        
        return False
    
    def _create_approval_gate(self, state: ClassificationState) -> Dict:
        """Create approval gate with educational explanation"""
        return {
            "gate_id": str(uuid.uuid4()),
            "agent": state.get("current_agent"),
            "title": "Approval Required",
            "proposal": self._get_proposal(state),
            "explanation": self._get_educational_explanation(state),
            "code": self._get_generated_code(state),
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
```

### TODO 2.3: Add Conditional Edges for Approval Gates
**Status**: 🔴 **PENDING**
**Priority**: HIGH
**Reference**: [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/overview)
**File**: `backend/app/workflows/classification_workflow.py`
**Fix Required**:

```python
def _should_continue_or_pause(self, state: ClassificationState) -> str:
    """Conditional edge: continue workflow or pause for approval"""
    
    # Check if approval gate is pending
    approval_gates = state.get("approval_gates", [])
    pending_gates = [g for g in approval_gates if g["status"] == "pending"]
    
    if pending_gates:
        logger.info("Workflow paused for approval gate")
        return "pause"  # Will trigger interrupt
    
    # Check if agent failed
    current_agent = state.get("current_agent")
    if state["agent_statuses"].get(current_agent) == AgentStatus.FAILED:
        return "error"
    
    return "continue"

# In graph building:
workflow.add_conditional_edges(
    "data_cleaning",
    self._should_continue_or_pause,
    {
        "continue": "project_management",
        "pause": "__interrupt__",  # LangGraph interrupt
        "error": "error_recovery"
    }
)
```

---

## 🎯 PHASE 3: FRONTEND FIXES

### TODO 3.1: Fix Frontend Progress Bar with Correct Agent Mapping
**Status**: 🔴 **PENDING**
**Priority**: HIGH
**File**: `frontend/app/page.tsx`
**Issue**: Progress bar doesn't update, wrong agent names
**Fix Required**:

```typescript
// Agent mapping between backend and frontend
const AGENT_MAPPING = {
  'data_discovery': { id: 'discovery', label: 'Discovery', order: 0 },
  'eda_analysis': { id: 'eda', label: 'EDA', order: 1 },
  'data_cleaning': { id: 'cleaning', label: 'Cleaning', order: 2 },
  'feature_engineering': { id: 'feature', label: 'Feature Eng.', order: 3 },
  'ml_building': { id: 'model', label: 'Model Build', order: 4 },
  'model_evaluation': { id: 'eval', label: 'Evaluation', order: 5 },
  'technical_reporter': { id: 'report', label: 'Reporting', order: 6 },
  'project_manager': { id: 'pm', label: 'PM', order: 7 }
}

// Update agents based on backend status
const updateAgentsFromStatus = (backendStatus: any) => {
  const agentStatuses = backendStatus.agent_status || backendStatus.agent_statuses
  
  const updatedAgents = Object.entries(AGENT_MAPPING).map(([backendName, info]) => {
    const status = agentStatuses[backendName]
    return {
      id: info.id,
      label: info.label,
      status: mapBackendStatus(status),
      time: status === 'completed' ? '1:23' : '', // Get from backend later
      order: info.order
    }
  }).sort((a, b) => a.order - b.order)
  
  setAgents(updatedAgents)
}

// Map backend status to frontend display
const mapBackendStatus = (backendStatus: string): 'pending' | 'active' | 'complete' | 'failed' => {
  const mapping = {
    'pending': 'pending',
    'running': 'active',
    'completed': 'complete',
    'failed': 'failed'
  }
  return mapping[backendStatus] || 'pending'
}

// In pollWorkflowStatus:
const pollWorkflowStatus = async (wfId: string) => {
  const interval = setInterval(async () => {
    const res = await fetch(`http://localhost:8000/api/workflow/status/${wfId}`)
    const data = await res.json()
    
    // Update agents
    updateAgentsFromStatus(data)
    
    // Update progress
    setProgress(data.progress || 0)
    
    // Check if completed
    if (data.status === 'completed' || data.workflow_status === 'completed') {
      clearInterval(interval)
      fetchResults(wfId)
    }
  }, 2000)  // Poll every 2 seconds
}
```

### TODO 3.2: Replace Hardcoded Data with API Results
**Status**: 🔴 **PENDING**
**Priority**: HIGH
**File**: `frontend/app/page.tsx`
**Issue**: Frontend shows iris data (petal_length, petal_width) instead of actual dataset
**Fix Required**:

```typescript
// Fetch and display actual results
const fetchResults = async (wfId: string) => {
  try {
    const res = await fetch(`http://localhost:8000/api/workflow/results/${wfId}`)
    const data = await res.json()
    
    console.log('Fetched results:', data)  // Debug
    
    // Extract actual data
    const metrics = data.results?.evaluation_metrics || {}
    const features = data.results?.feature_importance_model || {}
    const plots = data.results?.eda_plots || []
    
    // Update results state with ACTUAL data
    setResults({
      // Metrics from model evaluation
      accuracy: (metrics.accuracy * 100).toFixed(1) || '0',
      f1_score: metrics.f1_score?.toFixed(2) || '0',
      precision: (metrics.precision * 100).toFixed(1) || '0',
      recall: (metrics.recall * 100).toFixed(1) || '0',
      
      // Feature importance from actual dataset
      features: Object.entries(features)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([name, importance]) => ({
          name: name,
          importance: (importance * 100).toFixed(1)
        })),
      
      // EDA plots
      plots: plots,
      
      // Dataset info
      datasetShape: data.results?.dataset_info?.shape || [0, 0],
      targetColumn: data.target_column || 'unknown',
      
      // Files for download
      files: {
        dataset: `/api/workflow/download/${wfId}/dataset`,
        notebook: `/api/workflow/download/${wfId}/notebook`,
        model: `/api/workflow/download/${wfId}/model`,
        report: `/api/workflow/download/${wfId}/report`
      }
    })
    
    setActiveView('results')
    toast.success('Workflow completed!')
  } catch (error) {
    console.error('Error fetching results:', error)
    toast.error('Failed to fetch results')
  }
}

// Update ResultsView to use actual data
const ResultsView = () => (
  <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
    <div className="max-w-6xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold">Analysis Complete! 🎉</h2>
      
      {/* Display ACTUAL dataset info */}
      <div className="bg-white p-4 rounded-lg">
        <p>Dataset: {results.datasetShape[0]} rows × {results.datasetShape[1]} columns</p>
        <p>Target: {results.targetColumn}</p>
      </div>
      
      {/* Display ACTUAL metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Accuracy" value={`${results.accuracy}%`} color="blue" />
        <MetricCard label="F1 Score" value={results.f1_score} color="green" />
        <MetricCard label="Precision" value={`${results.precision}%`} color="purple" />
        <MetricCard label="Recall" value={`${results.recall}%`} color="orange" />
      </div>
      
      {/* Display ACTUAL EDA plots */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="font-semibold mb-4">Exploratory Data Analysis</h3>
        <div className="grid grid-cols-2 gap-4">
          {results.plots.map((plotUrl, idx) => (
            <img 
              key={idx}
              src={`http://localhost:8000${plotUrl}`}
              alt={`EDA Plot ${idx + 1}`}
              className="w-full h-64 object-contain border rounded"
            />
          ))}
        </div>
      </div>
      
      {/* Display ACTUAL feature importance */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="font-semibold mb-4">Feature Importance</h3>
        {results.features.map((feature, idx) => (
          <FeatureBar 
            key={idx}
            label={feature.name}
            value={parseFloat(feature.importance)}
          />
        ))}
      </div>
      
      {/* Downloadable files */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="font-semibold mb-4">Your Deliverables</h3>
        <DeliverableItem 
          icon="📊" 
          name="cleaned_dataset.csv" 
          downloadUrl={results.files.dataset}
        />
        <DeliverableItem 
          icon="📓" 
          name="analysis_notebook.ipynb" 
          downloadUrl={results.files.notebook}
        />
        <DeliverableItem 
          icon="🤖" 
          name="trained_model.joblib" 
          downloadUrl={results.files.model}
        />
      </div>
    </div>
  </div>
)
```

### TODO 3.3: Implement Project Manager Chat Panel
**Status**: 🔴 **PENDING**
**Priority**: MEDIUM
**File**: `frontend/app/page.tsx`
**Issue**: PM panel shows mock data, doesn't fetch real PM messages
**Fix Required**:

```typescript
// Fetch PM messages from backend
const fetchPMMessages = async (wfId: string) => {
  const res = await fetch(`http://localhost:8000/api/workflow/pm-messages/${wfId}`)
  const data = await res.json()
  setPmMessages(data.messages || [])
}

// Poll for PM messages during workflow
useEffect(() => {
  if (workflowStatus === 'running' && workflowId) {
    const interval = setInterval(() => {
      fetchPMMessages(workflowId)
    }, 3000)  // Poll every 3 seconds
    
    return () => clearInterval(interval)
  }
}, [workflowStatus, workflowId])

// Display PM messages
const PMMessage = ({ message }) => (
  <div className="space-y-2">
    <div className="flex items-center justify-between text-xs">
      <span className="font-medium text-purple-600">{message.agent}</span>
      <span className="text-gray-400">{formatTime(message.timestamp)}</span>
    </div>
    <div className="bg-gray-50 rounded-lg p-3 text-sm">
      {message.message}
    </div>
    {message.code && (
      <details>
        <summary className="text-xs text-blue-600 cursor-pointer">
          View Generated Code
        </summary>
        <pre className="bg-gray-900 text-white p-3 rounded text-xs overflow-x-auto mt-2">
          {message.code}
        </pre>
      </details>
    )}
  </div>
)

// Handle approval gates
{pmMessages.filter(m => m.type === 'approval').map((gate, idx) => (
  <div key={idx} className="bg-amber-50 border-2 border-amber-300 rounded-lg p-4">
    <AlertCircle className="w-5 h-5 text-amber-600" />
    <p className="font-semibold">{gate.title}</p>
    <p className="text-sm">{gate.proposal}</p>
    <div className="bg-white rounded p-2 my-2">
      <p className="text-xs text-gray-700">{gate.explanation}</p>
    </div>
    <div className="flex space-x-2">
      <button onClick={() => handleApproval(gate.gate_id, 'approve')} 
        className="flex-1 bg-green-600 text-white py-2 px-3 rounded">
        ✓ Approve
      </button>
      <button onClick={() => handleApproval(gate.gate_id, 'reject')}
        className="flex-1 bg-red-100 text-red-700 py-2 px-3 rounded">
        Reject
      </button>
    </div>
  </div>
))}
```

---

## 🎯 PHASE 4: FILE DOWNLOADS & ADDITIONAL FEATURES

### TODO 4.1: Add File Download Endpoints
**Status**: 🔴 **PENDING**
**Priority**: MEDIUM
**File**: `backend/app/api/workflow_routes.py`
**Fix Required**:

```python
from fastapi.responses import FileResponse
import os

@router.get("/download/{workflow_id}/{file_type}")
async def download_workflow_file(workflow_id: str, file_type: str) -> FileResponse:
    """
    Download generated files from workflow.
    
    Supported file types:
    - dataset: Cleaned CSV
    - notebook: Jupyter notebook
    - model: Trained model (.joblib)
    - report: Technical report (.md)
    """
    try:
        # Define file paths based on type
        file_paths = {
            "dataset": f"backend/data/cleaned_{workflow_id}.csv",
            "notebook": f"backend/notebooks/classification_project_{workflow_id}.ipynb",
            "model": f"backend/models/model_{workflow_id}.joblib",
            "report": f"backend/reports/report_{workflow_id}.md"
        }
        
        if file_type not in file_paths:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
        
        file_path = file_paths[file_type]
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try alternative locations
            alt_paths = {
                "notebook": glob.glob(f"backend/notebooks/*{workflow_id}*.ipynb"),
                "model": glob.glob(f"backend/models/*{workflow_id}*.joblib")
            }
            
            if file_type in alt_paths and alt_paths[file_type]:
                file_path = alt_paths[file_type][0]
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"File not found: {file_type} for workflow {workflow_id}"
                )
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Return file
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@router.get("/pm-messages/{workflow_id}")
async def get_pm_messages(workflow_id: str) -> Dict[str, Any]:
    """Get Project Manager messages for a workflow"""
    try:
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        state = workflow_states[workflow_id]
        messages = state.get("pm_messages", [])
        
        return {
            "workflow_id": workflow_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting PM messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### TODO 4.2: Add User Question Handling for PM
**Status**: 🔴 **PENDING**
**Priority**: LOW
**Files**: `backend/app/api/workflow_routes.py`, `frontend/app/page.tsx`
**Fix Required**:

**Backend**:
```python
@router.post("/ask-pm/{workflow_id}")
async def ask_project_manager(
    workflow_id: str,
    question: Dict[str, str]
) -> Dict[str, Any]:
    """Ask Project Manager a question during workflow"""
    try:
        if workflow_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Add question to state
        workflow_states[workflow_id]["user_question"] = question.get("question")
        
        # PM will answer on next execution
        # For now, return acknowledgment
        return {
            "status": "received",
            "message": "Question will be answered by Project Manager",
            "workflow_id": workflow_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Frontend**:
```typescript
const askPM = async (question: string) => {
  try {
    const res = await fetch(`http://localhost:8000/api/workflow/ask-pm/${workflowId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    })
    
    if (res.ok) {
      toast.success('Question sent to PM')
      setUserQuestion('')
    }
  } catch (error) {
    toast.error('Failed to send question')
  }
}
```

---

## 📊 SUMMARY & PRIORITIES

### Must Do First (Blocking Issues):
1. ✅ Fix agent execution order (DONE)
2. 🔴 Fix EDA dataset access (TODO 1.2)
3. 🔴 Create plots directory & generate visualizations (TODO 1.3)
4. 🔴 Fix dataset passing between agents (TODO 1.4)

### High Priority (Core Functionality):
5. 🔴 Fix frontend progress bar mapping (TODO 3.1)
6. 🔴 Replace hardcoded frontend data (TODO 3.2)
7. 🔴 Implement proper LangGraph state (TODO 2.1)
8. 🔴 Improve PM coordination (TODO 2.2)

### Medium Priority (Enhanced Features):
9. 🔴 Add file download endpoints (TODO 4.1)
10. 🔴 Implement PM chat panel (TODO 3.3)
11. 🔴 Add approval gate conditional edges (TODO 2.3)

### Low Priority (Nice to Have):
12. 🔴 Add user question handling (TODO 4.2)

---

## 🎯 TESTING CHECKLIST

After completing todos, test:
- [ ] Upload Loan Approval Dataset
- [ ] Verify agent order: Discovery → EDA → Cleaning → Feature → ML → Eval → Report
- [ ] Check EDA plots are generated and displayed on frontend
- [ ] Verify progress bar updates correctly for each agent
- [ ] Confirm frontend shows actual dataset features (not iris data)
- [ ] Test file downloads (dataset, notebook, model)
- [ ] Check PM messages appear in real-time
- [ ] Test approval gates trigger correctly
- [ ] Verify Layer 2 (sandbox) executes when validation passes

---

**Status**: Ready to implement
**Next Action**: Start with TODO 1.2 (EDA dataset access)

