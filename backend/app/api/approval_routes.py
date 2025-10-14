"""
API routes for approval gate management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

from ..workflows.approval_gates import (
    ApprovalGateManager, 
    ApprovalGateType, 
    ApprovalStatus,
    get_approval_gate_definition,
    should_trigger_approval_gate,
    create_approval_proposal,
    generate_educational_explanation
)
from ..workflows.state_management import state_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/approval-gates", tags=["approval-gates"])

# Global approval gate manager (in production, this should be stored in database)
approval_manager = ApprovalGateManager()

class ApprovalRequest(BaseModel):
    comments: Optional[str] = None

class RejectionRequest(BaseModel):
    comments: str

class ModificationRequest(BaseModel):
    modifications: Dict[str, Any]
    comments: str

class ApprovalGateResponse(BaseModel):
    gates: List[Dict[str, Any]]
    workflow_paused: bool

@router.get("/{workflow_id}")
async def get_approval_gates(workflow_id: str) -> ApprovalGateResponse:
    """
    Get all approval gates for a workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        List of approval gates and workflow status
    """
    try:
        # Get active gates
        active_gates = approval_manager.get_active_gates()
        completed_gates = approval_manager.completed_gates
        
        # Convert to dictionaries for JSON serialization
        all_gates = []
        for gate in active_gates + completed_gates:
            gate_dict = {
                "gate_id": gate["gate_id"],
                "gate_type": gate["gate_type"].value if hasattr(gate["gate_type"], 'value') else str(gate["gate_type"]),
                "title": gate["title"],
                "description": gate["description"],
                "proposal": gate["proposal"],
                "educational_explanation": gate["educational_explanation"],
                "status": gate["status"].value if hasattr(gate["status"], 'value') else str(gate["status"]),
                "created_at": gate["created_at"].isoformat() if hasattr(gate["created_at"], 'isoformat') else str(gate["created_at"]),
                "approved_at": gate["approved_at"].isoformat() if gate["approved_at"] and hasattr(gate["approved_at"], 'isoformat') else str(gate["approved_at"]) if gate["approved_at"] else None,
                "user_comments": gate["user_comments"],
                "modifications": gate["modifications"]
            }
            all_gates.append(gate_dict)
        
        # Check if workflow is paused (has pending gates)
        workflow_paused = approval_manager.has_pending_gates()
        
        return ApprovalGateResponse(
            gates=all_gates,
            workflow_paused=workflow_paused
        )
        
    except Exception as e:
        logger.error(f"Error getting approval gates for workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get approval gates")

@router.post("/{workflow_id}/{gate_id}/approve")
async def approve_gate(
    workflow_id: str, 
    gate_id: str, 
    request: ApprovalRequest
) -> Dict[str, Any]:
    """
    Approve an approval gate.
    
    Args:
        workflow_id: The workflow identifier
        gate_id: The gate identifier
        request: Approval request with optional comments
        
    Returns:
        Success response
    """
    try:
        success = approval_manager.approve_gate(gate_id, request.comments)
        
        if not success:
            raise HTTPException(status_code=404, detail="Gate not found")
        
        # Check if workflow can resume
        if not approval_manager.has_pending_gates():
            # Resume workflow logic would go here
            logger.info(f"All gates approved for workflow {workflow_id}, workflow can resume")
        
        return {"success": True, "message": "Gate approved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving gate {gate_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to approve gate")

@router.post("/{workflow_id}/{gate_id}/reject")
async def reject_gate(
    workflow_id: str, 
    gate_id: str, 
    request: RejectionRequest
) -> Dict[str, Any]:
    """
    Reject an approval gate.
    
    Args:
        workflow_id: The workflow identifier
        gate_id: The gate identifier
        request: Rejection request with required comments
        
    Returns:
        Success response
    """
    try:
        success = approval_manager.reject_gate(gate_id, request.comments)
        
        if not success:
            raise HTTPException(status_code=404, detail="Gate not found")
        
        return {"success": True, "message": "Gate rejected successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting gate {gate_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reject gate")

@router.post("/{workflow_id}/{gate_id}/modify")
async def modify_gate(
    workflow_id: str, 
    gate_id: str, 
    request: ModificationRequest
) -> Dict[str, Any]:
    """
    Modify an approval gate.
    
    Args:
        workflow_id: The workflow identifier
        gate_id: The gate identifier
        request: Modification request with modifications and comments
        
    Returns:
        Success response
    """
    try:
        success = approval_manager.modify_gate(
            gate_id, 
            request.modifications, 
            request.comments
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Gate not found")
        
        return {"success": True, "message": "Gate modified successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying gate {gate_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to modify gate")

@router.post("/{workflow_id}/resume")
async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Resume a paused workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        Success response
    """
    try:
        # Check if there are any pending gates
        if approval_manager.has_pending_gates():
            raise HTTPException(
                status_code=400, 
                detail="Cannot resume workflow with pending approval gates"
            )
        
        # Resume workflow logic would go here
        logger.info(f"Resuming workflow {workflow_id}")
        
        return {"success": True, "message": "Workflow resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resume workflow")

@router.post("/{workflow_id}/pause")
async def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Pause a workflow.
    
    Args:
        workflow_id: The workflow identifier
        
    Returns:
        Success response
    """
    try:
        # Pause workflow logic would go here
        logger.info(f"Pausing workflow {workflow_id}")
        
        return {"success": True, "message": "Workflow paused successfully"}
        
    except Exception as e:
        logger.error(f"Error pausing workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to pause workflow")

@router.post("/{workflow_id}/create-gate")
async def create_approval_gate(
    workflow_id: str,
    gate_type: str,
    current_agent: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new approval gate for a workflow.
    
    Args:
        workflow_id: The workflow identifier
        gate_type: Type of approval gate to create
        current_agent: Current agent being executed
        state: Current workflow state
        
    Returns:
        Created gate information
    """
    try:
        # Convert string to enum
        try:
            gate_type_enum = ApprovalGateType(gate_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid gate type: {gate_type}")
        
        # Check if gate should be triggered
        if not should_trigger_approval_gate(current_agent, state, gate_type_enum):
            return {"success": False, "message": "Gate should not be triggered at this time"}
        
        # Get gate definition
        gate_definition = get_approval_gate_definition(gate_type_enum)
        if not gate_definition:
            raise HTTPException(status_code=400, detail=f"No definition found for gate type: {gate_type}")
        
        # Create proposal
        proposal = create_approval_proposal(gate_type_enum, state)
        
        # Generate educational explanation
        educational_explanation = generate_educational_explanation(gate_type_enum, proposal, state)
        
        # Create the gate
        gate = approval_manager.create_gate(
            gate_type=gate_type_enum,
            title=gate_definition["title"],
            description=gate_definition["description"],
            proposal=proposal,
            educational_explanation=educational_explanation
        )
        
        return {
            "success": True,
            "gate": {
                "gate_id": gate["gate_id"],
                "gate_type": gate["gate_type"].value,
                "title": gate["title"],
                "description": gate["description"],
                "status": gate["status"].value
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating approval gate: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create approval gate")
