"""
Test Suite for BaseAgent Double-Layer Architecture

This test suite validates the double-layer architecture implementation in BaseAgent.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.agents.base_agent import BaseAgent
from app.workflows.state_management import ClassificationState, create_initial_state


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing double-layer architecture"""

    def __init__(self, **kwargs):
        super().__init__(agent_name="test_agent", agent_version="1.0.0", **kwargs)
        self.layer1_called = False
        self.layer2_called = False

    def perform_layer1_analysis(self, state: ClassificationState) -> Dict[str, Any]:
        """Test Layer 1 implementation"""
        self.layer1_called = True
        return {
            "test_result": "layer1_data",
            "quality_score": 75.0,
            "issues_found": 5
        }

    def generate_layer2_code(self, layer1_results: Dict[str, Any], state: ClassificationState) -> str:
        """Test Layer 2 prompt generation"""
        self.layer2_called = True
        return f"""
Generate Python code to improve upon this analysis:
Layer 1 found {layer1_results['issues_found']} issues.

import pandas as pd
def improve_analysis(data):
    return {{"test_result": "layer2_data", "quality_score": 85.0}}

result = improve_analysis(data)
"""

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "layer1_called": self.layer1_called,
            "layer2_called": self.layer2_called
        }


class TestBaseAgentDoubleLayer:
    """Test cases for double-layer architecture"""

    @pytest.fixture
    def initial_state(self):
        """Create initial state for testing"""
        return create_initial_state(
            session_id="test_session",
            dataset_id="test_dataset"
        )

    @pytest.fixture
    def test_agent_layer1_only(self):
        """Create agent with Layer 2 disabled"""
        return TestAgent(enable_layer2=False)

    @pytest.fixture
    def test_agent_full(self):
        """Create agent with Layer 2 enabled"""
        return TestAgent(enable_layer2=True)

    def test_agent_initialization_layer1_only(self, test_agent_layer1_only):
        """Test agent initialization with Layer 2 disabled"""
        agent = test_agent_layer1_only

        assert agent.agent_name == "test_agent"
        assert agent.enable_layer2 is False
        assert agent.sandbox_executor is None
        assert agent.code_validator is None
        assert agent.llm_service is None

    def test_agent_initialization_layer2_enabled(self, test_agent_full):
        """Test agent initialization with Layer 2 enabled"""
        agent = test_agent_full

        assert agent.agent_name == "test_agent"
        # Note: Services might not initialize if dependencies are missing
        # This is expected behavior (graceful degradation)

    @pytest.mark.asyncio
    async def test_layer1_execution_only(self, test_agent_layer1_only, initial_state):
        """Test execution with only Layer 1 (Layer 2 disabled)"""
        agent = test_agent_layer1_only
        state = initial_state

        result_state = await agent.execute(state)

        # Verify Layer 1 was called
        assert agent.layer1_called is True
        assert agent.layer2_called is False

        # Verify results
        assert "agent_results" in result_state
        assert "test_agent" in result_state["agent_results"]

        agent_result = result_state["agent_results"]["test_agent"]
        assert agent_result["layer_used"] == "layer1"
        assert agent_result["layer2_attempted"] is False
        assert agent_result["results"]["test_result"] == "layer1_data"

    @pytest.mark.asyncio
    async def test_layer1_fallback_when_layer2_fails(self, initial_state):
        """Test fallback to Layer 1 when Layer 2 fails"""
        agent = TestAgent(enable_layer2=True)

        # Mock services to simulate Layer 2 failure
        agent.code_validator = Mock()
        agent.llm_service = Mock()
        agent.sandbox_executor = Mock()

        # Mock LLM to return invalid code
        agent.llm_service.generate_code = AsyncMock(return_value={
            "code": "invalid python code that won't parse",
            "provider": "gemini"
        })

        # Mock validator to reject code
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.errors = ["Syntax error"]
        agent.code_validator.validate.return_value = mock_validation_result

        state = initial_state
        result_state = await agent.execute(state)

        # Verify Layer 1 was called
        assert agent.layer1_called is True

        # Verify Layer 2 was attempted but failed
        agent_result = result_state["agent_results"]["test_agent"]
        assert agent_result["layer_used"] == "layer1"
        assert agent_result["layer2_attempted"] is True
        assert agent_result["layer2_error"] is not None

    @pytest.mark.asyncio
    async def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented"""

        # Try to create agent without implementing abstract methods
        with pytest.raises(TypeError):
            class IncompleteAgent(BaseAgent):
                def __init__(self):
                    super().__init__(agent_name="incomplete")

                # Missing perform_layer1_analysis
                # Missing generate_layer2_code
                # Missing get_agent_info

            agent = IncompleteAgent()

    def test_prepare_sandbox_datasets_default(self, test_agent_layer1_only, initial_state):
        """Test default implementation of _prepare_sandbox_datasets"""
        agent = test_agent_layer1_only
        datasets = agent._prepare_sandbox_datasets(initial_state)

        # Default implementation returns empty dict
        assert datasets == {}

    def test_can_use_layer2_checks(self, test_agent_full):
        """Test _can_use_layer2 validation logic"""
        agent = test_agent_full

        # Initially might be False if services failed to initialize
        initial_result = agent._can_use_layer2()

        # If services are None, should return False
        if agent.sandbox_executor is None:
            assert initial_result is False

        # Mock services to test True case
        agent.enable_layer2 = True
        agent.sandbox_executor = Mock()
        agent.code_validator = Mock()
        agent.llm_service = Mock()

        assert agent._can_use_layer2() is True

        # Disable Layer 2
        agent.enable_layer2 = False
        assert agent._can_use_layer2() is False

    @pytest.mark.asyncio
    async def test_execute_layer2_in_sandbox_timeout(self, initial_state):
        """Test sandbox execution timeout handling"""
        agent = TestAgent(enable_layer2=True, sandbox_timeout=5)

        # Mock sandbox to simulate timeout
        agent.sandbox_executor = Mock()
        agent.sandbox_executor.timeout = 5
        agent.sandbox_executor.execute_code.return_value = {
            "status": "TIMEOUT",
            "error": "Execution timed out after 5 seconds",
            "execution_time": 5
        }

        generated_code = "print('test')"
        layer1_results = {"test": "data"}

        with pytest.raises(TimeoutError):
            await agent.execute_layer2_in_sandbox(
                generated_code,
                layer1_results,
                initial_state
            )

    @pytest.mark.asyncio
    async def test_execute_layer2_in_sandbox_error(self, initial_state):
        """Test sandbox execution error handling"""
        agent = TestAgent(enable_layer2=True)

        # Mock sandbox to simulate error
        agent.sandbox_executor = Mock()
        agent.sandbox_executor.execute_code.return_value = {
            "status": "ERROR",
            "error": "Python error: NameError",
            "execution_time": 1
        }

        generated_code = "print(undefined_variable)"
        layer1_results = {"test": "data"}

        with pytest.raises(RuntimeError):
            await agent.execute_layer2_in_sandbox(
                generated_code,
                layer1_results,
                initial_state
            )

    def test_process_sandbox_results_default(self, test_agent_full, initial_state):
        """Test default implementation of process_sandbox_results"""
        agent = test_agent_full

        # Success case
        sandbox_output = {
            "status": "SUCCESS",
            "output": {"result": "test_data"},
            "execution_time": 2.5
        }

        result = agent.process_sandbox_results(
            sandbox_output,
            {"layer1": "data"},
            initial_state
        )

        assert result == {"result": "test_data"}

        # Failure case
        sandbox_output_fail = {
            "status": "FAILURE",
            "error": "Test error",
            "execution_time": 1.0
        }

        with pytest.raises(ValueError):
            agent.process_sandbox_results(
                sandbox_output_fail,
                {"layer1": "data"},
                initial_state
            )

    def test_update_state_with_results(self, test_agent_full, initial_state):
        """Test state update with results and metadata"""
        agent = test_agent_full

        results = {"test": "data", "score": 85.0}
        layer_used = "layer2"
        layer2_attempted = True
        layer2_error = None

        updated_state = agent._update_state_with_results(
            initial_state,
            results,
            layer_used,
            layer2_attempted,
            layer2_error
        )

        # Check agent results are stored
        assert "agent_results" in updated_state
        assert "test_agent" in updated_state["agent_results"]

        agent_result = updated_state["agent_results"]["test_agent"]
        assert agent_result["results"] == results
        assert agent_result["layer_used"] == "layer2"
        assert agent_result["layer2_attempted"] is True
        assert agent_result["layer2_error"] is None
        assert "timestamp" in agent_result

        # Check layer usage tracking
        assert "layer_usage" in updated_state
        assert updated_state["layer_usage"]["test_agent"] == "layer2"


class TestCodeValidatorIntegration:
    """Test CodeValidator integration with BaseAgent"""

    def test_code_validator_blocks_dangerous_code(self):
        """Test that CodeValidator blocks dangerous operations"""
        from app.services.code_validator import get_code_validator

        validator = get_code_validator()

        # Test dangerous operations
        dangerous_codes = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.run(['ls'])",
            "eval('malicious code')",
            "exec('dangerous code')",
            "__import__('os').system('ls')",
            "import socket; socket.socket()",
            "open('/etc/passwd', 'r')"
        ]

        for code in dangerous_codes:
            result = validator.validate(code)
            assert result.is_valid is False, f"Should block: {code}"
            assert len(result.security_issues) > 0 or len(result.errors) > 0

    def test_code_validator_allows_safe_code(self):
        """Test that CodeValidator allows safe ML code"""
        from app.services.code_validator import get_code_validator

        validator = get_code_validator()

        safe_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

result = train_model(X_train, y_train)
"""

        result = validator.validate(safe_code)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.security_issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
