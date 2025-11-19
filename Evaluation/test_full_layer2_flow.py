"""
Test Full Layer 2 Flow with Docker Sandbox
Tests: Layer 1 → LLM Prompt → Code Generation → Docker Execution → Results
"""

import sys
import asyncio
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from test_cases.base_test_framework import BaseAgentTest
from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.workflows.state_management import ClassificationState, WorkflowStatus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/full_layer2_flow.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_docker():
    """Check if Docker is running."""
    import subprocess
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ Docker is running")
            return True
        else:
            logger.error(f"✗ Docker check failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Cannot check Docker: {e}")
        return False


def check_sandbox_image():
    """Check if sandbox image exists."""
    import subprocess
    try:
        result = subprocess.run(['docker', 'images', 'ds-capstone-ml-sandbox'], capture_output=True, text=True, timeout=5)
        if 'ds-capstone-ml-sandbox' in result.stdout:
            logger.info("✓ Sandbox image exists")
            return True
        else:
            logger.warning("✗ Sandbox image not found. Building...")
            return build_sandbox_image()
    except Exception as e:
        logger.error(f"✗ Cannot check sandbox image: {e}")
        return False


def build_sandbox_image():
    """Build the sandbox Docker image."""
    import subprocess
    dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.sandbox"
    if not dockerfile_path.exists():
        logger.error(f"Dockerfile not found at {dockerfile_path}")
        return False
    
    logger.info("Building sandbox image (this may take a few minutes)...")
    try:
        result = subprocess.run(
            ['docker', 'build', '-t', 'ds-capstone-ml-sandbox', '-f', str(dockerfile_path), str(dockerfile_path.parent.parent / "backend")],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        if result.returncode == 0:
            logger.info("✓ Sandbox image built successfully")
            return True
        else:
            logger.error(f"✗ Build failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Build error: {e}")
        return False


def ensure_volumes():
    """Ensure Docker volumes exist."""
    import subprocess
    volumes = ['sandbox_code', 'sandbox_results', 'sandbox_data']
    for volume in volumes:
        try:
            result = subprocess.run(['docker', 'volume', 'inspect', volume], capture_output=True, timeout=5)
            if result.returncode != 0:
                logger.info(f"Creating volume: {volume}")
                subprocess.run(['docker', 'volume', 'create', volume], check=True)
            else:
                logger.info(f"✓ Volume exists: {volume}")
        except Exception as e:
            logger.warning(f"Could not check/create volume {volume}: {e}")


async def test_agent_with_full_flow(agent, agent_name: str, state: ClassificationState, dataset_path: str):
    """Test an agent with full Layer 2 Docker flow."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {agent_name} - Full Layer 2 Flow")
    logger.info(f"{'='*80}")
    
    # Step 1: Layer 1
    logger.info("\n[Step 1/5] Executing Layer 1 (hardcoded analysis)...")
    try:
        result = await agent.execute(state)
        logger.info("✓ Layer 1 completed")
        
        # Check what Layer 1 produced
        layer1_results = {}
        if agent_name == 'data_discovery':
            layer1_results = {
                'data_types': state.get('data_types', {}),
                'discovery_results': state.get('discovery_results') is not None
            }
        elif agent_name == 'eda_analysis':
            layer1_results = {
                'statistical_summary': state.get('statistical_summary') is not None,
                'eda_plots': len(state.get('eda_plots', [])),
                'correlation_matrix': state.get('correlation_matrix') is not None
            }
        elif agent_name == 'data_cleaning':
            layer1_results = {
                'cleaned_dataset': state.get('cleaned_dataset') is not None,
                'cleaning_summary': state.get('cleaning_summary') is not None,
                'data_quality_score': state.get('data_quality_score')
            }
        
        logger.info(f"Layer 1 Results: {layer1_results}")
        
    except Exception as e:
        logger.error(f"✗ Layer 1 failed: {e}")
        return {'success': False, 'error': f'Layer 1 failed: {e}'}
    
    # Step 2: Check if Layer 2 was attempted
    logger.info("\n[Step 2/5] Checking Layer 2 execution...")
    
    # Check logs for Layer 2 attempts
    layer2_attempted = False
    layer2_code_generated = False
    layer2_docker_executed = False
    layer2_docker_success = False
    
    # Try to extract Layer 2 info from result
    if isinstance(result, dict):
        layer_info = result.get('data', {}).get('layer', 'layer1')
        if layer_info == 'layer2':
            layer2_attempted = True
            layer2_code_generated = True
            layer2_docker_executed = True
            layer2_docker_success = True
    
    logger.info(f"Layer 2 Attempted: {layer2_attempted}")
    logger.info(f"Code Generated: {layer2_code_generated}")
    logger.info(f"Docker Executed: {layer2_docker_executed}")
    logger.info(f"Docker Success: {layer2_docker_success}")
    
    # Step 3: Check Docker execution details
    logger.info("\n[Step 3/5] Checking Docker sandbox execution...")
    
    # Check for Docker containers
    import subprocess
    try:
        containers = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}} {{.Status}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if containers.stdout.strip():
            logger.info(f"Recent sandbox containers:\n{containers.stdout}")
        else:
            logger.warning("No sandbox containers found")
    except Exception as e:
        logger.warning(f"Could not check containers: {e}")
    
    # Step 4: Check results
    logger.info("\n[Step 4/5] Analyzing results...")
    
    final_results = {
        'layer1_success': True,
        'layer1_results': layer1_results,
        'layer2_attempted': layer2_attempted,
        'layer2_code_generated': layer2_code_generated,
        'layer2_docker_executed': layer2_docker_executed,
        'layer2_docker_success': layer2_docker_success,
        'final_result': result
    }
    
    # Step 5: Summary
    logger.info("\n[Step 5/5] Summary:")
    logger.info(f"  Layer 1: {'✓ Success' if final_results['layer1_success'] else '✗ Failed'}")
    logger.info(f"  Layer 2 Code Generated: {'✓' if layer2_code_generated else '✗'}")
    logger.info(f"  Docker Execution: {'✓ Success' if layer2_docker_success else '✗ Not executed/Failed'}")
    
    return final_results


async def test_full_flow():
    """Test full Layer 2 flow on weatherAUS dataset."""
    logger.info("="*80)
    logger.info("FULL LAYER 2 FLOW TEST - weatherAUS Dataset")
    logger.info("="*80)
    
    # Step 0: Check Docker
    logger.info("\n[Step 0] Checking Docker setup...")
    if not check_docker():
        logger.error("Docker is not running! Please start Docker Desktop.")
        return {'success': False, 'error': 'Docker not running'}
    
    if not check_sandbox_image():
        logger.error("Sandbox image not available!")
        return {'success': False, 'error': 'Sandbox image not available'}
    
    ensure_volumes()
    
    # Step 1: Load dataset
    logger.info("\n[Step 1] Loading dataset...")
    dataset_path = '/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project/Evaluation/datasets/real_world/weatherAUS.csv'
    target_col = 'RainTomorrow'
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return {'success': False, 'error': 'Dataset not found'}
    
    # Load and sample dataset
    logger.info("Loading dataset (sampling to 2000 rows for faster testing)...")
    df = pd.read_csv(dataset_path)
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled to {len(df)} rows")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Step 2: Create state
    logger.info("\n[Step 2] Creating workflow state...")
    base_test = BaseAgentTest()
    state = base_test.create_state(df, target_col)
    
    # Ensure target column is set
    state['target_column'] = target_col
    logger.info(f"State created with target_column: {state.get('target_column')}")
    
    # Step 3: Test agents
    logger.info("\n[Step 3] Testing agents with full Layer 2 flow...")
    
    agents_to_test = [
        ('Data Discovery', DataDiscoveryAgent()),
        ('EDA Analysis', EDAAgent()),
        ('Data Cleaning', EnhancedDataCleaningAgent()),
    ]
    
    results = {}
    
    for agent_name, agent in agents_to_test:
        try:
            result = await test_agent_with_full_flow(agent, agent_name, state, dataset_path)
            results[agent_name] = result
            
            # Update state for next agent
            # (In real workflow, state is updated by agents)
            
        except Exception as e:
            logger.error(f"Error testing {agent_name}: {e}", exc_info=True)
            results[agent_name] = {'success': False, 'error': str(e)}
    
    # Step 4: Generate report
    logger.info("\n" + "="*80)
    logger.info("FULL FLOW TEST SUMMARY")
    logger.info("="*80)
    
    for agent_name, result in results.items():
        logger.info(f"\n{agent_name}:")
        logger.info(f"  Layer 1: {'✓' if result.get('layer1_success') else '✗'}")
        logger.info(f"  Layer 2 Code Generated: {'✓' if result.get('layer2_code_generated') else '✗'}")
        logger.info(f"  Docker Executed: {'✓' if result.get('layer2_docker_executed') else '✗'}")
        logger.info(f"  Docker Success: {'✓' if result.get('layer2_docker_success') else '✗'}")
    
    # Save results
    output_path = Path("Evaluation/results/full_layer2_flow_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return {'success': True, 'results': results}


if __name__ == "__main__":
    result = asyncio.run(test_full_flow())
    if result.get('success'):
        print("\n✓ Full flow test completed!")
    else:
        print(f"\n✗ Test failed: {result.get('error')}")
        sys.exit(1)

