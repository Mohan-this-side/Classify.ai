"""
Test Full Layer 2 Flow on weatherAUS Dataset
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
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from test_cases.base_test_framework import BaseAgentTest
from app.agents.data_analysis.data_discovery_agent import DataDiscoveryAgent
from app.agents.data_analysis.eda_agent import EDAAgent
from app.agents.data_cleaning.enhanced_data_cleaning_agent import EnhancedDataCleaningAgent
from app.workflows.state_management import ClassificationState, WorkflowStatus, state_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/weatherAUS_full_flow.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_docker_setup():
    """Check Docker setup."""
    logger.info("Checking Docker setup...")
    
    # Check Docker daemon
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ Docker daemon is running")
        else:
            logger.error(f"✗ Docker daemon not accessible: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Cannot connect to Docker: {e}")
        return False
    
    # Check sandbox image
    try:
        result = subprocess.run(['docker', 'images', 'ds-capstone-ml-sandbox'], capture_output=True, text=True, timeout=5)
        if 'ds-capstone-ml-sandbox' in result.stdout:
            logger.info("✓ Sandbox image exists")
        else:
            logger.error("✗ Sandbox image not found. Please build it first.")
            return False
    except Exception as e:
        logger.error(f"✗ Cannot check image: {e}")
        return False
    
    # Ensure volumes exist
    volumes = ['sandbox_code', 'sandbox_results', 'sandbox_data']
    for volume in volumes:
        try:
            subprocess.run(['docker', 'volume', 'inspect', volume], capture_output=True, timeout=5, check=False)
            logger.info(f"✓ Volume exists: {volume}")
        except:
            logger.info(f"Creating volume: {volume}")
            subprocess.run(['docker', 'volume', 'create', volume], check=True)
    
    return True


def monitor_docker_execution(container_name: str):
    """Monitor Docker container execution."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Monitoring Docker Container: {container_name}")
    logger.info(f"{'='*60}")
    
    # Check container status
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Names}} {{.Status}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            logger.info(f"Container status: {result.stdout.strip()}")
        else:
            logger.warning("Container not found")
    except Exception as e:
        logger.warning(f"Could not check container: {e}")
    
    # Check logs
    try:
        result = subprocess.run(
            ['docker', 'logs', container_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout:
            logger.info(f"Container logs:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Container errors:\n{result.stderr}")
    except Exception as e:
        logger.warning(f"Could not get logs: {e}")


async def test_agent_full_flow(agent, agent_name: str, state: ClassificationState, dataset_path: str):
    """Test agent with full Layer 2 Docker flow."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {agent_name} - Full Layer 2 Flow")
    logger.info(f"{'='*80}")
    
    flow_results = {
        'agent': agent_name,
        'layer1_completed': False,
        'layer2_code_generated': False,
        'layer2_docker_executed': False,
        'layer2_docker_success': False,
        'layer2_results': None,
        'errors': []
    }
    
    # Step 1: Execute agent
    logger.info("\n[Step 1] Executing agent (Layer 1 + Layer 2)...")
    try:
        result = await agent.execute(state)
        flow_results['layer1_completed'] = True
        logger.info("✓ Agent execution completed")
        
        # Check Layer 1 results
        if agent_name == 'Data Discovery':
            if state.get('data_types') or state.get('discovery_results'):
                logger.info("✓ Layer 1: Data types and discovery results found in state")
        elif agent_name == 'EDA Analysis':
            if state.get('statistical_summary') or state.get('eda_plots'):
                logger.info("✓ Layer 1: Statistical summary and plots found in state")
        elif agent_name == 'Data Cleaning':
            if state.get('cleaned_dataset') is not None or state.get('cleaning_summary'):
                logger.info("✓ Layer 1: Cleaned dataset and summary found in state")
        
    except Exception as e:
        logger.error(f"✗ Agent execution failed: {e}")
        flow_results['errors'].append(f"Execution failed: {e}")
        return flow_results
    
    # Step 2: Check Layer 2 execution
    logger.info("\n[Step 2] Checking Layer 2 execution...")
    
    # Check if Layer 2 was attempted by looking at result
    if isinstance(result, dict):
        layer_info = result.get('data', {}).get('layer', 'layer1')
        if layer_info == 'layer2':
            flow_results['layer2_code_generated'] = True
            flow_results['layer2_docker_executed'] = True
            logger.info("✓ Layer 2 code was generated and executed")
        else:
            logger.info(f"Layer used: {layer_info}")
    
    # Check for Docker containers
    logger.info("\n[Step 3] Checking Docker execution...")
    try:
        # Get recent sandbox containers
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}} {{.Status}} {{.CreatedAt}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            containers = result.stdout.strip().split('\n')
            # Get most recent
            if containers:
                latest_container = containers[-1].split()[0]
                logger.info(f"Latest sandbox container: {latest_container}")
                monitor_docker_execution(latest_container)
                flow_results['layer2_docker_executed'] = True
                
                # Check if container completed successfully
                if 'Exited (0)' in containers[-1]:
                    flow_results['layer2_docker_success'] = True
                    logger.info("✓ Docker container executed successfully")
                elif 'Exited' in containers[-1]:
                    logger.warning(f"⚠ Docker container exited with error: {containers[-1]}")
                else:
                    logger.info(f"Container status: {containers[-1]}")
        else:
            logger.warning("No sandbox containers found - Layer 2 may not have executed in Docker")
    except Exception as e:
        logger.warning(f"Could not check Docker containers: {e}")
    
    # Step 4: Extract results
    logger.info("\n[Step 4] Extracting results...")
    if isinstance(result, dict):
        flow_results['layer2_results'] = result.get('data', {})
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"{agent_name} Flow Summary:")
    logger.info(f"  Layer 1 Completed: {'✓' if flow_results['layer1_completed'] else '✗'}")
    logger.info(f"  Layer 2 Code Generated: {'✓' if flow_results['layer2_code_generated'] else '✗'}")
    logger.info(f"  Docker Executed: {'✓' if flow_results['layer2_docker_executed'] else '✗'}")
    logger.info(f"  Docker Success: {'✓' if flow_results['layer2_docker_success'] else '✗'}")
    logger.info("="*60)
    
    return flow_results


async def test_weatherAUS_full_flow():
    """Test full Layer 2 flow on weatherAUS dataset."""
    logger.info("="*80)
    logger.info("FULL LAYER 2 FLOW TEST - weatherAUS Dataset")
    logger.info("="*80)
    
    # Check Docker
    if not check_docker_setup():
        logger.error("Docker setup check failed!")
        return {'success': False, 'error': 'Docker setup failed'}
    
    # Load dataset
    logger.info("\n[Loading Dataset]")
    dataset_path = '/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project/Evaluation/datasets/real_world/weatherAUS.csv'
    target_col = 'RainTomorrow'
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return {'success': False, 'error': 'Dataset not found'}
    
    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Sample for faster testing
    if len(df) > 2000:
        logger.info(f"Sampling dataset from {len(df)} to 2000 rows...")
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:10]}...")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Create state
    logger.info("\n[Creating State]")
    base_test = BaseAgentTest()
    state = base_test.create_state(df, target_col)
    
    # Ensure target column is properly set
    state['target_column'] = target_col
    logger.info(f"State created with target_column: {state.get('target_column')}")
    
    # Test agents
    logger.info("\n[Testing Agents]")
    agents_to_test = [
        ('Data Discovery', DataDiscoveryAgent()),
        ('EDA Analysis', EDAAgent()),
        ('Data Cleaning', EnhancedDataCleaningAgent()),
    ]
    
    results = {}
    
    for agent_name, agent in agents_to_test:
        try:
            result = await test_agent_full_flow(agent, agent_name, state, dataset_path)
            results[agent_name] = result
            
            # Wait a bit between agents
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error testing {agent_name}: {e}", exc_info=True)
            results[agent_name] = {
                'agent': agent_name,
                'error': str(e),
                'layer1_completed': False,
                'layer2_code_generated': False,
                'layer2_docker_executed': False,
                'layer2_docker_success': False
            }
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FULL FLOW TEST SUMMARY")
    logger.info("="*80)
    
    for agent_name, result in results.items():
        logger.info(f"\n{agent_name}:")
        logger.info(f"  Layer 1: {'✓' if result.get('layer1_completed') else '✗'}")
        logger.info(f"  Code Generated: {'✓' if result.get('layer2_code_generated') else '✗'}")
        logger.info(f"  Docker Executed: {'✓' if result.get('layer2_docker_executed') else '✗'}")
        logger.info(f"  Docker Success: {'✓' if result.get('layer2_docker_success') else '✗'}")
        if result.get('errors'):
            logger.info(f"  Errors: {result['errors']}")
    
    # Save results
    output_path = Path("Evaluation/results/weatherAUS_full_flow_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return {'success': True, 'results': results}


if __name__ == "__main__":
    result = asyncio.run(test_weatherAUS_full_flow())
    if result.get('success'):
        print("\n✓ Full flow test completed!")
        print(f"Results: {result.get('results', {})}")
    else:
        print(f"\n✗ Test failed: {result.get('error')}")
        sys.exit(1)

