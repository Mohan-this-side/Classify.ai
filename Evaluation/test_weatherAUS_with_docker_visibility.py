"""
Test Full Layer 2 Flow with Docker Visibility
Shows Docker execution details and extracts results properly.
"""

import sys
import asyncio
import logging
import json
import pandas as pd
import subprocess
import time
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
from app.workflows.state_management import ClassificationState, WorkflowStatus, state_manager
from app.services.sandbox_executor import SandboxExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Evaluation/weatherAUS_docker_visibility.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_docker_and_volumes():
    """Check Docker and volumes."""
    logger.info("Checking Docker setup...")
    
    # Check Docker
    try:
        subprocess.run(['docker', 'ps'], check=True, capture_output=True, timeout=5)
        logger.info("✓ Docker daemon is running")
    except:
        logger.error("✗ Docker daemon not accessible")
        return False
    
    # Check image
    try:
        result = subprocess.run(['docker', 'images', 'ds-capstone-ml-sandbox'], capture_output=True, text=True, timeout=5)
        if 'ds-capstone-ml-sandbox' in result.stdout:
            logger.info("✓ Sandbox image exists")
        else:
            logger.error("✗ Sandbox image not found")
            return False
    except:
        return False
    
    # Ensure volumes
    for volume in ['sandbox_code', 'sandbox_results', 'sandbox_data']:
        try:
            subprocess.run(['docker', 'volume', 'inspect', volume], capture_output=True, timeout=5, check=False)
        except:
            subprocess.run(['docker', 'volume', 'create', volume], check=True)
    
    return True


def get_docker_results(container_name: str):
    """Get results from Docker container before it's removed."""
    results = {
        'container_name': container_name,
        'status': 'UNKNOWN',
        'output': '',
        'error': '',
        'logs': ''
    }
    
    try:
        # Get container logs
        log_result = subprocess.run(
            ['docker', 'logs', container_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        results['logs'] = log_result.stdout + log_result.stderr
        
        # Get container status
        status_result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.State.Status}}', container_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        results['status'] = status_result.stdout.strip()
        
        # Try to get results from volume
        executor = SandboxExecutor()
        try:
            volume_results = executor._get_results()
            results.update(volume_results)
        except Exception as e:
            logger.warning(f"Could not get volume results: {e}")
        
    except Exception as e:
        logger.warning(f"Could not get Docker results: {e}")
    
    return results


async def test_agent_with_docker_visibility(agent, agent_name: str, state: ClassificationState):
    """Test agent with full Docker visibility."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {agent_name} - Full Docker Visibility")
    logger.info(f"{'='*80}")
    
    flow_info = {
        'agent': agent_name,
        'layer1_completed': False,
        'layer2_prompt_generated': False,
        'layer2_code_generated': False,
        'layer2_code_validated': False,
        'layer2_docker_executed': False,
        'layer2_docker_success': False,
        'docker_container': None,
        'docker_output': None,
        'docker_error': None,
        'final_result': None
    }
    
    # Track container name before execution
    containers_before = set()
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        containers_before = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except:
        pass
    
    # Execute agent
    logger.info("\n[Step 1] Executing agent...")
    try:
        result = await agent.execute(state)
        flow_info['layer1_completed'] = True
        flow_info['final_result'] = result
        
        logger.info("✓ Agent execution completed")
        
    except Exception as e:
        logger.error(f"✗ Agent execution failed: {e}")
        flow_info['error'] = str(e)
        return flow_info
    
    # Check for new containers
    logger.info("\n[Step 2] Checking Docker execution...")
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'name=sandbox-', '--format', '{{.Names}} {{.Status}} {{.CreatedAt}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            containers_after = set(line.split()[0] for line in result.stdout.strip().split('\n') if line.strip())
            new_containers = containers_after - containers_before
            
            if new_containers:
                # Get the most recent container
                latest_container = list(new_containers)[-1]
                flow_info['docker_container'] = latest_container
                flow_info['layer2_docker_executed'] = True
                
                logger.info(f"✓ Found Docker container: {latest_container}")
                
                # Get detailed results
                docker_results = get_docker_results(latest_container)
                flow_info['docker_output'] = docker_results.get('output', '')
                flow_info['docker_error'] = docker_results.get('error', '')
                flow_info['docker_logs'] = docker_results.get('logs', '')
                flow_info['docker_status'] = docker_results.get('status', '')
                
                logger.info(f"\nDocker Container Details:")
                logger.info(f"  Container: {latest_container}")
                logger.info(f"  Status: {docker_results.get('status', 'UNKNOWN')}")
                logger.info(f"  Status Code: {docker_results.get('status', 'UNKNOWN')}")
                
                if docker_results.get('output'):
                    logger.info(f"\nDocker Output (first 500 chars):")
                    logger.info(docker_results['output'][:500])
                
                if docker_results.get('error'):
                    logger.info(f"\nDocker Error (first 500 chars):")
                    logger.info(docker_results['error'][:500])
                
                if docker_results.get('status') == 'SUCCESS' or 'SUCCESS' in docker_results.get('status_code', ''):
                    flow_info['layer2_docker_success'] = True
                    logger.info("✓ Docker execution succeeded!")
                else:
                    logger.warning("⚠ Docker execution may have failed")
                    
            else:
                logger.warning("No new Docker containers found")
        else:
            logger.warning("No Docker containers found")
    except Exception as e:
        logger.warning(f"Could not check Docker containers: {e}")
    
    # Check Layer 2 code generation and success from state
    logger.info("\n[Step 3] Checking Layer 2 code generation and success...")
    
    # Check state for Layer 2 success indicators
    layer2_success = False
    layer2_code_generated = False
    
    # Check for layer2_success flag in state (set by agents when Layer 2 succeeds)
    for key in state.keys():
        if 'layer2' in key.lower() and 'success' in key.lower():
            if state.get(key) is True:
                layer2_success = True
                layer2_code_generated = True
                logger.info(f"✓ Found Layer 2 success indicator in state: {key}")
                break
    
    # Also check for sandbox_execution_time (set when Docker executes successfully)
    if state.get('sandbox_execution_time') is not None:
        layer2_success = True
        layer2_code_generated = True
        logger.info(f"✓ Found sandbox_execution_time in state: {state.get('sandbox_execution_time')}s")
    
    # If Docker was executed, Layer 2 code was generated
    if flow_info['layer2_docker_executed']:
        layer2_code_generated = True
        logger.info("✓ Docker container was created - Layer 2 code was generated")
    
    # If Docker succeeded, Layer 2 succeeded
    if flow_info['layer2_docker_success']:
        layer2_success = True
    
    flow_info['layer2_code_generated'] = layer2_code_generated
    flow_info['layer2_docker_success'] = layer2_success or flow_info['layer2_docker_success']
    
    if layer2_code_generated:
        logger.info("✓ Layer 2 code was generated and executed")
    if layer2_success:
        logger.info("✓ Layer 2 execution succeeded!")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"{agent_name} Flow Summary:")
    logger.info(f"  Layer 1: {'✓' if flow_info['layer1_completed'] else '✗'}")
    logger.info(f"  Code Generated: {'✓' if flow_info['layer2_code_generated'] else '✗'}")
    logger.info(f"  Docker Executed: {'✓' if flow_info['layer2_docker_executed'] else '✗'}")
    logger.info(f"  Docker Success: {'✓' if flow_info['layer2_docker_success'] else '✗'}")
    if flow_info['docker_container']:
        logger.info(f"  Container: {flow_info['docker_container']}")
    logger.info("="*60)
    
    return flow_info


async def test_weatherAUS_full_flow():
    """Test full Layer 2 flow on weatherAUS with Docker visibility."""
    logger.info("="*80)
    logger.info("FULL LAYER 2 FLOW TEST - weatherAUS Dataset (with Docker Visibility)")
    logger.info("="*80)
    
    # Check Docker
    if not check_docker_and_volumes():
        return {'success': False, 'error': 'Docker setup failed'}
    
    # Load dataset
    logger.info("\n[Loading Dataset]")
    dataset_path = '/Users/mohan/NEU/FALL 2025/AGENTS V1/ds-capstone-project/Evaluation/datasets/real_world/weatherAUS.csv'
    target_col = 'RainTomorrow'
    
    df = pd.read_csv(dataset_path)
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    logger.info(f"Dataset: {df.shape}, Target: {target_col}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Create state
    base_test = BaseAgentTest()
    state = base_test.create_state(df, target_col)
    state['target_column'] = target_col
    
    # Test agents
    logger.info("\n[Testing Agents with Docker Visibility]")
    agents_to_test = [
        ('Data Discovery', DataDiscoveryAgent()),
        ('EDA Analysis', EDAAgent()),
        ('Data Cleaning', EnhancedDataCleaningAgent()),
    ]
    
    results = {}
    
    for agent_name, agent in agents_to_test:
        try:
            result = await test_agent_with_docker_visibility(agent, agent_name, state)
            results[agent_name] = result
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error testing {agent_name}: {e}", exc_info=True)
            results[agent_name] = {'agent': agent_name, 'error': str(e)}
    
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
        if result.get('docker_container'):
            logger.info(f"  Container: {result['docker_container']}")
        if result.get('docker_output'):
            logger.info(f"  Output length: {len(result['docker_output'])} chars")
        if result.get('docker_error'):
            logger.info(f"  Error: {result['docker_error'][:200]}")
    
    # Save results
    output_path = Path("Evaluation/results/weatherAUS_docker_visibility_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return {'success': True, 'results': results}


if __name__ == "__main__":
    result = asyncio.run(test_weatherAUS_full_flow())
    if result.get('success'):
        print("\n✓ Full flow test completed!")
    else:
        print(f"\n✗ Test failed: {result.get('error')}")
        sys.exit(1)

