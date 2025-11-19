"""
Quick script to check evaluation status and progress.
"""

import json
from pathlib import Path
from datetime import datetime

def check_status():
    """Check evaluation status."""
    results_path = Path("Evaluation/results/agent_level/all_agent_tests.json")
    
    print("="*80)
    print("EVALUATION STATUS CHECK")
    print("="*80)
    
    # Check if results file exists
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n✓ Results file found: {results_path}")
        print(f"  Total test results: {len(results)}")
        
        # Count by agent
        agent_counts = {}
        for result in results:
            agent = result.get('test_name', 'unknown')
            if agent not in agent_counts:
                agent_counts[agent] = {'total': 0, 'passed': 0}
            agent_counts[agent]['total'] += 1
            if result.get('passed', False):
                agent_counts[agent]['passed'] += 1
        
        print("\nPer-Agent Status:")
        for agent, counts in sorted(agent_counts.items()):
            rate = counts['passed'] / counts['total'] * 100 if counts['total'] > 0 else 0
            print(f"  {agent:25s}: {counts['passed']:2d}/{counts['total']:2d} ({rate:5.1f}%)")
        
        # Count by dataset
        dataset_counts = {}
        for result in results:
            dataset = result.get('dataset_name', 'unknown')
            if dataset not in dataset_counts:
                dataset_counts[dataset] = {'total': 0, 'passed': 0}
            dataset_counts[dataset]['total'] += 1
            if result.get('passed', False):
                dataset_counts[dataset]['passed'] += 1
        
        print("\nPer-Dataset Status:")
        for dataset, counts in sorted(dataset_counts.items()):
            rate = counts['passed'] / counts['total'] * 100 if counts['total'] > 0 else 0
            print(f"  {dataset:25s}: {counts['passed']:2d}/{counts['total']:2d} ({rate:5.1f}%)")
        
        # Expected total
        expected_agents = 8
        expected_datasets = 8
        expected_total = expected_agents * expected_datasets
        
        print(f"\nExpected: {expected_total} tests ({expected_agents} agents × {expected_datasets} datasets)")
        print(f"Current:  {len(results)} tests")
        
        if len(results) < expected_total:
            print(f"\n⚠ Evaluation still in progress... ({len(results)}/{expected_total} tests completed)")
        else:
            print("\n✓ Evaluation complete!")
    else:
        print(f"\n⚠ Results file not found: {results_path}")
        print("  Evaluation may still be running or hasn't started yet.")
    
    # Check log files
    log_files = list(Path("Evaluation").glob("comprehensive_eval_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLatest log: {latest_log}")
        print(f"  Size: {latest_log.stat().st_size / 1024:.1f} KB")
        print(f"  Modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
    
    print("\n" + "="*80)
    print("To monitor progress in real-time:")
    print(f"  tail -f {latest_log if log_files else 'Evaluation/comprehensive_eval_*.log'}")
    print("="*80)

if __name__ == "__main__":
    check_status()

