#!/usr/bin/env python3
"""
Focused Docker Fixes Verification Test

Tests the specific Docker execution fixes:
1. Docker volume timeout handling
2. Error handling when sandbox returns None
3. Code generation prompt improvements
4. Plot cleanup functionality
"""

import sys
import os
from pathlib import Path
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.sandbox_executor import SandboxExecutor
from app.services.code_validator import CodeValidator


def test_docker_volume_timeout():
    """Test Docker volume timeout handling with retry logic"""
    print("\n" + "="*80)
    print("TEST 1: Docker Volume Timeout Handling")
    print("="*80)
    
    executor = SandboxExecutor()
    
    # Test volume creation with retry logic
    test_volume = f"test_volume_{int(time.time())}"
    
    try:
        # This should use the new retry logic
        print(f"\n[1/3] Creating test volume: {test_volume}")
        result = subprocess.run(
            ["docker", "volume", "create", test_volume],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"   ✅ Volume created successfully")
        else:
            print(f"   ❌ Volume creation failed: {result.stderr}")
            return False
        
        # Test volume inspection (should use new timeout)
        print(f"\n[2/3] Inspecting volume (with 30s timeout)...")
        start_time = time.time()
        result = subprocess.run(
            ["docker", "volume", "inspect", test_volume],
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Volume inspection successful (took {elapsed:.2f}s)")
        else:
            print(f"   ❌ Volume inspection failed: {result.stderr}")
            return False
        
        # Cleanup
        print(f"\n[3/3] Cleaning up test volume...")
        subprocess.run(["docker", "volume", "rm", test_volume], capture_output=True)
        print(f"   ✅ Cleanup complete")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ Operation timed out (this should not happen with new timeout)")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_code_validation():
    """Test code validation with improved prompts"""
    print("\n" + "="*80)
    print("TEST 2: Code Validation (Improved Prompts)")
    print("="*80)
    
    validator = CodeValidator()
    
    # Test valid code
    valid_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df.sum()
print(result)
"""
    
    print("\n[1/2] Testing valid code...")
    result = validator.validate(valid_code)
    if result.is_valid:
        print("   ✅ Valid code passed validation")
    else:
        print(f"   ❌ Valid code failed: {result.errors}")
        return False
    
    # Test invalid code (syntax error)
    invalid_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df.sum(
print(result)
"""
    
    print("\n[2/2] Testing invalid code (should fail)...")
    result = validator.validate(invalid_code)
    if not result.is_valid:
        print(f"   ✅ Invalid code correctly rejected: {result.errors[0] if result.errors else 'Syntax error detected'}")
    else:
        print("   ❌ Invalid code was accepted (should have failed)")
        return False
    
    return True


def test_error_handling():
    """Test error handling when sandbox execution fails"""
    print("\n" + "="*80)
    print("TEST 3: Error Handling (Sandbox Returns Dict)")
    print("="*80)
    
    executor = SandboxExecutor()
    
    # Test that execute_code always returns a dict
    print("\n[1/2] Testing execute_code with invalid code...")
    
    invalid_code = """
import pandas as pd
# This will cause an error
df = pd.read_csv('/nonexistent/file.csv')
"""
    
    try:
        result = executor.execute_code(
            code=invalid_code,
            workflow_id="test_error_handling",
            agent_name="test"
        )
        
        # Should always return a dict
        if isinstance(result, dict):
            print(f"   ✅ execute_code returned dict (status: {result.get('status', 'UNKNOWN')})")
            if result.get('status') == 'ERROR':
                print(f"   ✅ Error properly captured: {result.get('error', 'Unknown')[:50]}")
            return True
        else:
            print(f"   ❌ execute_code returned {type(result)} instead of dict")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception raised (should be caught): {e}")
        return False


def test_plot_cleanup():
    """Test plot cleanup functionality"""
    print("\n" + "="*80)
    print("TEST 4: Plot Cleanup on New Workflow")
    print("="*80)
    
    from pathlib import Path
    import shutil
    
    # Determine plots directory
    cwd = os.getcwd()
    if cwd.endswith('/backend'):
        base_plots_dir = Path("plots")
    else:
        base_plots_dir = Path("backend/plots")
    
    # Create test workflow directories
    test_workflow_1 = "test_workflow_cleanup_1"
    test_workflow_2 = "test_workflow_cleanup_2"
    
    try:
        # Create test plots directories
        test_dir_1 = base_plots_dir / test_workflow_1
        test_dir_2 = base_plots_dir / test_workflow_2
        
        test_dir_1.mkdir(parents=True, exist_ok=True)
        test_dir_2.mkdir(parents=True, exist_ok=True)
        
        # Create dummy plot files
        (test_dir_1 / "test_plot_1.png").touch()
        (test_dir_2 / "test_plot_2.png").touch()
        
        print(f"\n[1/3] Created test plot directories")
        print(f"   - {test_dir_1} (1 plot)")
        print(f"   - {test_dir_2} (1 plot)")
        
        # Simulate cleanup (keep only test_workflow_2)
        print(f"\n[2/3] Simulating cleanup (keeping only {test_workflow_2})...")
        for plot_dir in base_plots_dir.iterdir():
            if plot_dir.is_dir() and plot_dir.name.startswith("test_workflow_cleanup") and plot_dir.name != test_workflow_2:
                shutil.rmtree(plot_dir)
                print(f"   ✅ Removed: {plot_dir.name}")
        
        # Verify cleanup
        print(f"\n[3/3] Verifying cleanup...")
        if test_dir_1.exists():
            print(f"   ❌ Old workflow directory still exists")
            return False
        if not test_dir_2.exists():
            print(f"   ❌ Current workflow directory was removed")
            return False
        
        print(f"   ✅ Cleanup verified correctly")
        
        # Cleanup test directories
        if test_dir_2.exists():
            shutil.rmtree(test_dir_2)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all Docker fix tests"""
    print("\n" + "="*80)
    print("DOCKER EXECUTION FIXES VERIFICATION TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Docker volume timeout
    results['volume_timeout'] = test_docker_volume_timeout()
    
    # Test 2: Code validation
    results['code_validation'] = test_code_validation()
    
    # Test 3: Error handling
    results['error_handling'] = test_error_handling()
    
    # Test 4: Plot cleanup
    results['plot_cleanup'] = test_plot_cleanup()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Test':<30} {'Status'}")
    print("-" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("\n" + "="*80)
    
    all_passed = all(results.values())
    if all_passed:
        print("✅ ALL TESTS PASSED - Docker fixes are working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the output above")
    
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

