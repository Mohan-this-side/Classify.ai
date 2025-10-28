#!/usr/bin/env python3
"""
Basic ML Sandbox Test - Step 1

Just test if we can build and run the sandbox image.
"""

import subprocess
import time

def test_1_build_image():
    """Test building the sandbox Docker image"""
    print("=" * 60)
    print("TEST 1: Building ML Sandbox Image")
    print("=" * 60)
    
    cmd = [
        'docker', 'build',
        '-f', 'docker/Dockerfile.sandbox',
        '-t', 'ds-capstone-ml-sandbox',
        'backend'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Image built successfully")
        return True
    else:
        print("❌ Build failed:")
        print(result.stderr)
        return False

def test_2_run_container():
    """Test running the sandbox container"""
    print("\n" + "=" * 60)
    print("TEST 2: Running Sandbox Container")
    print("=" * 60)
    
    # Create volumes if they don't exist
    for volume in ['sandbox_code', 'sandbox_results', 'sandbox_data']:
        subprocess.run(['docker', 'volume', 'create', volume], 
                      capture_output=True)
    
    # Run container
    cmd = [
        'docker', 'run', '--rm',
        '--name', 'test-sandbox',
        '-v', 'sandbox_code:/app/code:ro',
        '-v', 'sandbox_results:/app/results',
        '-v', 'sandbox_data:/app/data:ro',
        '-e', 'MAX_EXECUTION_TIME=5',
        'ds-capstone-ml-sandbox',
        '/app/entrypoint.sh'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Start container in background
    container = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit
    time.sleep(3)
    
    # Check if container is running
    check_cmd = ['docker', 'ps', '--filter', 'name=test-sandbox', '--format', '{{.Status}}']
    status = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if 'Up' in status.stdout:
        print("✅ Container is running")
        subprocess.run(['docker', 'stop', 'test-sandbox'], capture_output=True)
        return True
    else:
        print("❌ Container failed to start")
        stdout, stderr = container.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        subprocess.run(['docker', 'rm', '-f', 'test-sandbox'], capture_output=True)
        return False

def main():
    print("\n" + "="*60)
    print("BASIC SANDBOX TESTS - STEP 1")
    print("="*60)
    
    # Test 1: Build image
    if not test_1_build_image():
        print("\n❌ Test 1 FAILED - Cannot proceed")
        return 1
    
    # Test 2: Run container
    if not test_2_run_container():
        print("\n⚠️  Test 2 FAILED - Check logs")
        return 1
    
    print("\n" + "="*60)
    print("✅ ALL BASIC TESTS PASSED")
    print("="*60)
    print("\nNext steps:")
    print("1. Test with simple Python code")
    print("2. Test with ML libraries")
    print("3. Integrate with agents")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())


