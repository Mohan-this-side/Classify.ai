#!/usr/bin/env python3
"""
Simple test script for the ML sandbox functionality.
This script tests the SandboxExecutor class and validates the sandbox configuration.
"""

import sys
import os
import time
import traceback

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_sandbox_import():
    """Test if we can import the SandboxExecutor class."""
    try:
        from app.services.sandbox_executor import SandboxExecutor
        print("✅ SandboxExecutor imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import SandboxExecutor: {e}")
        return False

def test_sandbox_initialization():
    """Test if we can initialize the SandboxExecutor."""
    try:
        from app.services.sandbox_executor import SandboxExecutor
        executor = SandboxExecutor()
        print("✅ SandboxExecutor initialized successfully")
        print(f"   - Sandbox image: {executor.sandbox_image}")
        print(f"   - Timeout: {executor.timeout} seconds")
        print(f"   - Memory limit: {executor.memory_limit}")
        print(f"   - CPU limit: {executor.cpu_limit}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize SandboxExecutor: {e}")
        traceback.print_exc()
        return False

def test_docker_availability():
    """Test if Docker is available and working."""
    try:
        import subprocess
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker is available and running")
            return True
        else:
            print(f"❌ Docker is not working: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker command timed out")
        return False
    except FileNotFoundError:
        print("❌ Docker command not found")
        return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def test_docker_volumes():
    """Test if the required Docker volumes exist."""
    try:
        import subprocess
        volumes = ['sandbox_code', 'sandbox_results', 'sandbox_data']
        for volume in volumes:
            result = subprocess.run(['docker', 'volume', 'inspect', volume], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Volume '{volume}' exists")
            else:
                print(f"❌ Volume '{volume}' does not exist")
                return False
        return True
    except Exception as e:
        print(f"❌ Error checking Docker volumes: {e}")
        return False

def test_sandbox_image():
    """Test if the sandbox image exists."""
    try:
        import subprocess
        result = subprocess.run(['docker', 'images', 'ds-capstone-ml-sandbox'], 
                              capture_output=True, text=True, timeout=10)
        if 'ds-capstone-ml-sandbox' in result.stdout:
            print("✅ Sandbox image exists")
            return True
        else:
            print("❌ Sandbox image does not exist")
            print("   You may need to build it with: docker build -f docker/Dockerfile.sandbox -t ds-capstone-ml-sandbox backend")
            return False
    except Exception as e:
        print(f"❌ Error checking sandbox image: {e}")
        return False

def test_simple_code_execution():
    """Test simple code execution in the sandbox (if image exists)."""
    try:
        from app.services.sandbox_executor import SandboxExecutor
        
        # Simple test code
        test_code = """
import sys
print(f"Python version: {sys.version}")
print("Hello from the sandbox!")
"""
        
        executor = SandboxExecutor()
        print("🔄 Attempting to execute simple code in sandbox...")
        
        # This will fail if the image doesn't exist, but we can test the logic
        try:
            result = executor.execute_code(test_code)
            if result['status'] == 'SUCCESS':
                print("✅ Code executed successfully in sandbox")
                print(f"   Output: {result.get('output', '')}")
                return True
            else:
                print(f"❌ Code execution failed: {result.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Sandbox execution error: {e}")
            print("   This is expected if the Docker image hasn't been built yet")
            return False
            
    except Exception as e:
        print(f"❌ Error in simple code execution test: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing ML Sandbox Configuration")
    print("=" * 50)
    
    tests = [
        ("Import SandboxExecutor", test_sandbox_import),
        ("Initialize SandboxExecutor", test_sandbox_initialization),
        ("Check Docker Availability", test_docker_availability),
        ("Check Docker Volumes", test_docker_volumes),
        ("Check Sandbox Image", test_sandbox_image),
        ("Test Simple Code Execution", test_simple_code_execution),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The sandbox is ready to use.")
    elif passed >= total - 1:  # All except the image test
        print("⚠️  Almost ready! You just need to build the Docker image.")
        print("   Run: docker build -f docker/Dockerfile.sandbox -t ds-capstone-ml-sandbox backend")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
