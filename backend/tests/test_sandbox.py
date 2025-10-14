"""
Tests for the Sandbox Executor Service

This module tests the functionality of the sandbox executor service.
"""

import unittest
import os
import sys
import time

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.sandbox_executor import SandboxExecutor


class TestSandboxExecutor(unittest.TestCase):
    """Test cases for the SandboxExecutor class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.executor = SandboxExecutor()
    
    def test_basic_execution(self):
        """Test basic code execution."""
        code = """
print("Hello, World!")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Hello, World!", result["output"])
    
    def test_cpu_intensive_code(self):
        """Test CPU-intensive code to verify resource limits."""
        code = """
import numpy as np

# CPU-intensive operation
size = 2000
matrix_a = np.random.rand(size, size)
matrix_b = np.random.rand(size, size)
result = np.dot(matrix_a, matrix_b)
print(f"Matrix multiplication complete. Result shape: {result.shape}")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Matrix multiplication complete", result["output"])
    
    def test_memory_intensive_code(self):
        """Test memory-intensive code to verify resource limits."""
        code = """
import numpy as np

# Try to allocate a large array (should be within memory limits)
try:
    # ~500MB array
    large_array = np.ones((65536, 1024), dtype=np.float32)
    print(f"Array created with shape {large_array.shape} and size {large_array.nbytes / 1024 / 1024:.2f} MB")
except MemoryError:
    print("Memory allocation failed")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Array created with shape", result["output"])
    
    def test_excessive_memory_usage(self):
        """Test code that tries to use too much memory."""
        code = """
import numpy as np

# Try to allocate a very large array (should exceed memory limits)
try:
    # ~2GB array
    very_large_array = np.ones((262144, 1024), dtype=np.float32)
    print(f"Array created with shape {very_large_array.shape}")
except MemoryError:
    print("Memory allocation failed as expected")
"""
        result = self.executor.execute_code(code)
        # Either it will succeed with the "failed as expected" message,
        # or it will be killed by the container's OOM killer
        self.assertTrue(
            result["status"] == "SUCCESS" and "Memory allocation failed as expected" in result["output"]
            or result["status"] == "FAILED"
        )
    
    def test_timeout(self):
        """Test code that runs for too long."""
        code = """
import time

# Sleep for longer than the timeout
print("Starting long operation...")
time.sleep(60)  # This should exceed the timeout
print("This should not be printed")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "TIMEOUT")
    
    def test_network_access(self):
        """Test that network access is blocked."""
        code = """
import socket

try:
    # Try to connect to an external server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("www.google.com", 80))
    print("Network access succeeded - THIS IS BAD!")
    s.close()
except Exception as e:
    print(f"Network access failed as expected: {str(e)}")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Network access failed as expected", result["output"])
    
    def test_file_system_isolation(self):
        """Test that the file system is properly isolated."""
        code = """
import os

# List files in the current directory
print("Files in current directory:")
print(os.listdir("."))

# Try to write to a file in /tmp (should be allowed)
try:
    with open("/tmp/test.txt", "w") as f:
        f.write("Hello, World!")
    print("Successfully wrote to /tmp/test.txt")
    
    # Read the file back
    with open("/tmp/test.txt", "r") as f:
        content = f.read()
    print(f"Read from file: {content}")
except Exception as e:
    print(f"File operation failed: {str(e)}")

# Try to write to a file in /app (should be read-only)
try:
    with open("/app/test.txt", "w") as f:
        f.write("This should fail")
    print("WARNING: Successfully wrote to /app/test.txt - THIS IS BAD!")
except Exception as e:
    print(f"Writing to /app failed as expected: {str(e)}")
"""
        result = self.executor.execute_code(code)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("Successfully wrote to /tmp/test.txt", result["output"])
        self.assertIn("Writing to /app failed as expected", result["output"])


if __name__ == "__main__":
    unittest.main()
