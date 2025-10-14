#!/usr/bin/env python3
"""
🧪 Test Script for DS Capstone Multi-Agent System

This script tests the basic functionality of the system to ensure it's working properly.
"""

import requests
import time
import json
import pandas as pd
import io

# Test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            print("✅ Backend health check passed")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_workflow_start():
    """Test workflow start endpoint"""
    try:
        # Create a simple test dataset
        test_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        # Save to CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Prepare form data
        files = {
            'file': ('test_data.csv', csv_data, 'text/csv')
        }
        data = {
            'target_column': 'target',
            'description': 'Test classification workflow',
            'api_key': 'dummy_key',
            'user_id': 'test_user'
        }
        
        response = requests.post(f"{BACKEND_URL}/api/workflow/start", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow started successfully: {result.get('workflow_id')}")
            return result.get('workflow_id')
        else:
            print(f"❌ Workflow start failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Workflow start test failed: {e}")
        return None

def test_workflow_status(workflow_id):
    """Test workflow status endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/workflow/status/{workflow_id}")
        
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Workflow status retrieved: {status.get('status')} - {status.get('progress')}%")
            return status
        else:
            print(f"❌ Workflow status failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Workflow status test failed: {e}")
        return None

def test_workflow_progress(workflow_id):
    """Test workflow progress over time"""
    print("🔄 Monitoring workflow progress...")
    
    for i in range(10):  # Check 10 times over 20 seconds
        status = test_workflow_status(workflow_id)
        if status:
            current_agent = status.get('current_phase', 'Unknown')
            progress = status.get('progress', 0)
            print(f"  Step {i+1}: {current_agent} - {progress}%")
            
            if status.get('status') in ['completed', 'failed']:
                print(f"✅ Workflow finished with status: {status.get('status')}")
                return status
        
        time.sleep(2)
    
    print("⏰ Workflow monitoring timeout")
    return None

def main():
    """Run all tests"""
    print("🚀 Starting DS Capstone Multi-Agent System Tests")
    print("=" * 50)
    
    # Test 1: Backend Health
    if not test_backend_health():
        print("❌ Backend is not running. Please start it with: uvicorn app.main:app --reload")
        return
    
    # Test 2: Start Workflow
    workflow_id = test_workflow_start()
    if not workflow_id:
        print("❌ Failed to start workflow")
        return
    
    # Test 3: Monitor Progress
    final_status = test_workflow_progress(workflow_id)
    
    if final_status and final_status.get('status') == 'completed':
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Tests completed with issues. Check the logs for details.")

if __name__ == "__main__":
    main()
