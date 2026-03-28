import requests
import json
import logging
import time
import os
from src.monitoring.logger import setup_logger
from typing import Dict, Any
logger = setup_logger('test_api')

#configuration

API_URL = os.getenv("API_URL","http://localhost:8000")
REQUEST_TIMEOUT = 20

# Test cases

test_cases = [
    {
        "name": "Good Credit Customer",
        "data": {
            "age": 35,
            "income": 75000,
            "debt": 15000,
            "credit_limit": 50000,
            "credit_used": 20000,
            "employment_years": 8,
            "employment_type": "Salaried"
        }
    },
    {
        "name": "Bad Credit Customer",
        "data": {
            "age": 28,
            "income": 40000,
            "debt": 30000,
            "credit_limit": 25000,
            "credit_used": 24000,
            "employment_years": 2,
            "employment_type": "Self-employed"
        }
    },
    {
        "name": "High Income Customer",
        "data": {
            "age": 45,
            "income": 150000,
            "debt": 20000,
            "credit_limit": 100000,
            "credit_used": 10000,
            "employment_years": 15,
            "employment_type": "Salaried"
        }
    },
    {
        "name": "Young Professional",
        "data": {
            "age": 25,
            "income": 55000,
            "debt": 8000,
            "credit_limit": 35000,
            "credit_used": 5000,
            "employment_years": 3,
            "employment_type": "Salaried"
        }
    }
]

# Test Functions

def test_health() -> bool:
    """Test health check endpoint"""

    logger.info("-"*6)
    logger.info("Test: Health Check")
    logger.info("-"*6)

    try:
        start = time.time()

        response = requests.get(f"{API_URL}/health",timeout=REQUEST_TIMEOUT)
        latency = time.time() - start
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Time:{latency:.3f}s")

        if response.status_code == 200:
            data = response.json()

            logger.info("API status OK")
            logger.info(f"Version: {data.get('version')}")
            logger.info(f"Model Loaded: {data.get('models_loader')}")
            logger.info(f"Database Connected:{data.get('database_connected')}")

            return True
        
        logger.error("Health check failed")
        return False
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False

def test_prediction(test_case: Dict[str,Any])->bool:
    """Test predicton endpoint"""
    logger.info("-"*6)
    logger.info(f"Test: {test_case['name']}")
    logger.info('-'*6)
    try:
        logger.info("Input Data")

        for key, value in test_case["data"].items():
            logger.info(f"{key}:{value}")

        start = time.time()

        response = requests.post(f"{API_URL}/predict",json=test_case["data"],timeout=REQUEST_TIMEOUT)

        latency = time.time() - start
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Time: {latency:.3f}s")

        if response.status_code == 200:
            result = response.json()

            logger.info("Prediction successfull")
            logger.info(f"Customer ID: {result.get('customer_id')}")
            logger.info(f"Prediction:{result.get('predicted_class')}"
                        f"({'Good ' if result.get('predicted_class')==0 else 'Bad '}Credit)")
            logger.info(f"Probability: {result.get('probability'):.4f}")
            logger.info(f"Confidence: {result.get('confidence'):.4f}")
            logger.info(f"Model: {result.get('model_name')}")
            logger.info(f"Message: {result.get('message')}")
            
            return True
        
        logger.error("Prediction failed")
        logger.error(response.text)
        return False
    except Exception as e:
        logger.info(f"Prediction error:{e}")
        return False

def test_invalid_input() -> bool:

    """Test validation handling"""

    logger.info("=" * 6)
    logger.info("TEST: Invalid Input Validation")
    logger.info("=" * 6)

    invalid_data = {
        "age": -5,
        "income": 50000,
        "debt": 10000,
        "credit_limit": 30000,
        "credit_used": 15000,
        "employment_years": 5,
        "employment_type": "Salaried"
    }

    try:
        response = requests.post(f"{API_URL}/predict",json=invalid_data,timeout=REQUEST_TIMEOUT)
        if response.status_code == 422:
            logger.info("Validation error correctly caught")
            return True
        
        logger.warning(f"Excepted 422 but got {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Error:{e}")
        return False
    

# Main Test Suite
def main():
    logger.info("-"*6)
    logger.info("FASTAPI TEST SUITE")
    logger.info("-"*6)

    result = {}

    # check API availability
    try:
        response = requests.get(f"{API_URL}/health",timeout=5)
        if response.status_code != 200:
            logger.error("API not responding")
            logger.error("Start server with:")
            logger.error(
                "python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
            )
            return
    except Exception as e:
        logger.error(f"Cannot connect to API: {e}")
        return
    # run tests
    result["health"] = test_health()
    result["invalid_input"] = test_invalid_input()

    for case in test_cases:
        result[case['name']] = test_prediction(case)

    # Summary 
    logger.info("-"*6)
    logger.info("TEST SUMMARY")
    logger.info("-"*6)

    passed = sum(result.values())
    total = len(result)

    logger.info(f"Tests Passed: {passed}/{total}")

    for name, result in result.items():
        status = " PASS" if result else " FAIL"
        logger.info(f"{status} : {name}")

    if passed == total:
        logger.info("ALL TESTS PASSED")
    else:
        logger.warning(f" {total - passed} tests failed")


if __name__ == "__main__":
    main()



