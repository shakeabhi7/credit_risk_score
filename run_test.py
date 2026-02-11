#!/usr/bin/env python
"""Day 1 Setup Test Script"""

import sys
import os

from src.monitoring.logger import setup_logger

logger = setup_logger("day1_test")

TEST_CSV_PATH = "data/raw/credit_risk_test.csv"


def test_imports():
    """Test all critical imports"""
    logger.info("=" * 6)
    logger.info("TESTING IMPORTS")
    logger.info("=" * 6)

    try:
        from src.monitoring.logger import setup_logger
        logger.info("Logger import successful")
    except Exception as e:
        logger.error(f"Logger import failed: {e}")
        return False

    try:
        from src.utils import load_config
        logger.info("Utils import successful")
    except Exception as e:
        logger.error(f"Utils import failed: {e}")
        return False

    try:
        from src.data.data_loader import DataLoader
        logger.info("DataLoader import successful")
    except Exception as e:
        logger.error(f"DataLoader import failed: {e}")
        return False

    try:
        from src.data.data_validator import DataValidator, CreditRiskInput
        logger.info(" DataValidator import successful")
    except Exception as e:
        logger.error(f"DataValidator import failed: {e}")
        return False

    try:
        from src.database.mongo_client import MongoDBClient
        logger.info(" MongoDBClient import successful")
    except Exception as e:
        logger.error(f" MongoDBClient import failed: {e}")
        return False

    return True


def test_configs():
    """Test YAML config files"""
    logger.info("=" * 6)
    logger.info("TESTING CONFIG FILES")
    logger.info("=" * 6)

    from src.utils import load_config

    configs = [
        "src/config/data_config.yaml",
        "src/config/model_config.yaml",
        "src/config/feature_config.yaml",
    ]

    for path in configs:
        try:
            load_config(path)
            logger.info(f" {path} loaded successfully")
        except Exception as e:
            logger.error(f" {path} failed: {e}")
            return False

    return True


def test_data_generation():
    """Test synthetic data generation"""
    logger.info("=" * 6)
    logger.info("TESTING SYNTHETIC DATA GENERATION")
    logger.info("=" * 6)

    try:
        from src.data.synthetic_data_generator import generate_synthetic_data
        df = generate_synthetic_data(100, TEST_CSV_PATH)
        logger.info(f" Generated {len(df)} records")
        return True
    except Exception as e:
        logger.error(f" Data generation failed: {e}")
        return False


def test_data_loader():
    """Test data loader"""
    logger.info("=" * 6)
    logger.info("TESTING DATA LOADER")
    logger.info("=" * 6)

    try:
        from src.data.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_csv(TEST_CSV_PATH)

        logger.info(f" Loaded {len(df)} records from CSV")
        logger.info(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        logger.error(f" DataLoader failed: {e}")
        return False


def test_data_validator():
    """Test data validator"""
    logger.info("=" * 6)
    logger.info("TESTING DATA VALIDATOR")
    logger.info("=" * 6)

    try:
        import pandas as pd
        from src.data.data_validator import DataValidator

        df = pd.read_csv(TEST_CSV_PATH)
        validator = DataValidator()
        validator.validate(df)

        logger.info(" Data validation successful")
        return True
    except Exception as e:
        logger.error(f" Validator failed: {e}")
        return False


def test_mongodb():
    """Test MongoDB connection (optional)"""
    logger.info("=" * 6)
    logger.info("TESTING MONGODB")
    logger.info("=" * 6)

    try:
        from src.database.mongo_client import MongoDBClient
        db = MongoDBClient()

        if getattr(db, "connected", False):
            logger.info(" MongoDB connected successfully")
        else:
            logger.warning(" MongoDB not running (optional)")
        return True
    except Exception as e:
        logger.error(f" MongoDB test failed: {e}")
        return False


def cleanup():
    """Remove test artifacts"""
    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)
        logger.info(" Cleaned up test CSV file")


def main():
    """Run all Day-1 tests"""
    logger.info("=" * 6)
    logger.info("DAY 1 SETUP TEST SUITE")
    logger.info("=" *6 )

    results = {
        "imports": test_imports(),
        "configs": test_configs(),
        "data_generation": test_data_generation(),
        "data_loader": test_data_loader(),
        "data_validator": test_data_validator(),
        "mongodb": test_mongodb(),
    }

    logger.info("=" * 6)
    logger.info("TEST SUMMARY")
    logger.info("=" * 6)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        logger.info(f"{' PASS' if result else ' FAIL'}: {name}")

    logger.info(f"PASSED: {passed}/{total}")

    cleanup()

    if passed == total:
        logger.info("ALL TESTS PASSED! DAY 1 SETUP COMPLETE ")
        return True
    else:
        logger.warning(f"\n {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        sys.exit(1)
