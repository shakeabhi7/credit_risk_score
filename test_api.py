import requests
import json
import logging
import time
import os
from src.monitoring.logger import setup_logger

logger = setup_logger('test_api')

#configuration

API_URL = os.getenv("API_URL","https://localhost:8000")
REQUEST_TIMEOUT = 20