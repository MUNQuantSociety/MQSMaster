# tests/conftest.py

import pytest
import sys
import os

# This path manipulation allows the import below to work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use the correct, full import path for your connector.
from src.common.database.MQSDBConnector import MQSDBConnector

@pytest.fixture(scope="module")
def db_connection():
    """
    A pytest fixture that creates and yields a database connector instance.
    """
    try:
        db = MQSDBConnector()
        yield db
    except Exception as e:
        pytest.fail(f"❌ Failed to initialize the MQSDBConnector: {e}")