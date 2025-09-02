# tests/conftest.py

import pytest
# No more sys or os imports needed for path manipulation!

from data_infra.database.MQSDBConnector import MQSDBConnector

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