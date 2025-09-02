import logging

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.tradingOps.realtime.live import tradeExecutor # For live trading
# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import VolMomentum
from portfolios.portfolio_2.strategy import MomentumStrategy

from engines.run_engine import RunEngine


import pytest

def test_connection(db_connection):
    """
    Tests the database connection using the db_connection fixture.
    Pytest automatically passes the result of the db_connection() fixture
    into this test function as an argument.
    """
    # Arrange: The database connection is already provided by the fixture.
    
    # Act: Execute a simple query.
    query = "SELECT NOW();"
    response = db_connection.execute_query(query, fetch=True)
    
    # Assert: Check the results. These assertions determine if the test passes or fails.
    assert response['status'] == 'success', f"Query failed with message: {response.get('message')}"
    assert response['data'], "Query returned no data."
    assert 'now' in response['data'][0], "The 'now' key was not in the query result."

# You can add more tests that also use the same database connection fixture
# def test_another_feature(db_connection):
#     response = db_connection.execute_query("SELECT 1 AS number;", fetch=True)
#     assert response['status'] == 'success'
#     assert response['data'][0]['number'] == 1