# tests/test_samp.py

import pytest

# The test function simply "asks" for the fixture by name.
# Pytest will provide the database object from conftest.py.
def test_connection(db_connection):
    """
    Tests the database connection using the db_connection fixture.
    """
    query = "SELECT NOW();"
    response = db_connection.execute_query(query, fetch=True)
    
    assert response['status'] == 'success', f"Query failed with message: {response.get('message')}"
    assert response['data'], "Query returned no data."
    assert 'now' in response['data'][0], "The 'now' key was not in the query result."

def test_dummy():
    assert True