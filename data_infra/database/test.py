import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from MQSDBConnector import MQSDBConnector


# Initialize Database Connector Instance
db = MQSDBConnector()

# Test Connection
def test_connection():
    response = db.execute_query("SELECT NOW();", fetch=True)  # Get current timestamp
    if response['status'] == 'success':
        print("✅ PostgreSQL connection successful! Current DB Time:", response['data'][0]['now'])
    else:
        print("❌ Connection failed:", response['message'])

# Run Tests & Setup
if __name__ == "__main__":
    test_connection()
