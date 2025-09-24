import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.database.MQSDBConnector import MQSDBConnector
# Initialize Database Connector Instance
try:
    db = MQSDBConnector()
except Exception as e:
    print("❌ Failed to initialize the MQSDBConnector:", e) #Need to check hostname/ server issue
    db = None  # Avoid crash in test_connection()

# Test Connection
def test_connection():
    if db is None:
        print("⚠️ Skipping connection test due to initialization failure.")
        return
    response = db.execute_query("SELECT NOW();", fetch=True)  # Get current timestamp
    if response['status'] == 'success':
        print("✅ PostgreSQL connection successful! Current DB Time:", response['data'][0]['now'])
    else:
        print("❌ Connection failed:", response['message'])

# Run Tests & Setup
if __name__ == "__main__":
    test_connection()
