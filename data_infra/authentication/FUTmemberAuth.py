"""
memberAuth.py
-------------
Minimal multi-user authentication based on a user_creds table in PostgreSQL.
Passwords are stored in plaintext (NOT recommended in real production).
"""

import os
from dotenv import load_dotenv
from data_infra.database.MQSDBConnector import MQSDBConnector

# Load from the project's root .env
load_dotenv()

class MemberAuth:
    """
    For each user, we read their username/password from .env (MY_USER, MY_PASS).
    Then we compare against the user_creds table in PostgreSQL.
    If it matches, we store them in authorized_users in memory.
    """

    def __init__(self):
        # In-memory dict: {username: True} for users who have successfully logged in this session
        self.authorized_users = {}

        # Connect to DB (which has the user_creds table)
        self.db_connector = MQSDBConnector()
        result = self.db_connector.connect()
        if result['status'] == 'error':
            print("Error connecting to DB in MemberAuth:", result['message'])

        # By default, read MY_USER, MY_PASS from .env (each user sets these locally)
        self.local_user = os.getenv('MY_USER')
        self.local_pass = os.getenv('MY_PASS')

    def login(self, username: str = None, password: str = None) -> dict:
        """
        Checks if the provided (or .env-based) username/password is in user_creds table.
        If found, adds them to authorized_users in memory.
        """
        if username is None:
            username = self.local_user
        if password is None:
            password = self.local_pass

        if not username or not password:
            return {
                'status': 'error',
                'message': 'No username/password provided or set in .env'
            }

        # Attempt to retrieve from DB
        try:
            # We'll do a simple SELECT. In real usage, you might want parameter binding.
            sql = f"""
                SELECT * FROM user_creds
                WHERE username = '{username}' AND password = '{password}'
                LIMIT 1;
            """
            res = self.db_connector.read_db(sql=sql)
            if res['status'] == 'error':
                return res  # pass the error along

            if res['data'] and len(res['data']) > 0:
                # We found a match
                self.authorized_users[username] = True
                return {
                    'status': 'success',
                    'message': f"User '{username}' authenticated successfully."
                }
            else:
                return {
                    'status': 'error',
                    'message': f"No matching user found for '{username}'."
                }

        except Exception as ex:
            return {
                'status': 'error',
                'message': str(ex)
            }

    def is_authorized(self, username: str = None) -> bool:
        """
        Checks if the user is currently marked authorized in this session's memory.
        If username not provided, uses MY_USER from .env
        """
        if username is None:
            username = self.local_user
        return self.authorized_users.get(username, False)

    def logout(self, username: str = None) -> dict:
        """
        Logs the user out by removing them from authorized_users.
        """
        if username is None:
            username = self.local_user

        if username in self.authorized_users:
            del self.authorized_users[username]
            return {'status': 'success', 'message': f"User '{username}' logged out."}
        else:
            return {'status': 'error', 'message': f"User '{username}' not currently logged in."}

    def add_user(self, username: str, password: str) -> dict:
        """
        Adds a new user to the user_creds table with plaintext password (for demonstration).
        """
        # Check if user already exists
        check_sql = f"SELECT * FROM user_creds WHERE username = '{username}' LIMIT 1;"
        existing = self.db_connector.read_db(sql=check_sql)
        if existing['status'] == 'error':
            return existing
        if existing['data']:
            return {'status': 'error', 'message': 'Username already exists.'}

        # Insert new user
        insert_res = self.db_connector.inject_to_db(
            table='user_creds',
            data={'username': username, 'password': password}
        )
        return insert_res
