"""
memberAuth.py
-------------
Minimal in-memory multi-user authentication. 
No DB usage for now. Good for local dev & easy to extend later.
"""

import os
from dotenv import load_dotenv

# Load from local .env (optional) – but you can still do it if needed
load_dotenv()

class MemberAuth:
    """
    A dummy in-memory approach to user authentication.
    Future improvement: connect to a real DB table for persistent credentials.
    """

    def __init__(self):
        # Example in-memory store: {username: password}
        self.user_store = {}
        # In-memory set: users currently "logged in"
        self.logged_in_users = set()

    def add_user(self, username: str, password: str) -> dict:
        """
        Adds a user to the in-memory user_store. 
        In the future, you'd store this in a 'user_creds' table.
        """
        if username in self.user_store:
            return {'status': 'error', 'message': 'Username already exists.'}
        self.user_store[username] = password
        return {'status': 'success', 'message': f"User '{username}' added in memory."}

    def login(self, username: str, password: str) -> dict:
        """
        Verifies user credentials in memory. 
        If valid, marks them as logged in.
        """
        if username not in self.user_store:
            return {'status': 'error', 'message': f"User '{username}' not found."}
        if self.user_store[username] == password:
            self.logged_in_users.add(username)
            return {'status': 'success', 'message': f"User '{username}' logged in."}
        else:
            return {'status': 'error', 'message': 'Invalid password.'}

    def is_authorized(self, username: str) -> bool:
        """
        Checks if the user is currently logged in.
        """
        return username in self.logged_in_users

    def logout(self, username: str) -> dict:
        """
        Logs the user out (removes them from the logged_in_users set).
        """
        if username in self.logged_in_users:
            self.logged_in_users.remove(username)
            return {'status': 'success', 'message': f"User '{username}' logged out."}
        else:
            return {'status': 'error', 'message': f"User '{username}' is not logged in."}
