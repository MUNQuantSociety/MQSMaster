"""
apiAuth.py
----------
Manages authentication credentials for external APIs, such as Alpaca.
Alpaca keys are stored in a separate env file located in data_infra/alpaca.env
"""

import os
from dotenv import load_dotenv

# Explicitly load the alpaca.env located in data_infra/
# Adjust the path if needed, depending on your exact project structure.
load_dotenv(dotenv_path='data_infra/alpaca.env')

class APIAuth:
    """
    Handles loading or refreshing of API keys for external services.
    Here we focus on Alpaca, stored in data_infra/alpaca.env
    """

    def __init__(self):
        self.alpaca_key_id = os.getenv('ALPACA_API_KEY_ID')
        self.alpaca_secret_key = os.getenv('ALPACA_API_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    def get_alpaca_credentials(self) -> dict:
        """
        Returns a dict containing the Alpaca credentials and base URL.
        """
        return {
            'API_KEY_ID': self.alpaca_key_id,
            'API_SECRET_KEY': self.alpaca_secret_key,
            'BASE_URL': self.alpaca_base_url
        }
