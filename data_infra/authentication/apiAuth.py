"""
apiAuth.py
----------
Handles the Financial Modeling Prep (FMP) API key. 
For now, we just store it in env file. You need the FMP_API_KEY for it to run.
"""

import os
from dotenv import load_dotenv

# Optional: load .env if you want
load_dotenv()

class APIAuth:
    """
    Minimal approach for retrieving FMP API key. 
    Currently returns a known default if no environment variable is set.
    """

    def __init__(self):
        api_key_env = os.getenv('FMP_API_KEY')
        if api_key_env:
            self.fmp_api_key = api_key_env 
        else:
            print("⚠️ No FMP_API_KEY environment variable set. Using default key.")

    def get_fmp_api_key(self) -> str:
        """
        Returns the FMP API key. Right now, it's a known constant.
        """
        return self.fmp_api_key