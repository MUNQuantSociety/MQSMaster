"""
apiAuth.py
----------
Handles the Financial Modeling Prep (FMP) API key. 
For now, we just store it in code or optionally in an env variable for future flexibility.
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
        # Option A: Hard-code for dev
        self.fmp_api_key = "taxFvdsV3ZQiBkff3fkxrAcatQV9C8wG"

        # Option B: (If you want an env-based fallback)
        # api_key_env = os.getenv('FMP_API_KEY')
        # self.fmp_api_key = api_key_env if api_key_env else "taxFvdsV3ZQiBkff3fkxrAcatQV9C8wG"

    def get_fmp_api_key(self) -> str:
        """
        Returns the FMP API key. Right now, it's a known constant.
        """
        return self.fmp_api_key
