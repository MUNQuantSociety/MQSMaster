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
    """Lightweight accessor for the FMP API key.

    Issues fixed:
      - Previously `get_fmp_api_key` returned the *method object* (bug) instead of the key string.
      - Added explicit validation & clearer error messaging.
    """

    def __init__(self):
        api_key_env = os.getenv('FMP_API_KEY')
        self.fmp_api_key = (api_key_env or '').strip()
        if not self.fmp_api_key:
            # Deliberately avoid silently using a hard‑coded default; force visibility.
            print("⚠️ FMP_API_KEY not set or empty. Set it in your environment or .env file.")

    def get_fmp_api_key(self) -> str:
        """Return the loaded API key string ('' if unset)."""
        return self.fmp_api_key