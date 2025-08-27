import os
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()


GRAPH_VERSION = "v23.0"  # bump as needed
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"

APP_ID = os.getenv('FACEBOOK_APP_ID')
APP_SECRET = os.getenv('FACEBOOK_APP_SECRET')

SHORT_LIVED_USER_TOKEN = ""

def app_access_token(app_id: str, app_secret: str) -> str:
    # App token is used for /debug_token (format app_id|app_secret)
    return f"{app_id}|{app_secret}"

def make_appsecret_proof(access_token: str, app_secret: str) -> str:
    digest = hmac.new(app_secret.encode("utf-8"),
                      msg=access_token.encode("utf-8"),
                      digestmod=hashlib.sha256).hexdigest()
    return digest

def graph_get(path: str, access_token: str, params: dict = None):
    params = params or {}
    params["access_token"] = access_token
    params["appsecret_proof"] = make_appsecret_proof(access_token, APP_SECRET)
    url = f"{GRAPH_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def exchange_for_long_lived(short_token: str) -> dict:
    """
    Returns: dict like {"access_token": "...", "token_type": "bearer", "expires_in": 5184000}
    """
    url = f"{GRAPH_BASE}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "fb_exchange_token": short_token,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def debug_token(input_token: str) -> dict:
    # Use app access token to inspect any token
    inspect_token = app_access_token(APP_ID, APP_SECRET)
    url = f"{GRAPH_BASE}/debug_token"
    params = {
        "input_token": input_token,
        "access_token": inspect_token,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_page_tokens(long_lived_user_token: str) -> list[dict]:

    data = graph_get("/me/accounts", long_lived_user_token)
    return data.get("data", [])

if __name__ == "__main__":
    # 1) Exchange short-lived user token -> long-lived user token
    exchange = exchange_for_long_lived(SHORT_LIVED_USER_TOKEN)
    LONG_LIVED_USER_TOKEN = exchange["access_token"]
    print("Long-lived user token:", LONG_LIVED_USER_TOKEN)
    print("Long-lived user token acquired. Expires in (seconds):", exchange.get("expires_in"))

    # 2) Verify token details
    info = debug_token(LONG_LIVED_USER_TOKEN)
    print("Token debug data:", info.get("data", {}))

    # 3) OPTIONAL: Get Page tokens (use these for posting as a Page)
    pages = get_page_tokens(LONG_LIVED_USER_TOKEN)
    for p in pages:
        # Do not log full tokens in real apps!
        print(f"Page: {p.get('name')} ({p.get('id')}) — token present: {bool(p.get('access_token'))}")

    # Example: making a secure Graph call with appsecret_proof included
    me = graph_get("/me", LONG_LIVED_USER_TOKEN, params={"fields": "id,name"})
    print("Me:", me)
