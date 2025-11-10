# get_refresh_token.py
import os
import urllib.parse as up
import praw
from dotenv import load_dotenv

# Fill these or load from env
CLIENT_ID = os.getenv('REDDIT_API_KEY')
# For Installed app use an empty string (""); for Web app use the real secret
CLIENT_SECRET = os.getenv('REDDIT_API_SECRET')
USER_AGENT = "script:stocknear-bot:v1.0 (by u/your_username)"
REDIRECT_URI = "http://localhost:5173"

# Scopes your bot needs (add/remove as necessary)
SCOPES = ["identity", "submit", "read", "flair"]

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    user_agent=USER_AGENT,
)

state = "random_state_string"  # can be anything
auth_url = reddit.auth.url(SCOPES, state, "permanent")
print("\n1) Open this URL, approve the app, and log in if asked:\n")
print(auth_url)

redirected = input("\n2) After approving, you'll be redirected. Copy the FULL URL from the address bar and paste it here:\n> ").strip()

# Extract ?code=... from the pasted URL
code = up.parse_qs(up.urlparse(redirected).query).get("code", [None])[0]
if not code:
    raise SystemExit("No 'code' found. Make sure you pasted the FULL redirect URL, including ?code=...")

refresh_token = reddit.auth.authorize(code)
print("\nSUCCESS! Your REFRESH TOKEN is:\n")
print(refresh_token)
print("\nStore this as REDDIT_REFRESH_TOKEN.\n")
