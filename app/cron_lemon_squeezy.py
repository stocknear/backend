import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from pocketbase import PocketBase
import time
# Load API keys and credentials from environment variables
load_dotenv()
api_key = os.getenv('LEMON_SQUEEZY_API_KEY')
pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

# Initialize and authenticate PocketBase client
pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)



async def get_subscription_data():
    """
    Fetch all subscription data from Lemon Squeezy via paginated API calls.
    Ensures that no more than 100 API requests are made per minute.
    """
    base_url = "https://api.lemonsqueezy.com/v1/subscriptions"
    headers = {
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
        "Authorization": f"Bearer {api_key}"
    }
    page = 1
    all_subscriptions = []
    request_count = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            # If we have made 100 requests, wait for 60 seconds before continuing
            if request_count > 0 and request_count % 100 == 0:
                print("Reached 100 API requests. Waiting for 60 seconds...")
                await asyncio.sleep(60)

            url = f"{base_url}?page[number]={page}&page[size]=100"
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status}, {error_text}")
                    break

                data = await response.json()
                # Append the subscription data from this page
                if "data" in data:
                    all_subscriptions.extend(data["data"])

                # If a next-page link exists, increment the page counter; otherwise, break the loop
                if "links" in data and data["links"].get("next"):
                    page += 1
                else:
                    break

            request_count += 1

    return all_subscriptions

async def run():
    all_users = pb.collection("users").get_full_list()
    users_by_email = {user.email: user for user in all_users if hasattr(user, 'email')}

    all_subscriptions = await get_subscription_data()
    print(f"Total Subscriptions: {len(all_subscriptions)}\n")
    
    # Group subscriptions by email, prioritizing "active" status first, then latest updated_at
    subscriptions_by_email = {}
    for sub in all_subscriptions:
        attributes = sub.get('attributes', {})
        user_email = attributes.get('user_email')
        if not user_email:
            continue
        
        status = attributes.get('status', '').lower()
        updated_at = attributes.get('updated_at')
        existing_sub = subscriptions_by_email.get(user_email)

        # First-time entry: always add
        if not existing_sub:
            subscriptions_by_email[user_email] = sub
            continue

        existing_attrs = existing_sub.get('attributes', {})
        existing_status = existing_attrs.get('status', '').lower()
        existing_updated = existing_attrs.get('updated_at')

        # Prioritize "active" status
        if status == 'active':
            if existing_status == 'active':
                # Both active: keep the newer one
                if updated_at > existing_updated:
                    subscriptions_by_email[user_email] = sub
            else:
                # Replace non-active with active
                subscriptions_by_email[user_email] = sub
        else:
            if existing_status != 'active':
                # Neither is active: keep the newer one
                if updated_at > existing_updated:
                    subscriptions_by_email[user_email] = sub

    # Process filtered subscriptions
    for user_email, sub in subscriptions_by_email.items():
        try:
            attributes = sub.get('attributes', {})
            status = attributes.get('status', 'N/A').lower()
            user = users_by_email.get(user_email)

            if status in ['expired', 'refunded']:
                if user and getattr(user, 'tier', None) == 'Pro' and not getattr(user, 'lifetime', False):
                    pb.collection('users').update(user.id, {
                        'tier': 'Free'
                        })
                    print(f"Downgraded: {user_email}")
        except:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(e)
