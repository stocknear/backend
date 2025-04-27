import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from pocketbase import PocketBase
from dateutil.parser import isoparse

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
            # Limit API request rate if necessary
            if request_count > 0 and request_count % 200 == 0:
                print("Reached 100 API requests. Waiting for 60 seconds...")
                await asyncio.sleep(60)

            # Note: Do not include the meta parameter
            url = f"{base_url}?page[number]={page}&page[size]=100"
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status}, {error_text}")
                    break

                data = await response.json()

                # Extract subscription data and check for custom meta in attributes
                if "data" in data:
                    for item in data["data"]:
                        subscription = {
                            "id": item.get("id"),
                            "attributes": item.get("attributes", {}),
                        }
                        # Access your custom meta if it exists under a key like 'custom_meta'
                        if "custom_meta" in subscription["attributes"]:
                            subscription["custom_meta"] = subscription["attributes"]["custom_meta"]
                        all_subscriptions.append(subscription)

                # Handle pagination if a next-page link is available
                if "links" in data and data["links"].get("next"):
                    page += 1
                else:
                    break

            request_count += 1

    return all_subscriptions


async def run():

    # Fetch all users (assumes pb.collection(...).get_full_list() is synchronous)
    all_users = pb.collection("users").get_full_list()
    users_by_email = {user.email: user for user in all_users if hasattr(user, 'email')}
    
    # Fetch all subscriptions (awaited)
    all_subscriptions = await get_subscription_data()
    print(f"Total Subscriptions: {len(all_subscriptions)}\n")
    
    # Group subscriptions by email.
    # For each email, we want to always choose an active subscription over a non-active one.
    # If both subscriptions are of the same "activity" level, we choose the one with the later updated_at.
    subscriptions_by_email = {}
    for sub in all_subscriptions:
        attributes = sub.get('attributes', {})
        user_email = attributes.get('user_email')
        if not user_email:
            continue
        
        status = attributes.get('status', '').lower()
        updated_at_str = attributes.get('updated_at')
        try:
            updated_at = isoparse(updated_at_str) if updated_at_str else None
        except Exception as e:
            print(f"Error parsing updated_at ({updated_at_str}) for {user_email}: {e}")
            updated_at = None
        
        existing_sub = subscriptions_by_email.get(user_email)
        if not existing_sub:
            subscriptions_by_email[user_email] = sub
            continue
        
        # Get info from the already saved subscription
        existing_attrs = existing_sub.get('attributes', {})
        existing_status = existing_attrs.get('status', '').lower()
        existing_updated_str = existing_attrs.get('updated_at')
        try:
            existing_updated = isoparse(existing_updated_str) if existing_updated_str else None
        except Exception as e:
            print(f"Error parsing existing updated_at ({existing_updated_str}) for {user_email}: {e}")
            existing_updated = None
        
        # If the new sub is active, it should win over any non-active subscription.
        if status == 'active':
            if existing_status == 'active':
                # Both active: keep the one with the later updated_at.
                if updated_at and existing_updated and updated_at > existing_updated:
                    subscriptions_by_email[user_email] = sub
            else:
                # Replace a non-active subscription with an active one.
                subscriptions_by_email[user_email] = sub
        else:
            # New sub is not active.
            if existing_status != 'active':
                # Both are non-active: choose the one with the later updated_at.
                if updated_at and existing_updated and updated_at > existing_updated:
                    subscriptions_by_email[user_email] = sub
                # If one of the dates is missing, you might add extra logic here.
    
    # Process the subscriptions after grouping.
    for user_email, sub in subscriptions_by_email.items():
        try:
            attributes = sub.get('attributes', {})
            status = attributes.get('status', 'N/A').lower()
            user = users_by_email.get(user_email)
            
            if status in ['expired', 'refunded']:
                # Example logic: downgrade a Pro/Plus user if not lifetime and subscription is expired/refunded.
                if user and (getattr(user, 'tier', None) == 'Pro' or getattr(user, 'tier', None) == 'Plus') and not getattr(user, 'lifetime', False):
                    # Uncomment the line below to perform the update:
                    # pb.collection('users').update(user.id, {'tier': 'Free'})
                    print(f"Downgraded: {user_email}")
                    #print(attributes)

            
        except Exception as e:
            print(f"Error processing user {user_email}: {e}")


    ##finding users who have no payment data but have been manually upgraded to pro or plus
    all_payment_items = pb.collection('payments').get_full_list()
    all_payment_user_id = [item.user for item in all_payment_items if hasattr(item, 'user')]
    
    print("====USERS WITHOUT PAYMENT DATA===")
    for user in all_users:
        if user.id not in all_payment_user_id and (getattr(user, 'tier', None) == 'Pro' or getattr(user, 'tier', None) == 'Plus'):
            print(f"Downgraded: {user.email}")



if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(e)
