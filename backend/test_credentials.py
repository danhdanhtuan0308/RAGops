import os
from google.oauth2 import service_account
import json

# Try to parse the file as JSON first
try:
    with open('gcp-credentials.json', 'r') as f:
        json_content = json.load(f)
    print("JSON parsing successful")
    print(f"Keys in the JSON: {list(json_content.keys())}")
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
except Exception as e:
    print(f"Error reading file: {e}")

# Try to load as service account credentials
try:
    credentials = service_account.Credentials.from_service_account_file('gcp-credentials.json')
    print("Successfully loaded credentials")
    print(f"Project ID: {credentials.project_id}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"General error: {e}")
