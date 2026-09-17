import requests
import json

# 1. Read the raw text file (no need to format it as JSON first)
with open('input.txt', 'r', encoding='utf-8') as f:
    # If your input.txt already has the JSON brackets {} and "text": in it, 
    # you might want to remove those so input.txt is JUST the raw paper text!
    raw_text = f.read()

# 2. Automatically format it into a valid JSON payload
payload = {"text": raw_text}

print("Sending document to API... This might take a moment for a long text.")
response = requests.post("http://127.0.0.1:8000/simplify", json=payload)

# 3. Print the simplified result
if response.status_code == 200:
    result = response.json()
    print("\n--- SIMPLIFIED SUMMARY ---")
    print(result["simplified_text"])
else:
    print(f"Failed! Error {response.status_code}: {response.text}")
