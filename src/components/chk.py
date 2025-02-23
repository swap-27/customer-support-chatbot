import requests
import os

api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
api_key = os.getenv("HF_API_KEY")

headers = {"Authorization": f"Bearer {api_key}"}

data = {"inputs": "Hello, how can I help you?"}

response = requests.post(api_url, headers=headers, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())