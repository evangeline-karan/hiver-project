"""
Diagnostic: list the models your Groq API key actually has access to right now.
Run this whenever a model name gives a 404 -- model availability changes over time.

Usage:
    python check_groq_models.py
"""
import os
import requests

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not set.")

resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()["data"]

print(f"{len(data)} models available to this key:\n")
for m in data:
    print(f"  {m['id']}")
