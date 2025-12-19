import requests
import json

try:
    resp = requests.get("https://docs.langchain.com/mcp")
    print(f"Status: {resp.status_code}")
    print(f"Headers: {resp.headers}")
    with open("mcp_json.txt", "w", encoding="utf-8") as f:
        json.dump(resp.json(), f, indent=2)
except Exception as e:
    with open("mcp_json.txt", "w", encoding="utf-8") as f:
        f.write(str(e))
