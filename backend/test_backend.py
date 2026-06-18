import urllib.request
import urllib.error

try:
    req = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
    print("Status:", req.status)
    print("Body:", req.read().decode())
except Exception as e:
    print("ERR:", e)
