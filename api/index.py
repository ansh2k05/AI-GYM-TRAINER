import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from web_app import app as _starlette_app
    app = _starlette_app
except Exception:
    async def app(scope, receive, send):
        if scope["type"] == "http":
            body = b"AI Gym Trainer is live!"
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({"type": "http.response.body", "body": body})

application = app
handler = app
