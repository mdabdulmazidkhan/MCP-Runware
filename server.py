import os
import uvicorn

# The MCP-Runware ASGI app is called "app" in the main file
# If your repo uses a different variable, replace "app" below
from runware_mcp_server import app  # ASGI app

if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8081))
    # Run Uvicorn with the ASGI app
    uvicorn.run(app, host="0.0.0.0", port=port)

