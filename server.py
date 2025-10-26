import os
import uvicorn
import importlib

# --------- CONFIG ---------
# Replace 'runware_mcp_server' with the name of the file that contains your ASGI app
MAIN_FILE = "runware_mcp_server"  # e.g., if your main ASGI app is in main.py, write "main"
APP_NAME = "app"                  # The variable name of your ASGI app inside that file
# --------------------------

# Dynamically import the app
module = importlib.import_module(MAIN_FILE)
app = getattr(module, APP_NAME)

if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
