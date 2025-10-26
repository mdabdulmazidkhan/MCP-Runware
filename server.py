import os
import uvicorn
from runware_mcp_server import server as app  # Rename to "app" for Uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)

