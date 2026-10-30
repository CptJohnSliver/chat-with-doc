import os
import sys
from pathlib import Path

import uvicorn

backend_root = Path(__file__).resolve().parent.parent.parent

if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.app.api.api:app",
            host="127.0.0.1",
            reload=False,
            proxy_headers=True,
            server_header=True,
            port=int(os.getenv("API_PORT", "3624")),
            access_log=False,
            log_level="warning",
        )
    except Exception:
        # Логгер поднимется в startup приложения; здесь просто пробрасываем дальше
        raise
