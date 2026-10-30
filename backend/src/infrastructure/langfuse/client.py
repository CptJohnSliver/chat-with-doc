from langfuse import Langfuse

from src.common.config.config import settings

langfuse = Langfuse(
    public_key=settings.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=settings.get("LANGFUSE_SECRET_KEY"),
    base_url=settings.get("LANGFUSE_BASE_URL"),
    # release="v1.2.3",
    environment="dev",
    tracing_enabled=settings.get("LANGFUSE_ENABLED"),
)
