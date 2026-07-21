"""Wire the OpenAI client from .env.

Supports any OpenAI-compatible endpoint (Azure OpenAI ``.../openai/v1``,
proxies, gateways) via ``OPENAI_BASE_URL`` — in that case the model names in
config are your *deployment* names. Tracing export is disabled on custom
endpoints because the trace backend lives at api.openai.com.
"""

import os
from pathlib import Path

from .config import REPO_ROOT


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no extra dependency); never overrides real env vars."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configure_openai() -> None:
    """Call once before building a pipeline that talks to a real model."""
    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if base_url:
        from agents import set_default_openai_client, set_tracing_disabled
        from openai import AsyncOpenAI

        set_default_openai_client(AsyncOpenAI(base_url=base_url, api_key=api_key))
        set_tracing_disabled(True)
