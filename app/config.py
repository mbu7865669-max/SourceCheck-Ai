"""
Application configuration management for SourceCheck AI.

Loads and validates settings from environment variables with safe defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()


class Settings:
    """Application settings with environment variable overrides."""

    def __init__(self) -> None:
        self.APP_ENV: str = os.getenv("APP_ENV", "development")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))

        # Retrieval settings
        self.SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")
        self.MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        self.MAX_PAGES_TO_FETCH: int = int(os.getenv("MAX_PAGES_TO_FETCH", "3"))
        self.HTTP_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10.0"))
        self.HTTP_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_CONNECT_TIMEOUT_SECONDS", "5.0"))
        self.HTTP_READ_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "10.0"))
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "1"))
        self.MAX_REDIRECTS: int = int(os.getenv("MAX_REDIRECTS", "5"))
        self.MAX_PAGE_BYTES: int = int(os.getenv("MAX_PAGE_BYTES", "2000000"))  # 2 MB limit
        self.MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))  # 50k chars limit
        self.USER_AGENT: str = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 SourceCheckAI/1.0",
        )
        self.BLOCK_SSRF: bool = os.getenv("BLOCK_SSRF", "true").lower() in ("1", "true", "yes")

        # Evidence Extraction & Ranking settings
        self.MAX_PASSAGES_PER_PAGE: int = int(os.getenv("MAX_PASSAGES_PER_PAGE", "5"))
        self.MAX_TOTAL_PASSAGES: int = int(os.getenv("MAX_TOTAL_PASSAGES", "10"))
        self.MIN_PASSAGE_CHARS: int = int(os.getenv("MIN_PASSAGE_CHARS", "40"))
        self.MAX_PASSAGE_CHARS: int = int(os.getenv("MAX_PASSAGE_CHARS", "1000"))

        # Model Settings (Configurable: mock, local_rocm, local_server, opencode)
        self.MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "mock")
        self.MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        self.MODEL_ENDPOINT: str = os.getenv("MODEL_ENDPOINT", "")
        self.MODEL_TIMEOUT_SECONDS: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "15.0"))
        self.MAX_EVIDENCE_FOR_MODEL: int = int(os.getenv("MAX_EVIDENCE_FOR_MODEL", "5"))
        self.DEVICE: str = os.getenv("DEVICE", "cpu")
        self.MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

        # Agent service (the retrieval -> verification -> reasoning pipeline runs in
        # its own container/process; the web gateway proxies POST /answer to it).
        self.AGENT_ENDPOINT: str = os.getenv("AGENT_ENDPOINT", "http://127.0.0.1:8001")
        self.AGENT_PORT: int = int(os.getenv("AGENT_PORT", "8001"))
        # Generous: a full pipeline run can outlast a single bridge call
        # (retrieval + evidence + a 120s model budget + retries).
        self.AGENT_TIMEOUT_SECONDS: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "240.0"))

        # OpenCode Bridge Settings (OpenAI-compatible endpoint backed by the opencode CLI)
        # Served by bridge/opencode_bridge.py - see bridge/README.md.
        self.BRIDGE_ENDPOINT: str = os.getenv("BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1")
        self.BRIDGE_MODEL: str = os.getenv("BRIDGE_MODEL", "opencode/mimo-v2.6-flash-free")
        self.BRIDGE_API_KEY: str = os.getenv("BRIDGE_API_KEY", "")
        # Agentic free-model runs are much slower than a local 1.5B draft, so the bridge
        # gets its own budget instead of reusing MODEL_TIMEOUT_SECONDS (15s by default).
        self.BRIDGE_TIMEOUT_SECONDS: float = float(os.getenv("BRIDGE_TIMEOUT_SECONDS", "120.0"))
        # Ride out transient free-tier throttling instead of failing the whole request.
        # Retries apply only to 429/5xx, use exponential backoff, and are skipped
        # immediately when the bridge reports a long-horizon quota lockout.
        self.BRIDGE_MAX_RETRIES: int = int(os.getenv("BRIDGE_MAX_RETRIES", "3"))
        self.BRIDGE_RETRY_BACKOFF_SECONDS: float = float(
            os.getenv("BRIDGE_RETRY_BACKOFF_SECONDS", "2.0")
        )
        # A quota lockout worth waiting through, in seconds. Anything reported as
        # taking longer than this fails fast with the bridge's own retry hint rather
        # than burning the retry budget on sleeps that cannot possibly help.
        self.BRIDGE_MAX_RETRY_WAIT_SECONDS: float = float(
            os.getenv("BRIDGE_MAX_RETRY_WAIT_SECONDS", "300")
        )


# Global singleton instance for easy import across modules
settings = Settings()


def get_settings() -> Settings:
    """Return the global application settings instance."""
    return settings
