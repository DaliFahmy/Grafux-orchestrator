from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    jwt_secret: str = field(
        default_factory=lambda: os.environ.get("JWT_SECRET", "")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("S3_BUCKET", "")
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o")
    )
    history_max_turns: int = 20
    gemini_live_url: str = (
        "wss://generativelanguage.googleapis.com"
        "/ws/google.ai.generativelanguage.v1beta"
        ".GenerativeService.BidiGenerateContent"
    )
