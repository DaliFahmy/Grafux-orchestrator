from __future__ import annotations

from typing import List

from pydantic import BaseModel


class TopicGenerateRequest(BaseModel):
    topic_name: str
    category: str = "general"
    inputs: List[str] = []
    outputs: List[str] = []
    description: str = ""
