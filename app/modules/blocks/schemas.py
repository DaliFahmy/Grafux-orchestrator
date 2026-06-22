from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class TopicGenerateRequest(BaseModel):
    topic_name: str
    category: str = "general"
    inputs: List[str] = []
    outputs: List[str] = []
    description: str = ""
    # None = auto-decide (ground only when no rich content was provided);
    # True/False = force live-web grounding on/off.
    ground: bool | None = None
    # Selected AI model (e.g. "claude-opus-4-8", "gpt-5"); None → backend default.
    run_llm_model: str | None = None


class CodeGenerateRequest(BaseModel):
    block_name: str
    category: str = "general"
    description: str = ""
    language: str = "python"
    inputs: List[str] = []
    outputs: List[str] = []


class RunSearchRequest(BaseModel):
    block_name: str
    block_type: str  # "topics", "components", "procedures"
    context_message: str
    existing_output_ports: List[str] = []
    recreate_ports: bool = False
    # None = auto-decide (ground via live web search when no reference material
    # was attached); True/False = force live-web grounding on/off.
    ground: bool | None = None
    run_llm_model: str | None = None


class RunSelectionRequest(BaseModel):
    block_name: str
    criteria: str
    candidates: List[Dict[str, Any]] = []
    run_llm_model: str | None = None


class RunFilterRequest(BaseModel):
    block_name: str
    filter_type: str = "text"
    description: str = ""
    code: str = ""
    criteria: str = ""
    input_value: str = ""
    run_llm_model: str | None = None


class RegenerateToolRequest(BaseModel):
    block_name: str
    prompt: str
    regen_llm_model: str | None = None


class RegenerateFilterRequest(BaseModel):
    block_name: str
    filter_type: str = "text"
    prompt: str
    regen_llm_model: str | None = None
