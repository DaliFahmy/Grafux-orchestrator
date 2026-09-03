from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TopicGenerateRequest(BaseModel):
    topic_name: str
    category: str = "general"
    inputs: list[str] = []
    outputs: list[str] = []
    description: str = ""
    # None = auto-decide (ground only when no rich content was provided);
    # True/False = force live-web grounding on/off.
    ground: bool | None = None
    # Selected AI model (e.g. "claude-opus-4-8", "gpt-5"); None → backend default.
    run_llm_model: str | None = None


class CodeGenerateRequest(BaseModel):
    """Inputs for the code block's AI generation (create, Run and Regenerate).

    ``feedback`` plus ``previous_code`` switch the request from "write this" to
    "repair this": for an HDL language they select the [fix_rtl] prompt, whose
    contract is a minimal change to the given design that makes the reported
    failing tests pass, with the module interface left byte-identical. Both are
    required for that — feedback with no previous design has nothing to repair,
    and a previous design with no feedback has nothing to repair it against.
    """

    block_name: str
    category: str = "general"
    description: str = ""
    language: str = "python"
    # A failing-test report, normally the verilator block's ``failures`` output.
    feedback: str = ""
    # The design that produced it; its interface is frozen during the repair.
    previous_code: str = ""
    inputs: list[str] = []
    outputs: list[str] = []


class TestbenchGenerateRequest(BaseModel):
    """Inputs for the testbench block's AI generation (create, Run and Regenerate).

    ``spec`` is the behavioural specification the tests are derived from; ``rtl`` is
    only used to extract the module interface (port names) so the generated testbench
    references real signals. ``feedback`` carries a reviewer's notes on a previous
    testbench (e.g. "test_full_flag expects the wrong latency") for a repair round.
    """

    block_name: str
    category: str = "general"
    spec: str = ""
    rtl: str = ""
    top: str = ""
    framework: str = "cocotb"
    style: str = "directed+random"
    coverage_goals: str = ""
    extra_tests: str = ""
    feedback: str = ""
    inputs: list[str] = []
    outputs: list[str] = []


class ImageGenerateRequest(BaseModel):
    block_name: str
    category: str = "general"
    description: str = ""
    inputs: list[str] = []
    outputs: list[str] = []


class RunSearchRequest(BaseModel):
    block_name: str
    block_type: str  # "topics", "components", "procedures"
    context_message: str
    existing_output_ports: list[str] = []
    recreate_ports: bool = False
    # None = auto-decide (ground via live web search when no reference material
    # was attached); True/False = force live-web grounding on/off.
    ground: bool | None = None
    run_llm_model: str | None = None


class RunSelectionRequest(BaseModel):
    block_name: str
    criteria: str
    candidates: list[dict[str, Any]] = []
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
