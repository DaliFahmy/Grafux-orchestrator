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


class CodeHdlGenerateRequest(BaseModel):
    """Inputs for the code_hdl block's AI generation (create, Run and Regenerate).

    The RTL source of the verification loop. ``spec`` is the behavioural contract
    the design must satisfy — the same text the testbench block derives its tests
    from, which is what makes the two blocks agree about what "correct" means;
    it falls back to ``description`` when the user did not separate the two.
    ``top`` names the module every downstream block addresses, and ``constraints``
    carries what the spec does not: reset style, interface conventions, target
    technology.

    ``feedback`` plus ``previous_code`` switch the request from "write this" to
    "repair this", selecting the shared [fix_rtl] prompt exactly as the code
    block does. Both are required: feedback with no design has nothing to repair,
    and a design with no feedback has nothing to repair it against.
    """

    block_name: str
    category: str = "general"
    description: str = ""
    spec: str = ""
    language: str = "systemverilog"
    top: str = ""
    constraints: str = ""
    # A failing-test report, normally the verilator block's ``failures`` output.
    feedback: str = ""
    # The design that produced it; its interface is frozen during the repair.
    previous_code: str = ""
    inputs: list[str] = []
    outputs: list[str] = []
    run_llm_model: str = ""


class SpecHdlGenerateRequest(BaseModel):
    """Inputs for the spec_hdl block's AI generation (create, Run and Regenerate).

    The CONTRACT block: it writes the specification that ``code_hdl`` implements
    and ``testbench`` derives its tests from. ``explanation`` is the rough human
    description it starts from (falling back to ``description``, because in
    practice nobody writes the behaviour twice), and the remaining fields are the
    parameters every hardware spec needs but prose keeps leaving implicit —
    widths, clocking, reset polarity, combinational vs sequential, the handshake
    protocol, the throughput contract.

    ``previous_code`` is an EXISTING design, not a previous spec: with it the
    block reconciles the contract against the RTL that actually exists (or
    reverse-engineers a spec from legacy RTL), and its real port list is used to
    check the proposed interface.

    ``feedback`` is a review of the loop — verilator's failing tests, or an
    ``improvements_spec`` bullet (or, more crudely,
    an ``improvements_rtl`` one). It selects the REVISION
    flow rather than the fresh one, because when a design and its tests disagree
    the fault is often the spec they were both derived from, and [fix_rtl] can
    only ever repair the design.
    """

    block_name: str
    category: str = "general"
    description: str = ""
    explanation: str = ""
    # The design under discussion, normally the code_hdl block's ``code`` output.
    previous_code: str = ""
    # The spec as it currently stands; sent on Regenerate so a revision edits the
    # contract instead of writing an unrelated one from the same explanation.
    previous_spec: str = ""
    top: str = ""
    language: str = "systemverilog"
    data_width: str = ""
    addr_width: str = ""
    parameters: str = ""
    logic_style: str = ""
    reset_style: str = ""
    clocking: str = ""
    protocol: str = ""
    throughput: str = ""
    constraints: str = ""
    # A failing-test report or a run review; selects the revision flow.
    feedback: str = ""
    inputs: list[str] = []
    outputs: list[str] = []
    run_llm_model: str = ""


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


class RunAccumulateRequest(BaseModel):
    """Inputs for reviewing ONE new value against a memory block's stored record.

    The accumulate memory block appends ``new_data`` to its ``accumulated_data``
    port itself, locally, before it ever calls here -- so this endpoint never
    produces the record, only the commentary on it.  ``previous_accumulated`` is
    therefore the record as it stood BEFORE the append: comparing the new value
    against a record that already contains it would report every value as a repeat
    of itself.

    That record grows without bound, so the app sends its TAIL and marks the cut
    with a "[earlier entries truncated]" line; the prompt is told to treat a marked
    view as partial rather than claim an absence it cannot see.
    """

    block_name: str
    description: str = ""
    previous_accumulated: str = ""
    new_data: str = ""
    run_llm_model: str | None = None


class MergeInputItem(BaseModel):
    """One input port's contribution to a merge memory block's run."""

    port: str
    text: str = ""


class RunMergeRequest(BaseModel):
    """Inputs for reviewing the several values a memory block merged in ONE run.

    The merge memory block writes the raw labelled combination to its
    ``merged_data`` port itself, locally, before it calls here -- so this endpoint
    never produces the merge, only the review of it.  That split is what keeps the
    data safe when the model call fails.

    Unlike accumulate there is no history to compare against: these inputs are
    PEERS that arrived together, and each run REPLACES both outputs.  The question
    is therefore not "is this new?" but how the inputs agree, conflict and
    complement each other.

    Each input is labelled with the port it came from, and the caller uses the same
    ``--- <port> ---`` markers it wrote into ``merged_data``, so the review's
    citations line up with what the user sees on the block.
    """

    block_name: str
    description: str = ""
    inputs: list[MergeInputItem] = []
    run_llm_model: str | None = None


class RunFilterRequest(BaseModel):
    block_name: str
    filter_type: str = "text"
    description: str = ""
    code: str = ""
    criteria: str = ""
    input_value: str = ""
    run_llm_model: str | None = None



class ImprovementsRequest(BaseModel):
    """Inputs for reviewing a run that has ALREADY finished.

    ``run`` is the evidence, as a port-name -> text map taken straight off the
    block's outputs.  It is deliberately an open dict rather than named fields:
    the devices server grows output ports over time, and a fixed schema here
    would silently drop the newest evidence.  The app caps every value before
    sending (``EdaImprovements::buildRequestBody`` in
    Grafux-app/src/clients/grafux-devices/edaimprovements.h); the server's own
    cap is a backstop against a client that does not.

    ``verdict`` is a named field rather than a ``run`` key because the prompt
    BRANCHES on it -- a passing verilator run gets the coverage review, a
    failing one gets root-cause analysis -- and the app knows it unambiguously
    from VerificationResults::parse().  Making the model re-derive it from a
    JSON blob is exactly how a pass gets reviewed as a failure.
    """

    block_name: str
    kind: str = "device"          # "verilator" | "openroad" | "device"
    block_description: str = ""
    verdict: str = ""             # "passed" | "failed" | "error" | ""
    run: dict[str, str] = {}      # port name -> evidence text
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
