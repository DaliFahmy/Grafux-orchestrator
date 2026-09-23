"""
Tests for the post-silicon pair: the `post_silicon_verification` generator and
the `cpu` runtime block.

Three things are worth pinning here, and they are the three that break silently:
the PORT CONTRACT (the Qt side asserts the same lists, so a drift here surfaces
as a block whose ports depend on how it was created), the REGISTRATION (a type
missing from _ENRICHERS is never enriched, with no error anywhere), and the
GENERATOR's refusals — an empty explanation, a language the cpu block cannot run,
and a repair round that came back empty must each fail in a way that leaves the
user's work intact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.core.constants import EXPENSIVE_BLOCK_TYPES, BlockType
from app.modules.blocks import router as blocks_router
from app.modules.session import enrichment
from app.prompts import get_system_prompt


def _fake_settings(openai="sk-test", anthropic=""):
    return SimpleNamespace(
        openai_api_key=openai, anthropic_api_key=anthropic, openai_model="gpt-test"
    )


def _ports(params, side):
    return {p["port_name"]: p for p in params[f"{side}_ports"]}


# A well-formed model answer: a case that prints PASS lines and returns 0.
_GOOD_CASE = (
    "#include <stdio.h>\n"
    "int main(void) {\n"
    '    printf("PASS: cache writeback\\n");\n'
    "    return 0;\n"
    "}\n"
)

_GOOD_RESULT = {
    "code": _GOOD_CASE,
    "case_explanation": "Writes a known pattern and reads it back.",
    "verifies": "1. The L1 writes back modified lines on eviction.",
    "improvements": "Does not cover multi-core coherency.",
    "next_case_suggestion": "Check the same path under concurrent readers.",
    "language": "c",
    "compliance": "",
}


@pytest.fixture
def fake_llm(monkeypatch):
    """Answer the generator's JSON call from a mutable dict, with no network."""
    state = {"result": dict(_GOOD_RESULT), "messages": []}

    async def fake_json(system, user, **kwargs):
        state["messages"].append(user)
        result = state["result"]
        return dict(result) if isinstance(result, dict) else result

    monkeypatch.setattr(blocks_router, "_call_openai_json", fake_json)
    monkeypatch.setattr(blocks_router, "get_settings", lambda: _fake_settings())
    return state


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_both_types_are_registered_block_types():
    assert BlockType.POST_SILICON_VERIFICATION.value == "post_silicon_verification"
    assert BlockType.CPU.value == "cpu"


def test_only_the_block_that_rents_a_pod_is_expensive():
    """
    The cpu block provisions a machine; the generator only calls a model. Metering
    the generator as a pod run would charge a block agent against a budget meant
    for machines it never rented.
    """
    assert "cpu" in EXPENSIVE_BLOCK_TYPES
    assert "post_silicon_verification" not in EXPENSIVE_BLOCK_TYPES


def test_both_types_are_enriched():
    """A type absent from _ENRICHERS is silently never enriched."""
    assert enrichment._ENRICHERS["cpu"] is enrichment._enrich_scaffold_block
    assert (enrichment._ENRICHERS["post_silicon_verification"]
            is enrichment._enrich_post_silicon_verification_block)


def test_only_the_generator_has_a_prompt_section():
    """
    The cpu block produces its content at Run, from a real machine. Giving it a
    [create_*] section would route it through the AI create path and hand the
    user a fabricated result in place of a measurement.
    """
    from app.core.constants import BLOCK_TYPE_SECTION

    assert (BLOCK_TYPE_SECTION["post_silicon_verification"]
            == "create_post_silicon_verification")
    assert "cpu" not in BLOCK_TYPE_SECTION


def test_the_generator_prompt_section_exists():
    prompt = get_system_prompt("create_post_silicon_verification")
    assert prompt and "post-silicon validation engineer" in prompt


def test_the_improve_run_prompt_covers_the_cpu_kind():
    """Without this clause the cpu block's `analysis` port gets a generic review."""
    prompt = get_system_prompt("improve_run")
    assert "- cpu - " in prompt


def test_the_chat_prompt_offers_both_types():
    prompt = get_system_prompt("chat_assistant")
    assert "post_silicon_verification, cpu" in prompt
    # The distinction that actually matters: not the pre-silicon testbench.
    assert "post_silicon_verification=verify" in prompt
    assert "cpu=run/benchmark" in prompt


# ---------------------------------------------------------------------------
# The port contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_silicon_scaffold_ports():
    result = await blocks_router.generate_scaffold_payload(
        block_type="post_silicon_verification",
        block_name="cache writeback",
        category="l1",
        description="check the L1 writes back on eviction",
    )
    params = result["tool_calls"][0]["params"]
    assert params["name"] == "cache_writeback"

    assert set(_ports(params, "input")) == {
        "block_description", "explanation", "language", "constraints",
        "coverage_goals", "previous_case", "feedback",
    }
    assert set(_ports(params, "output")) == {
        "code", "language", "case_explanation", "verifies", "improvements",
        "next_case_suggestion", "status", "errors",
    }


@pytest.mark.asyncio
async def test_cpu_scaffold_ports():
    result = await blocks_router.generate_scaffold_payload(
        block_type="cpu", block_name="run case", category="l1",
        description="run the writeback case",
    )
    params = result["tool_calls"][0]["params"]

    assert set(_ports(params, "input")) == {
        "block_description", "spec", "code", "language", "args", "build_flags",
        "defines", "include_dirs", "files", "repetitions", "warmup", "timeout",
        "instance_type", "image", "api_keys",
    }
    assert set(_ports(params, "output")) == {
        "status", "passed", "response", "results", "benchmark", "duration",
        "machine", "errors", "warnings", "log", "artifacts", "eda_id", "cost",
        "analysis",
    }


@pytest.mark.asyncio
async def test_the_two_blocks_wire_together():
    """
    The pair is only useful if the obvious wiring is the correct one. Every port
    the canvas chain depends on must exist on both ends.
    """
    gen = (await blocks_router.generate_scaffold_payload(
        block_type="post_silicon_verification", block_name="g",
        description="d"))["tool_calls"][0]["params"]
    cpu = (await blocks_router.generate_scaffold_payload(
        block_type="cpu", block_name="c", description="d"))["tool_calls"][0]["params"]

    gen_out, cpu_in = set(_ports(gen, "output")), set(_ports(cpu, "input"))
    cpu_out, gen_in = set(_ports(cpu, "output")), set(_ports(gen, "input"))

    assert {"code", "language"} <= gen_out & cpu_in       # the program and how to build it
    assert "verifies" in gen_out and "spec" in cpu_in     # the claim, as review evidence
    assert "analysis" in cpu_out and "feedback" in gen_in  # the return leg


@pytest.mark.asyncio
async def test_cpu_defaults_are_seeded():
    """
    An unwired count port must mean "use the default". These are the values the Qt
    dialog seeds too, so a block made either way behaves identically.
    """
    params = (await blocks_router.generate_scaffold_payload(
        block_type="cpu", block_name="c", description="d"))["tool_calls"][0]["params"]
    ins = _ports(params, "input")
    assert ins["language"]["port_content"] == "cpp"
    assert ins["repetitions"]["port_content"] == "5"
    assert ins["warmup"]["port_content"] == "1"
    assert ins["build_flags"]["port_content"] == "-O2"


@pytest.mark.asyncio
async def test_the_cpu_spec_port_is_wire_only():
    """
    `spec` carries what the CASE claims to verify. Seeding it would put a second
    copy of that claim on the canvas for the review to cite instead of the one the
    case was actually written from — the reason verilator's `spec` is wire-only too.
    """
    spec = blocks_router._SCAFFOLD_SPECS["cpu"]
    assert "spec" in spec.inputs
    assert "spec" not in spec.seed_map.values()
    assert "spec" not in spec.defaults

    params = (await blocks_router.generate_scaffold_payload(
        block_type="cpu", block_name="c",
        description="check the writeback path"))["tool_calls"][0]["params"]
    assert _ports(params, "input")["spec"]["port_content"] == ""


def test_the_in_and_out_explanations_have_different_names():
    """
    The input is what the user wants verified; the output is what the generated
    case actually does. Same word for two different things on one block face is a
    trap, so they are named apart rather than defended with a comment.
    """
    spec = blocks_router._SCAFFOLD_SPECS["post_silicon_verification"]
    assert "explanation" in spec.inputs and "explanation" not in spec.outputs
    assert "case_explanation" in spec.outputs and "case_explanation" not in spec.inputs


def test_bulky_cpu_evidence_is_droppable_in_a_review():
    """
    `response` is program stdout and can be enormous. Without it in the drop
    order, a chatty case evicts the parsed results and the benchmark instead.
    """
    assert "response" in blocks_router._IMPROVEMENTS_DROP_ORDER
    order = list(blocks_router._IMPROVEMENTS_DROP_ORDER)
    # Dropped before the parsed, compact evidence it is the raw form of.
    assert "results" not in order and "benchmark" not in order


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_fills_every_output_port(fake_llm):
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="writeback", category="l1",
        explanation="check the L1 writes back on eviction", language="c",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    # Compared stripped: _strip_code_fences trims surrounding whitespace, which is
    # what keeps a fenced answer and a bare one from differing by a newline.
    assert outs["code"]["port_content"].strip() == _GOOD_CASE.strip()
    assert outs["verifies"]["port_content"].startswith("1.")
    assert outs["next_case_suggestion"]["port_content"]
    assert outs["case_explanation"]["port_content"]
    assert outs["status"]["port_content"] == "ok"
    assert outs["errors"]["port_content"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("language,expected", [
    ("c", "c"), ("C", "c"), ("c++", "cpp"), ("cpp", "cpp"),
    ("py", "python"), ("python", "python"), ("c11", "c"),
])
async def test_supported_languages_are_normalised(fake_llm, language, expected):
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language=language,
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["language"]["port_content"] == expected


@pytest.mark.asyncio
async def test_the_requested_language_wins_over_the_models_echo(fake_llm):
    """
    The cpu block compiles whatever this port says. A hallucinated echo here
    becomes a build failure one block downstream, with nothing pointing at the
    cause — so the server surfaces what was ASKED for.
    """
    fake_llm["result"] = {**_GOOD_RESULT, "language": "rust"}
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="cpp",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["language"]["port_content"] == "cpp"


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["rust", "go", "assembly", "verilog"])
async def test_a_language_the_cpu_block_cannot_run_is_refused(fake_llm, language):
    """
    Raised rather than silently corrected: the enricher catches it and scaffolds,
    which leaves the user their own language on the port to fix. Quietly writing
    the case in another language would be a worse answer than none.
    """
    with pytest.raises(ValueError, match="cpu block can run"):
        await blocks_router.generate_post_silicon_verification_payload(
            block_name="b", explanation="check something", language=language,
        )


@pytest.mark.asyncio
async def test_an_empty_explanation_is_refused(fake_llm):
    with pytest.raises(ValueError, match="explanation"):
        await blocks_router.generate_post_silicon_verification_payload(
            block_name="b", explanation="", description="",
        )


@pytest.mark.asyncio
async def test_no_llm_key_returns_none_so_the_caller_can_scaffold(monkeypatch):
    monkeypatch.setattr(
        blocks_router, "get_settings", lambda: _fake_settings(openai="", anthropic=""))
    assert await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something") is None


@pytest.mark.asyncio
async def test_a_case_with_no_pass_or_fail_lines_is_flagged(fake_llm):
    """
    The cpu block's verdict is the exit code AND the printed checks. A silent case
    can only ever be judged on whether it crashed, which wastes the block.
    """
    fake_llm["result"] = {
        **_GOOD_RESULT, "code": "int main(void) { return 0; }"}
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert "PASS" in outs["improvements"]["port_content"]


@pytest.mark.asyncio
async def test_a_c_case_without_main_is_flagged(fake_llm):
    fake_llm["result"] = {**_GOOD_RESULT, "code": 'void t(void){ puts("PASS: x"); }'}
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert "main()" in outs["improvements"]["port_content"]


@pytest.mark.asyncio
async def test_cpp_constructs_in_a_c_case_are_flagged(fake_llm):
    fake_llm["result"] = {
        **_GOOD_RESULT,
        "code": '#include <iostream>\nint main(){ std::cout << "PASS: x"; }',
    }
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert "set the language to cpp" in outs["improvements"]["port_content"]


@pytest.mark.asyncio
async def test_a_python_case_is_not_asked_for_main(fake_llm):
    fake_llm["result"] = {**_GOOD_RESULT, "code": 'print("PASS: x")', "language": "python"}
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="python",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "ok"


@pytest.mark.asyncio
async def test_an_unchanged_case_after_feedback_is_rejected(fake_llm):
    """
    Feedback that produced a byte-identical answer means the review was ignored.
    Reporting it as a clean regeneration would hide that.
    """
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
        previous_case=_GOOD_CASE, feedback="also check the unaligned case",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["status"]["port_content"] == "needs_review"
    assert "identical" in outs["improvements"]["port_content"]


@pytest.mark.asyncio
async def test_an_empty_repair_never_blanks_an_existing_case(fake_llm):
    """
    A generation that comes back empty must not destroy the one thing on the block
    that was working. The draft before it is worth more than nothing.
    """
    fake_llm["result"] = {**_GOOD_RESULT, "code": ""}
    result = await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
        previous_case=_GOOD_CASE, feedback="tighten the bounds check",
    )
    outs = _ports(result["tool_calls"][0]["params"], "output")
    assert outs["code"]["port_content"].strip() == _GOOD_CASE.strip()


@pytest.mark.asyncio
async def test_the_previous_case_reaches_the_prompt_on_a_revision(fake_llm):
    """
    Without it, "change what the feedback says and keep the rest" has nothing to
    keep, and a reviewed run rewrites every check — losing the ones nobody
    complained about. The trap testbench documents.
    """
    await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
        previous_case="int main(void){ /* MARKER */ return 0; }",
        feedback="also check the unaligned case",
    )
    assert "MARKER" in fake_llm["messages"][0]
    assert "also check the unaligned case" in fake_llm["messages"][0]


@pytest.mark.asyncio
async def test_constraints_and_coverage_goals_reach_the_prompt(fake_llm):
    await blocks_router.generate_post_silicon_verification_payload(
        block_name="b", explanation="check something", language="c",
        constraints="no dynamic allocation", coverage_goals="wrap-around at 2^32",
    )
    message = fake_llm["messages"][0]
    assert "no dynamic allocation" in message
    assert "wrap-around at 2^32" in message


# ---------------------------------------------------------------------------
# The fallback stub
# ---------------------------------------------------------------------------

def test_the_fallback_is_port_complete_and_keeps_the_existing_case():
    """
    A failed generation hands back a block the user can still work with — and one
    that has not lost the case it already had.
    """
    from app.modules.blocks.schemas import PostSiliconVerificationGenerateRequest

    body = PostSiliconVerificationGenerateRequest(
        block_name="writeback", category="l1", explanation="check it",
        language="c", previous_case=_GOOD_CASE,
    )
    result = blocks_router._simple_post_silicon_verification_response(
        body, error="AI not configured")
    params = result["tool_calls"][0]["params"]
    outs = _ports(params, "output")

    assert set(outs) == set(
        blocks_router._SCAFFOLD_SPECS["post_silicon_verification"].outputs)
    assert outs["code"]["port_content"] == _GOOD_CASE
    assert outs["status"]["port_content"] == "error"
    assert outs["errors"]["port_content"] == "AI not configured"
