"""The canvas tool vocabulary must stay ONE vocabulary.

It has been described in three places -- the declarations in canvas_tools.py, the
``##ACTIONS##`` grammar in Msg_config, and the C++ ``BlockAgentController::applyAction``
-- and they had already drifted (a hand-copied READ_PORT_TOOL with a different
``required`` list; ``set_loop_time``/``rename_block`` implemented in C++ but
declared to no model). These tests make the declarations the source and every
other shape a derivation, so the next divergence fails here instead of at runtime.

Complements test_canvas_tools_forwarding.py, which guards the create_block
parameters specifically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from app.modules.session import canvas_tools
from app.modules.session.canvas_context import READ_PORT_TOOL

# Declared tools that are READS resolved on the server, not canvas mutations, so
# they deliberately have no action mapping.
_READ_ONLY = {"read_port_value"}

# Handled by the Qt client but declared to no model, each for a reason. This set
# is the decision, not an oversight -- the test exists to force one.
_LEGACY_CLIENT_ONLY = {
    # Renaming a block invalidates every name reference held by the other
    # agents, the journal and the saved canvas, so it stays a human action.
    "rename_block",
    # Loop count / wait time are run-scheduling settings the user owns.
    "set_loop_time",
}


def _names(declarations) -> set[str]:
    return {d["name"] for d in declarations}


# ── Declarations are well formed ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "decl",
    canvas_tools.CANVAS_FUNCTION_DECLARATIONS + canvas_tools.AGENT_COORDINATION_DECLARATIONS,
    ids=lambda d: d["name"],
)
def test_declaration_is_well_formed(decl):
    assert decl["name"] and decl["description"], decl["name"]
    schema = decl["parameters"]
    assert schema["type"] == "object"
    for prop_name, prop in schema["properties"].items():
        assert prop.get("description"), f"{decl['name']}.{prop_name} has no description"
    for required in schema.get("required", []):
        assert required in schema["properties"], (
            f"{decl['name']} requires {required!r}, which it does not declare"
        )


def test_tool_names_are_unique_across_both_lists():
    canvas = _names(canvas_tools.CANVAS_FUNCTION_DECLARATIONS)
    agent = _names(canvas_tools.AGENT_COORDINATION_DECLARATIONS)
    assert not (canvas & agent), f"declared twice: {sorted(canvas & agent)}"


# ── Every mutation tool reaches the canvas ───────────────────────────────────


@pytest.mark.parametrize(
    "name",
    sorted(_names(canvas_tools.CANVAS_FUNCTION_DECLARATIONS) - _READ_ONLY),
)
def test_every_declared_canvas_tool_maps_to_an_action(name):
    """A declared tool with no serializer branch is a tool the model can call into a void."""
    decl = next(d for d in canvas_tools.CANVAS_FUNCTION_DECLARATIONS if d["name"] == name)
    args = {}
    for prop, spec in decl["parameters"]["properties"].items():
        args[prop] = ["x"] if spec.get("type") == "array" else f"v_{prop}"

    action = canvas_tools.function_call_to_action(name, args)

    assert action is not None, f"{name} is declared but function_call_to_action ignores it"
    assert action["type"] == name


def test_read_port_value_is_not_a_canvas_action():
    assert canvas_tools.function_call_to_action("read_port_value", {}) is None


# ── Provider shapes are derived, not restated ────────────────────────────────


def test_openai_and_anthropic_converters_cover_every_declaration():
    everything = _names(
        canvas_tools.CANVAS_FUNCTION_DECLARATIONS
        + canvas_tools.AGENT_COORDINATION_DECLARATIONS
    )
    assert {t["function"]["name"] for t in canvas_tools.to_openai_tools()} == everything
    assert {t["name"] for t in canvas_tools.to_anthropic_tools()} == everything


def test_selection_by_name_is_order_stable():
    """Tool order must not depend on the caller's argument order.

    Caching renders `tools` before `system`; a tool array that reshuffles between
    steps silently invalidates the whole prefix on every step.
    """
    forward = [t["name"] for t in canvas_tools.to_anthropic_tools(["run_block", "add_port"])]
    backward = [t["name"] for t in canvas_tools.to_anthropic_tools(["add_port", "run_block"])]
    assert forward == backward


def test_selection_ignores_unknown_names():
    assert canvas_tools.to_openai_tools(["not_a_tool"]) == []


def test_read_port_tool_is_derived_from_the_declaration():
    """The chat path's tool and the agents' tool are now literally the same object."""
    declared = next(
        d for d in canvas_tools.CANVAS_FUNCTION_DECLARATIONS if d["name"] == "read_port_value"
    )
    assert READ_PORT_TOOL == canvas_tools.to_openai_tools(["read_port_value"])[0]
    assert READ_PORT_TOOL["function"]["parameters"] == declared["parameters"]


# ── The C++ consumer ─────────────────────────────────────────────────────────


def _controller_source() -> str | None:
    """BlockAgentController.cpp, if the Qt app is checked out alongside us."""
    path = (
        Path(__file__).resolve().parents[2]
        / "Grafux-app/src/ui/diagram/blocks/blockagentcontroller.cpp"
    )
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def test_every_action_the_client_handles_is_declared_or_listed_as_legacy():
    source = _controller_source()
    if source is None:
        pytest.skip("Grafux-app is not checked out alongside the orchestrator")

    # `type == QStringLiteral("run_block")` -- the if-chain in applyAction.
    handled = set(re.findall(r'type == QStringLiteral\("(\w+)"\)', source))
    assert handled, "could not find the action if-chain; has applyAction been restructured?"

    undeclared = handled - _names(canvas_tools.CANVAS_FUNCTION_DECLARATIONS) - _LEGACY_CLIENT_ONLY
    assert not undeclared, (
        f"the Qt client handles {sorted(undeclared)} but no model is told they exist. "
        "Declare them in CANVAS_FUNCTION_DECLARATIONS, or add them to "
        "_LEGACY_CLIENT_ONLY with the reason they stay client-only."
    )


def test_every_declared_mutation_is_handled_by_the_client():
    source = _controller_source()
    if source is None:
        pytest.skip("Grafux-app is not checked out alongside the orchestrator")

    handled = set(re.findall(r'type == QStringLiteral\("(\w+)"\)', source))
    # load_block is applied by ChatPanelWidget rather than the controller.
    declared = _names(canvas_tools.CANVAS_FUNCTION_DECLARATIONS) - _READ_ONLY - {"load_block"}

    unhandled = declared - handled
    assert not unhandled, (
        f"declared to the model but the Qt client cannot apply {sorted(unhandled)}"
    )
