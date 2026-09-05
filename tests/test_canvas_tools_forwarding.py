from __future__ import annotations

import pytest
from app.modules.session import canvas_tools
from app.modules.session.enrichment import _enrich_memory_block


def _create_block_declaration() -> dict:
    for decl in canvas_tools.CANVAS_FUNCTION_DECLARATIONS:
        if decl["name"] == "create_block":
            return decl
    raise AssertionError("create_block declaration is gone")


# ── every declared param must actually reach the action ──────────────────────
#
# The failure this guards is silent: function_call_to_action drops a key the model
# was told to send, the enricher normalises the missing value to a default, and the
# block lands as the wrong subtype with nothing logged anywhere. That is exactly
# how every voice-created memory block was a snapshot one for as long as
# memory_mode existed.

# Params deliberately not forwarded, with the reason. Adding a name here is a
# decision; the test exists to make it one.
_INTENTIONALLY_NOT_FORWARDED: set[str] = set()  # a real exemption goes here, with its reason


def test_every_declared_create_block_param_is_forwarded():
    declared = set(_create_block_declaration()["parameters"]["properties"])
    args = {key: f"value_for_{key}" for key in declared}
    # The list/array params are typed, so give them plausible values.
    args["inputs"] = ["a"]
    args["outputs"] = ["b"]
    args["connections"] = ["gmail"]

    action = canvas_tools.function_call_to_action("create_block", args)

    missing = declared - set(action) - _INTENTIONALLY_NOT_FORWARDED
    assert not missing, (
        f"declared to the model but never put on the action: {sorted(missing)}. "
        "Add them to the forwarding tuple in function_call_to_action, or to "
        "_INTENTIONALLY_NOT_FORWARDED with a reason."
    )


@pytest.mark.parametrize("mode", ["snapshot", "sequential", "accumulate", "merge"])
def test_memory_mode_survives_the_voice_path(mode):
    action = canvas_tools.function_call_to_action(
        "create_block",
        {"block_type": "memory", "block_name": "notes", "description": "x",
         "memory_mode": mode},
    )
    assert action["memory_mode"] == mode


def test_filter_type_survives_the_voice_path():
    action = canvas_tools.function_call_to_action(
        "create_block",
        {"block_type": "filter", "block_name": "f", "description": "x",
         "filter_type": "image"},
    )
    assert action["filter_type"] == "image"


# ── the enricher: mode -> ports ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enrich_memory_block_merge_ports():
    action = {"block_name": "team_answers", "description": "what everyone said",
              "memory_mode": "merge"}
    await _enrich_memory_block(action, "test-session")

    assert action["memory_mode"] == "merge"
    assert {p["port_name"] for p in action["input_ports"]} == {
        "block_description", "input_1", "input_2",
    }
    assert {p["port_name"] for p in action["output_ports"]} == {
        "merged_data", "analysis",
    }


@pytest.mark.asyncio
async def test_enrich_memory_block_normalises_an_unknown_mode():
    action = {"block_name": "notes", "description": "x", "memory_mode": "nonsense"}
    await _enrich_memory_block(action, "test-session")

    # Normalised value is written BACK onto the action, so the client persists the
    # mode whose ports it actually received.
    assert action["memory_mode"] == "snapshot"
    assert {p["port_name"] for p in action["input_ports"]} == {
        "block_description", "input",
    }
