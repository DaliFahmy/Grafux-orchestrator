"""
Pure helpers for reasoning about Verilog interfaces and cocotb testbenches.

The testbench block generates tests from the SPEC, but the tests still have to
drive real signals, so the generator needs the module's port list to (a) tell the
model which names exist and (b) catch a testbench that invents a signal. The
interface parser here is a port of ``EDA/flow.py`` in Grafux-devices (same regex
rules, same non-ANSI/ANSI handling) so both sides agree on what a design's ports
are; keeping it dependency-free means the orchestrator does not need a Verilog
toolchain on the box to validate what it just wrote.

Everything here is synchronous, side-effect free and unit-tested in isolation.
"""

from __future__ import annotations

import ast
import re

_DEFAULT_TOP = "top"

_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", re.MULTILINE)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")

# Words that can be the last identifier of a port item without being its name.
_PORT_NOISE = frozenset({
    "input", "output", "inout", "wire", "reg", "logic", "signed", "unsigned",
    "bit", "byte", "int", "integer", "real", "time", "tri", "wand", "wor",
    "supply0", "supply1", "parameter", "localparam", "var",
})

# ``dut.<name>`` references in a cocotb testbench. The trailing lookahead keeps
# ``dut.clk.value`` → ``clk`` and drops cocotb's own handle helpers.
_DUT_REF_RE = re.compile(r"\bdut\.([A-Za-z_][A-Za-z0-9_$]*)")
_HANDLE_HELPERS = frozenset({"_log", "_name", "_path", "_id", "_handle", "_sub_handles"})

_COCOTB_TEST_RE = re.compile(r"@cocotb\.test\s*\(")


def infer_top_module(rtl: str, fallback: str = _DEFAULT_TOP) -> str:
    """The LAST module declared in ``rtl`` (submodules conventionally come first)."""
    names = _MODULE_RE.findall(rtl or "")
    return names[-1] if names else fallback


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub(" ", text or "")


def _balanced(text: str, start: int) -> str:
    """Contents of the parenthesised group opening at ``text[start]``; "" if unclosed."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return ""


def _split_top_level(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [item for item in items if item.strip()]


def module_ports(rtl: str, top: str) -> list[str]:
    """
    Ports declared in module ``top``'s header, in declaration order.

    Handles ANSI (``input wire [7:0] a``) and non-ANSI (``module m(a, b);``)
    headers and an optional ``#(...)`` parameter list. An empty list means
    "could not tell", never "portless" — callers must not validate against it.
    """
    source = _strip_comments(rtl)
    name = (top or "").strip()
    if not name:
        return []
    match = re.search(r"\bmodule\s+" + re.escape(name) + r"\b", source)
    if not match:
        return []

    pos = match.end()

    def skip_space(index: int) -> int:
        while index < len(source) and source[index].isspace():
            index += 1
        return index

    pos = skip_space(pos)
    if pos < len(source) and source[pos] == "#":
        pos = skip_space(pos + 1)
        if pos >= len(source) or source[pos] != "(":
            return []
        params = _balanced(source, pos)
        if not params:
            return []
        pos = skip_space(pos + len(params) + 2)
    if pos >= len(source) or source[pos] != "(":
        return []

    ports: list[str] = []
    for item in _split_top_level(_balanced(source, pos)):
        names = [n for n in _IDENT_RE.findall(item) if n not in _PORT_NOISE]
        if names:
            ports.append(names[-1])
    return ports


def dut_signals(testbench: str) -> list[str]:
    """Distinct ``dut.<signal>`` names a cocotb testbench references, in first-use order."""
    seen: list[str] = []
    for name in _DUT_REF_RE.findall(testbench or ""):
        if name in _HANDLE_HELPERS or name in seen:
            continue
        seen.append(name)
    return seen


def hallucinated_signals(testbench: str, ports: list[str]) -> list[str]:
    """
    ``dut.<name>`` references that are not ports of the design.

    Returns ``[]`` when ``ports`` is empty: an unparsable interface must not
    condemn every testbench. Internal-signal peeking (``dut.count_reg``) is
    reported too — the prompt forbids it because tests derived from internals
    stop being spec tests.
    """
    if not ports:
        return []
    known = set(ports)
    return [name for name in dut_signals(testbench) if name not in known]


def has_cocotb_test(testbench: str) -> bool:
    """True when the source declares at least one ``@cocotb.test()`` coroutine."""
    return bool(_COCOTB_TEST_RE.search(testbench or ""))


def python_syntax_error(source: str) -> str:
    """The SyntaxError message for ``source``, or "" when it parses."""
    try:
        ast.parse(source or "")
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return ""


def validate_testbench(testbench: str, rtl: str, top: str) -> list[str]:
    """
    Human-readable problems with a generated cocotb testbench (empty = accepted).

    Used both to decide on a repair round and, when repair does not fully fix
    things, as the text surfaced on the block's ``improvements`` port — so the
    user sees why a testbench may not run rather than discovering it in the
    simulator log.
    """
    problems: list[str] = []
    if not (testbench or "").strip():
        return ["empty testbench"]
    err = python_syntax_error(testbench)
    if err:
        problems.append(f"Python syntax error: {err}")
    if not has_cocotb_test(testbench):
        problems.append("no @cocotb.test() coroutine found")
    ports = module_ports(rtl, top)
    bad = hallucinated_signals(testbench, ports)
    if bad:
        problems.append(
            "references signals that are not ports of "
            f"{top or infer_top_module(rtl)}: {', '.join(bad)} (ports: {', '.join(ports)})"
        )
    return problems


def validate_rtl_fix(new_code: str, previous_code: str, top: str) -> list[str]:
    """
    Human-readable problems with a repaired design (empty = accepted).

    The repair contract is narrow on purpose: the testbench was written against a
    specific interface and every downstream block is wired to it, so a fix that
    renames the module or changes its ports is not a fix — it silently breaks the
    tests it was supposed to satisfy and every connection on the canvas. These
    checks are what turn "keep the header identical" from an instruction in a
    prompt into something the server can actually enforce.
    """
    problems: list[str] = []
    new = (new_code or "").strip()
    if not new:
        return ["the fixer returned no code"]
    if "endmodule" not in new:
        problems.append("the returned design has no `endmodule`; it looks truncated")

    if new == (previous_code or "").strip():
        # Not pedantry: the client loop would otherwise re-provision a pod to
        # reproduce the identical failure until the iteration budget ran out.
        problems.append("the returned design is identical to the one that failed")

    top_name = (top or "").strip() or infer_top_module(previous_code)
    new_top = infer_top_module(new)
    if top_name and new_top != top_name:
        problems.append(
            f"the module was renamed from '{top_name}' to '{new_top}'; the testbench "
            "and every downstream block address it by name"
        )
        return problems

    # An unparsable interface on either side means "cannot tell", never "no ports"
    # — condemning a fix on the strength of a regex that failed would be worse
    # than accepting one that is fine.
    old_ports = module_ports(previous_code, top_name)
    new_ports = module_ports(new, top_name)
    if old_ports and new_ports and old_ports != new_ports:
        added = [p for p in new_ports if p not in old_ports]
        removed = [p for p in old_ports if p not in new_ports]
        detail = []
        if removed:
            detail.append("removed " + ", ".join(removed))
        if added:
            detail.append("added " + ", ".join(added))
        if not detail:
            detail.append("reordered the ports")
        problems.append(
            "the module interface changed (" + "; ".join(detail)
            + "); it must stay identical to " + ", ".join(old_ports)
        )
    return problems
