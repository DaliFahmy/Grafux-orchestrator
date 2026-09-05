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
import json
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


# ── Design generation: language dispatch, interface extraction, validation ────
#
# The testbench helpers above answer "what are this design's ports?". The
# helpers below answer "is this design worth showing the user at all?" — the
# question the code_hdl block asks about source it has just generated, before a
# simulator ever sees it.

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\bTODO\b|\bFIXME\b|\.\.\.\s*unchanged\s*\.\.\.", re.IGNORECASE)

_VHDL_ENTITY_RE = re.compile(r"\bentity\s+([A-Za-z_][A-Za-z0-9_]*)\s+is\b", re.IGNORECASE)
_VHDL_ARCH_RE = re.compile(
    r"\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s+of\s+([A-Za-z_][A-Za-z0-9_]*)\s+is\b",
    re.IGNORECASE,
)
_VHDL_END_RE = re.compile(r"\bend\b", re.IGNORECASE)
_VHDL_COMMENT_RE = re.compile(r"--[^\n]*")

# Block-delimiting keyword pairs. Truncation part-way through a design is the
# dominant failure of an LLM asked for a whole file, and it is invisible to a
# bare "endmodule is present" check: a design that stops inside a case statement
# still ends with the word `endmodule` if the model closed it hopefully.
_VERILOG_PAIRS = (
    (r"\bmodule\b", r"\bendmodule\b", "module", "endmodule"),
    (r"\bbegin\b", r"\bend\b", "begin", "end"),
    (r"\bcase[xz]?\b", r"\bendcase\b", "case", "endcase"),
    (r"\bfunction\b", r"\bendfunction\b", "function", "endfunction"),
    (r"\btask\b", r"\bendtask\b", "task", "endtask"),
    (r"\bgenerate\b", r"\bendgenerate\b", "generate", "endgenerate"),
)

# Simulation-only constructs. WARNINGS, never problems: SystemVerilog FPGA code
# legitimately initialises `logic` in an `initial` block, and rejecting a correct
# design is worse than annotating a questionable one.
_SIM_ONLY = (
    (re.compile(r"^\s*initial\b", re.MULTILINE), "an `initial` block"),
    (re.compile(r"#\s*\d"), "a `#` delay"),
    (
        re.compile(r"\$(?:display|finish|monitor|stop|dumpfile|dumpvars)\b"),
        "a simulation system task ($display/$finish/...)",
    ),
)

_VERILOG_ONLY_RE = re.compile(r"\bendmodule\b|\balways(?:_ff|_comb|_latch)?\b")


def hdl_family(language: str) -> str:
    """``"verilog"`` (which covers SystemVerilog), ``"vhdl"``, or ``""`` for anything else."""
    key = (language or "").strip().lower().replace(" ", "")
    if key in ("verilog", "v", "sv", "systemverilog"):
        return "verilog"
    if key == "vhdl":
        return "vhdl"
    return ""


def _strip_comments_vhdl(text: str) -> str:
    return _VHDL_COMMENT_RE.sub("", text or "")


def vhdl_entity(code: str, fallback: str = _DEFAULT_TOP) -> str:
    """The FIRST entity declared in ``code`` (VHDL puts the interface first)."""
    m = _VHDL_ENTITY_RE.search(_strip_comments_vhdl(code))
    return m.group(1) if m else fallback


def design_interface(code: str, language: str, top: str = "") -> tuple[str, list[str]]:
    """
    The ``(top, ports)`` a generated design actually has.

    The single entry point the router uses, so it never branches on language.
    VHDL yields an empty port list on purpose — no VHDL port parser exists here,
    and writing one would serve a path that dead-ends at the simulator, since
    Verilator cannot run VHDL. As everywhere else in this module, an empty list
    means "could not tell", never "the design has no ports".
    """
    fallback = (top or "").strip() or _DEFAULT_TOP
    if hdl_family(language) == "vhdl":
        return (vhdl_entity(code, fallback), [])
    name = infer_top_module(code, fallback)
    return (name, module_ports(code, name))


def _unbalanced(text: str) -> list[str]:
    problems: list[str] = []
    for open_re, close_re, open_kw, close_kw in _VERILOG_PAIRS:
        opens = len(re.findall(open_re, text))
        closes = len(re.findall(close_re, text))
        if opens != closes:
            problems.append(
                f"unbalanced `{open_kw}`/`{close_kw}` ({opens} vs {closes}); "
                "the design looks truncated"
            )
    if text.count("(") != text.count(")"):
        problems.append(
            f"unbalanced parentheses ({text.count('(')} opening, {text.count(')')} closing)"
        )
    return problems


def _shared_problems(code: str) -> list[str]:
    problems: list[str] = []
    if _FENCE_RE.search(code):
        problems.append("the source still contains markdown code fences")
    if _PLACEHOLDER_RE.search(code):
        problems.append("the source contains a TODO/FIXME or an `... unchanged ...` marker")
    return problems


def _validate_vhdl(src: str) -> list[str]:
    bare = _strip_comments_vhdl(src)
    problems: list[str] = []
    entity = _VHDL_ENTITY_RE.search(bare)
    if not entity:
        problems.append("no `entity <name> is` declaration found")
    arch = _VHDL_ARCH_RE.search(bare)
    if not arch:
        problems.append("no `architecture <name> of <entity> is` declaration found")
    elif entity and arch.group(2).lower() != entity.group(1).lower():
        problems.append(
            f"the architecture is declared `of {arch.group(2)}` but the entity is "
            f"`{entity.group(1)}`"
        )
    if len(_VHDL_END_RE.findall(bare)) < 2:
        problems.append("fewer than two `end` keywords; the design looks truncated")
    if _VERILOG_ONLY_RE.search(bare):
        problems.append("the source contains Verilog constructs but VHDL was requested")
    return problems


def validate_hdl_design(code: str, language: str, top: str = "") -> tuple[list[str], list[str]]:
    """
    ``(problems, warnings)`` for a freshly generated design.

    The create-time sibling of :func:`validate_rtl_fix`, which cannot be reused
    because it judges a design against the one it replaces and there is no
    previous design on a first generation. The split matters: **problems** are
    worth spending a repair round on, **warnings** only annotate the block's
    ``improvements`` port and mark it ``needs_review``. Failing a design over an
    ``initial`` block would reject correct SystemVerilog; leaving a truncated
    module unremarked would hand the user something that cannot elaborate.
    """
    src = (code or "").strip()
    if not src:
        return (["the generator returned no code"], [])

    problems = _shared_problems(src)
    warnings: list[str] = []

    if hdl_family(language) == "vhdl":
        return (problems + _validate_vhdl(src), warnings)

    bare = _STRING_RE.sub('""', _strip_comments(src))
    if not re.search(r"\bmodule\b", bare):
        problems.append("no `module` declaration found")
    problems.extend(_unbalanced(bare))

    name = infer_top_module(src, "")
    wanted = (top or "").strip()
    if wanted and name and name != wanted:
        problems.append(
            f"the top module is named '{name}' but '{wanted}' was requested; the testbench "
            "and every downstream block address it by name"
        )
    resolved = name or wanted
    if resolved and not module_ports(src, resolved):
        problems.append(
            f"the port list of '{resolved}' could not be parsed; a testbench cannot be "
            "written against an interface that does not read"
        )
    if has_cocotb_test(src) or (name and name.endswith("_tb")):
        problems.append("this looks like a testbench, not a design")

    for pattern, label in _SIM_ONLY:
        if pattern.search(bare):
            warnings.append(
                f"the design contains {label}; that is simulation-only and will not synthesize"
            )
    return (problems, warnings)


# ── Specification generation: validating a contract, not source ───────────────
#
# The helpers above judge HDL. A spec_hdl block emits no HDL at all — it emits
# the CONTRACT that the code_hdl and testbench blocks are both handed — so the
# question here is different: is this document usable by two independent readers
# who will never see each other's output? That reduces to three checks a machine
# can actually make. The requirements have to be individually addressable (the
# testbench writes one test per REQ and cites it back), the proposed interface
# has to parse (a downstream block reads it as JSON), and the signals discussed
# have to be signals that exist. Everything else about a spec is a judgement call
# and belongs to the reviewer, not to a regex.

_REQ_LINE_RE = re.compile(r"^\s*REQ-(\d+)\s*:", re.MULTILINE)

# A clause that forbids something, which is where designs and tests diverge: an
# implementation that does the right thing on the happy path and something
# convenient on the path nobody wrote down.
_MUST_NOT_RE = re.compile(r"must\s+not|shall\s+not|may\s+not|never\b", re.IGNORECASE)

_DIRECTIONS = frozenset({"input", "output", "inout"})


def parse_interface(interface: str) -> list[dict]:
    """The ``interface`` JSON as a list of port dicts; ``[]`` when it does not read.

    Deliberately forgiving about the surrounding shape (a bare array, or an object
    with a "ports"/"signals" key — models produce both) and strict about nothing:
    the caller decides whether an empty result is a problem, because on the
    no-key fallback path it simply means "nothing was generated".
    """
    text = (interface or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        for key in ("ports", "signals", "interface"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def interface_signals(interface: str) -> list[str]:
    """The ``name`` of every entry in the ``interface`` JSON, in order."""
    names = []
    for item in parse_interface(interface):
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def requirement_ids(requirements: str) -> list[str]:
    """The ``REQ-<n>`` identifiers found in an enumerated requirements list."""
    return [f"REQ-{n}" for n in _REQ_LINE_RE.findall(requirements or "")]


def validate_spec(
    spec: str,
    requirements: str,
    interface: str,
    signals_analysis: str = "",
    design: str = "",
    top: str = "",
) -> tuple[list[str], list[str]]:
    """
    ``(problems, warnings)`` for a freshly generated specification.

    Same split as :func:`validate_hdl_design`: **problems** are worth a repair
    round, **warnings** only annotate ``improvements`` and mark the block
    ``needs_review``. The line between them is whether a downstream block would
    actually break. Unparsable ``interface`` JSON breaks one — something reads it
    — so it is a problem; a spec with no MUST-NOT clause is merely a weak spec,
    and failing it would reject a perfectly usable contract for a stylistic
    shortfall.

    ``design`` is an existing implementation, when there is one. A spec whose
    interface contradicts the RTL that exists is reported as a warning and never
    as a problem: the disagreement may well be the RTL's fault, and this block's
    whole purpose is to state the intended behaviour rather than ratify what was
    built.
    """
    problems: list[str] = []
    warnings: list[str] = []

    if not (spec or "").strip():
        return (["the generator returned no specification"], [])

    for text, label in ((spec, "specification"), (requirements, "requirements")):
        if _FENCE_RE.search(text or ""):
            problems.append(f"the {label} is wrapped in markdown fences")
        if _PLACEHOLDER_RE.search(text or ""):
            problems.append(
                f"the {label} contains a placeholder (TODO/FIXME); an unanswered "
                "question belongs on the improvements port, not in the contract"
            )

    reqs = requirement_ids(requirements)
    if not reqs:
        problems.append(
            "no enumerated 'REQ-<n>:' requirements; the testbench block writes one "
            "test per requirement and cites it back, so an unnumbered spec is not traceable"
        )
    elif len(set(reqs)) != len(reqs):
        seen, dupes = set(), []
        for r in reqs:
            if r in seen and r not in dupes:
                dupes.append(r)
            seen.add(r)
        problems.append(f"duplicate requirement numbers: {', '.join(dupes)}")

    if reqs and not _MUST_NOT_RE.search(requirements or ""):
        warnings.append(
            "no requirement forbids anything; a contract with only positive clauses "
            "is satisfied by an implementation that does the right thing on the happy "
            "path and anything at all elsewhere"
        )

    signals = interface_signals(interface)
    if (interface or "").strip() and not signals:
        problems.append(
            "the interface is not a JSON array of {name, direction, width, description} "
            "objects; downstream blocks read it as JSON"
        )
    if not (interface or "").strip():
        warnings.append("the specification proposes no interface")

    for item in parse_interface(interface):
        direction = str(item.get("direction", "")).strip().lower()
        name = str(item.get("name", "")).strip() or "<unnamed>"
        if direction not in _DIRECTIONS:
            shown = direction or "(missing)"
            problems.append(
                f"signal '{name}' has direction '{shown}'; "
                f"expected one of {', '.join(sorted(_DIRECTIONS))}"
            )

    if signals and (signals_analysis or "").strip():
        analysed = set(_IDENT_RE.findall(signals_analysis))
        missing = [s for s in signals if s not in analysed]
        if missing:
            warnings.append(
                "the signal analysis does not mention " + ", ".join(missing)
            )

    # A spec written next to an existing design: report drift, never condemn it.
    if (design or "").strip() and signals:
        actual = module_ports(design, (top or "").strip() or infer_top_module(design))
        if actual:
            extra = [s for s in signals if s not in actual]
            absent = [s for s in actual if s not in signals]
            detail = []
            if extra:
                detail.append("specified but not implemented: " + ", ".join(extra))
            if absent:
                detail.append("implemented but not specified: " + ", ".join(absent))
            if detail:
                warnings.append(
                    "the specified interface differs from the existing design ("
                    + "; ".join(detail)
                    + "); decide which one is wrong before regenerating either"
                )
    return (problems, warnings)


# ── The whole contract as one document ───────────────────────────────────────
#
# A spec_hdl block scatters its answer over a dozen ports because each of them
# is separately WIRABLE: the design block wants `spec`, the testbench wants
# `requirements`, a reviewer wants `assumptions`. Nothing was left for the
# reader who wants all of it — to paste into a review, to hand to a block that
# takes one blob of prose, or simply to read on the canvas without opening
# seven files.
#
# `full_spec` is that reader's port, and it is composed HERE rather than asked
# of the model, for the reason `accumulated_data` is assembled client-side on
# the memory block: a value derived from values already in hand cannot be lost
# by an LLM failure, cannot contradict the ports it summarises, and costs
# nothing. It is a VIEW, never a second source of truth — if the two ever
# disagree the sibling ports win.
#
# `status` and `errors` are deliberately absent. "needs_review" and "AI not
# configured" are bookkeeping about the run; this document is the contract.
_FULL_SPEC_SECTIONS: tuple[tuple[str, str], ...] = (
    ("top", "Top Module"),
    ("explanation", "Overview"),
    ("spec", "Specification"),
    ("requirements", "Requirements"),
    ("interface", "Interface"),
    ("signals_analysis", "Signals Analysis"),
    ("parameters", "Parameters"),
    ("timing", "Timing"),
    ("assumptions", "Assumptions"),
    ("improvements", "Open Questions"),
)

# Sections whose body is machine syntax rather than prose, and the fence
# language to wrap them in so a reader (human or model) cannot mistake where
# the JSON stops.
_FULL_SPEC_FENCED = {"interface": "json"}


def compose_full_spec(outputs: dict[str, str]) -> str:
    """Every substantive spec_hdl output as one markdown document.

    ``outputs`` is the port-name -> content mapping the block is about to
    return; keys it does not carry are simply absent from the document. An
    empty port contributes NOTHING - not even its heading - because a page of
    bare headings reads like a generation failure rather than a spec that had
    no assumptions to record.

    Returns ``""`` when the only thing on hand is the module name, so the
    no-key fallback writes an empty port rather than a title page: ``top``
    alone is a block that has not been run, not a specification.
    """
    sections: list[str] = []
    substantive = False
    for port, heading in _FULL_SPEC_SECTIONS:
        body = str(outputs.get(port, "") or "").strip()
        if not body:
            continue
        if port != "top":
            substantive = True
        fence = _FULL_SPEC_FENCED.get(port)
        if fence:
            body = f"```{fence}\n{body}\n```"
        sections.append(f"## {heading}\n{body}")
    if not substantive:
        return ""
    top = str(outputs.get("top", "") or "").strip()
    title = f"# Full Specification: {top}" if top else "# Full Specification"
    return "\n\n".join([title, *sections]) + "\n"
