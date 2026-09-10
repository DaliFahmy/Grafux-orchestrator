"""
Pure helpers that turn instruction text into checkable directives, and check them.

The three HDL block types (``code_hdl``, ``spec_hdl``, ``testbench``) all take
free-form instruction text on input ports — ``feedback`` above all, plus
``constraints``, ``extra_tests``, ``coverage_goals`` and spec_hdl's pinned design
parameters. Handing that text to a model is necessary but not sufficient: a
prompt is a request, and the run has to be able to say afterwards whether the
request was honoured. If ``feedback`` says ``make parameter FIFO_DEPTH = 32;``
then the emitted design either declares that parameter or it does not, and the
server can tell which.

So this module does two things:

* :func:`extract_directives` mines instruction text for the parts whose
  satisfaction is *decidable* — a parameter value, a literal snippet, a required
  or forbidden identifier, a port, a test name, a requirement id — and keeps
  every remaining line as an unverifiable ``prose`` directive.
* :func:`unmet` checks a generated artifact against those directives and returns
  the same ``(problems, warnings)`` split the validators in :mod:`hdl` use: a
  problem is worth a repair round, a warning only annotates ``improvements``.

**The honesty rule.** A ``prose`` directive is NEVER reported as unmet. "Make the
FIFO nicer" cannot be checked by a regex, and a checker that guessed would
reject good designs for failing a test it cannot actually run — which is worse
than not checking, because it burns the repair budget the decidable directives
need. Prose is enforced through the prompt and through the model's own
``compliance`` answer (:func:`compliance_gaps`), never here. This mirrors
``hdl.module_ports`` returning ``[]`` for "could not tell" rather than "no ports".

Everything here is synchronous, side-effect free and unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hdl import _strip_comments, _strip_comments_vhdl, module_ports

# ── What a directive is ───────────────────────────────────────────────────────

#: Directive kinds, in the order :func:`extract_directives` tries them. Every
#: kind except ``prose`` is decidable against a generated artifact.
KIND_PARAMETER = "parameter"
#: A pinned NUMBER whose name is a human label rather than an identifier
#: ("Data width: 16"). All that can be checked is that the number itself turns up
#: in the artifact - the design is free to call the parameter whatever it likes.
KIND_VALUE = "value"
KIND_LITERAL = "literal"
KIND_FORBIDDEN = "forbidden"
KIND_IDENTIFIER = "identifier"
KIND_SIGNAL = "signal"
KIND_TEST = "test"
KIND_REQUIREMENT = "requirement"
KIND_PROSE = "prose"

#: Kinds that change a module's PORT LIST, which is what ``hdl.validate_rtl_fix``
#: refuses to let a repair touch. The router uses this to keep an
#: interface-changing instruction off the [fix_rtl] prompt, where a correct
#: answer would be rejected by validation. A parameter or width change is
#: deliberately NOT here: validate_rtl_fix compares port names and order only,
#: so widening a port survives it untouched.
INTERFACE_KINDS = frozenset({KIND_SIGNAL})


@dataclass(frozen=True)
class Directive:
    """One instruction, and what would prove it was applied.

    ``text`` is the verbatim source line. It is what the model is shown and what
    the user is shown on ``improvements`` when the instruction goes unmet — a
    paraphrase would leave them hunting for a sentence they never wrote.
    """

    kind: str
    target: str
    value: str = ""
    text: str = ""
    source: str = ""
    #: ``False`` => a miss is a warning, not a problem. Used for directives the
    #: server INFERRED (a parameter table read out of spec prose) rather than
    #: ones the user stated: the inference may simply be wrong, and rejecting a
    #: design over it would be the checker's bug, not the model's.
    binding: bool = True
    #: Extra names a check may need (unused today; keeps the dataclass frozen
    #: while leaving room for a directive that carries context).
    extra: tuple[str, ...] = field(default=())

    def evidence(self) -> str:
        """The short "[must appear: ...]" hint shown next to the instruction."""
        if self.kind == KIND_PARAMETER:
            return f"parameter {self.target} = {self.value}"
        if self.kind == KIND_VALUE:
            return f"{self.target} of exactly {self.value}"
        if self.kind == KIND_LITERAL:
            return self.target
        if self.kind == KIND_FORBIDDEN:
            return f"must NOT contain {self.target}"
        if self.kind == KIND_TEST:
            return f"a test named {self.target}"
        if self.kind == KIND_SIGNAL:
            return f"a port named {self.target}"
        if self.kind == KIND_REQUIREMENT:
            return f"{self.target} still present"
        if self.kind == KIND_IDENTIFIER:
            return self.target
        return ""


# ── Extraction ────────────────────────────────────────────────────────────────

# A parameter NAME must be UPPER_SNAKE. Hardware convention, and load-bearing as
# a filter: matching any identifier would mine "make it 32 bits deep" for a
# directive about the word "it", and the checker would then reject every design.
_PARAM_NAME = r"[A-Z][A-Z0-9_]{2,}"

# A value we can actually look for: a decimal, or a sized/based Verilog literal.
# An expression ("$clog2(DEPTH)") is deliberately not decidable — the design may
# spell the same value a different legal way.
_PARAM_VALUE = r"(?:\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ]+|\d+)"

# `parameter FIFO_DEPTH = 32` / `localparam [3:0] WIDTH = 8`
_PARAM_DECL_RE = re.compile(
    r"\b(?:parameter|localparam)\s+(?:\w+\s+)?(?:\[[^\]]*\]\s*)?"
    rf"({_PARAM_NAME})\s*=\s*({_PARAM_VALUE})"
)

# Prose forms: "FIFO_DEPTH = 32", "FIFO_DEPTH: 32", "set FIFO_DEPTH to 32",
# "DATA_WIDTH must be 16", "ADDR_WIDTH should be 4".
_PARAM_PROSE_RE = re.compile(
    rf"\b({_PARAM_NAME})\s*(?:=|:|\bto\b|\bmust\s+be\b|\bshould\s+be\b|\bis\b)\s*"
    rf"({_PARAM_VALUE})\b"
)

# Backticked or fenced text. A snippet only counts as a literal to reproduce when
# it looks like code — otherwise `full` in "check the `full` flag" would become a
# whole-line substring requirement.
_BACKTICK_RE = re.compile(r"`([^`\n]{2,200})`")
_FENCE_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
_CODEISH_RE = re.compile(r"[=;(){}\[\]]|<=|=>")

# A bare identifier we can require or forbid: an HDL/Python token, long enough
# not to be an English word by accident.
_TOKEN_RE = re.compile(r"\b((?:[a-z][a-z0-9]*_[a-z0-9_]{1,30})|always_ff|always_comb|always_latch|endmodule|nonblocking)\b")

_FORBID_RE = re.compile(
    r"\b(?:do\s+not|don'?t|never|must\s+not|shall\s+not|may\s+not|avoid|no\s+more)\b",
    re.IGNORECASE,
)
_REQUIRE_RE = re.compile(
    r"\b(?:use|using|must\s+use|always\s+use|name\s+it|call\s+it|rename\s+to|switch\s+to)\b",
    re.IGNORECASE,
)
_ADD_SIGNAL_RE = re.compile(
    r"\b(?:add|expose|include|introduce|provide)\b[^.\n]*?\b(?:port|output|input|signal|pin)\b",
    re.IGNORECASE,
)
_SIGNAL_ALT_RE = re.compile(
    r"\b(?:port|output|input|signal)\s+(?:called|named)\s+`?([A-Za-z_][A-Za-z0-9_$]{1,40})`?",
    re.IGNORECASE,
)

_TEST_NAME_RE = re.compile(r"\b(test_[A-Za-z0-9_]{2,60})\b")
_REQ_ID_RE = re.compile(r"\b(REQ-\d+)\b")

# "[must appear: count == DEPTH]" - an instruction naming its OWN evidence.
# render_instructions() writes this syntax into every prompt, and the verilator
# block's fix_rtl / fix_tb / fix_spec ports write it back: the triage states the
# literal it is certain must be reproduced verbatim, and that literal is then
# checked here rather than guessed at.  That is what closes the loop - a repair
# order the server can actually hold the next artifact to.
_MUST_APPEAR_RE = re.compile(r"\[must appear:\s*([^\]\n]{1,400})\]", re.IGNORECASE)

#: Longest hint we will accept as decidable.  A whole pasted line is not
#: something substring presence can honestly judge, and a hint the checker
#: cannot decide rejects a CORRECT answer while burning the repair rounds the
#: real ones need.  Same honesty rule as `prose`: when in doubt, stay silent.
_MUST_APPEAR_MAX = 80

# "Fixes: test_wrap_around, test_reset" - the fix ports' attribution line, which
# says which failing tests one order clears.  It is METADATA about the order
# above it, not an instruction of its own: mined as an instruction it renders to
# the RTL writer as "must appear: a test named test_wrap_around", and
# [create_code_hdl] forbids that block from writing tests at all.
_ATTRIBUTION_LINE_RE = re.compile(r"^fixes\s*:", re.IGNORECASE)

# Bullet/numbering noise stripped from the front of an instruction line so the
# verbatim text reads as a sentence rather than a fragment of markdown.
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|\[[ xX]\])\s*")

#: An instruction line longer than this is truncated in the prompt hint (never in
#: the verbatim ``text``, which the user has to be able to recognise).
_HINT_MAX = 120


def _lines(text: str) -> list[str]:
    """Instruction lines: bullets de-bulleted, blanks and headings dropped."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _BULLET_RE.sub("", raw).strip()
        if not line:
            continue
        # A markdown heading is a section label, not an instruction.
        if line.startswith("#") or set(line) <= set("=-_ "):
            continue
        out.append(line)
    return out


def _must_appear(line: str) -> tuple[list[str], str]:
    """The ``[must appear: ...]`` literals on a line, and the line without them.

    Mined and REMOVED before anything else looks at the line, for two reasons:
    the clause splitter must never cut a bracket in half (an HDL literal is full
    of semicolons, and half a hint is a hint for something nobody asked for),
    and ``render_instructions`` appends its own "[must appear: ...]" suffix, so
    a bracket left in the text would be printed twice.

    Literals we cannot decide are dropped rather than guessed at - see
    ``_MUST_APPEAR_MAX``.
    """
    literals: list[str] = []
    for body in _MUST_APPEAR_RE.findall(line):
        for item in body.split(";"):
            item = item.strip().strip("`").strip()
            if item and len(item) <= _MUST_APPEAR_MAX:
                literals.append(item)
    return literals, _MUST_APPEAR_RE.sub(" ", line).strip()


def _snippets(line: str) -> list[str]:
    """Backticked code-looking snippets in one line."""
    return [
        s.strip() for s in _BACKTICK_RE.findall(line)
        if _CODEISH_RE.search(s)
    ]


def _tokens(line: str) -> list[str]:
    """Candidate identifiers in one clause.

    Backticked names WIN OUTRIGHT when present: the user who wrote
    ``use `always_comb``` named their target, and falling through to the bare-token
    scan would let an identifier from elsewhere in the sentence be picked instead.
    """
    quoted = [s.strip() for s in _BACKTICK_RE.findall(line) if not _CODEISH_RE.search(s)]
    found = quoted or _TOKEN_RE.findall(line)
    seen: list[str] = []
    for t in found:
        if t and t not in seen:
            seen.append(t)
    return seen


# Where the SIGN of an instruction can flip: a semicolon, an "instead of", or a
# sentence boundary. The last one matters as much as the others - "Use
# `always_ff` for state. Never use `$display`." is a requirement followed by a
# prohibition, and reading the whole line as one prohibition forbids `always_ff`.
# The two-letter lookbehind keeps "e.g." and "i.e." from splitting a sentence in
# half (the character before that period is another period, not a letter).
# The subset of those markers that CONTRAST rather than merely separate. What
# follows one of them is the thing to get RID of, so its sign is inverted:
# "use `always_comb` instead of `always @(*)`" requires the first and forbids the
# second, and reading the tail as another requirement demands the very construct
# the instruction was written to remove -- which then rejects the design that
# obeyed. ",\s*not\s+" is the phrasing the verilator fix ports produce
# ("compare `count == DEPTH`, not `count == DEPTH-1`"); a bare " not " is
# deliberately NOT a marker, or "do not use X" would split into "do" and "use X".
_CONTRAST_RE = re.compile(r"\s+(?:instead\s+of|rather\s+than)\s+|,\s*not\s+")

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*;\s*"
    r"|\s+(?:instead\s+of|rather\s+than)\s+"
    r"|,\s*not\s+"
    r"|(?<=[a-z)][a-z)])\.\s+"
)


def _clauses(line: str) -> list[tuple[str, bool]]:
    """One instruction line split into independently-signed clauses.

    Each clause comes back with a flag saying whether a CONTRAST marker put it
    on the far side of a "not"/"instead of", in which case its sign is inverted.

    ``do not use `always @(*)`; use `always_comb` instead`` is a prohibition AND a
    requirement, and a single sign for the whole line gets one of them backwards
    - which is worse than missing it, because the checker then rejects the design
    that obeyed. Semicolons and "instead"/"rather than" are where the sign flips
    in practice; commas and "and" are not, so they are deliberately left alone.

    A backticked span is NEVER split, even though HDL snippets are full of
    semicolons: "the reset must read `if (!rst_n) q <= 1'b0;`" is one quoted
    literal, and cutting it in half loses the closing backtick - which silently
    demotes the whole instruction to unverifiable prose.
    """
    spans = [(m.start(), m.end()) for m in _BACKTICK_RE.finditer(line)]

    def quoted(pos: int) -> bool:
        return any(start <= pos < end for start, end in spans)

    parts: list[tuple[str, bool]] = []
    cursor = 0
    inverted = False
    for m in _CLAUSE_SPLIT_RE.finditer(line):
        if quoted(m.start()):
            continue
        parts.append((line[cursor:m.start()], inverted))
        # Only the clause DIRECTLY after a contrast marker is inverted; the next
        # separator decides the one after that from scratch.
        inverted = bool(_CONTRAST_RE.fullmatch(m.group(0)))
        cursor = m.end()
    parts.append((line[cursor:], inverted))
    out = [(p.strip(), inv) for p, inv in parts if p.strip()]
    return out or [(line, False)]


def extract_directives(
    text: str, *, source: str, binding: bool = True
) -> list[Directive]:
    """Every directive in one block of instruction text, in reading order.

    One line can yield several directives (``"set DATA_WIDTH = 8 and add an
    `almost_full` output"``), and always yields at least the ``prose`` fallback
    so the line still reaches the prompt and the compliance audit.
    """
    directives: list[Directive] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, target: str, line: str, value: str = "") -> bool:
        key = (kind, target, value)
        if not target or key in seen:
            return False
        seen.add(key)
        directives.append(Directive(
            kind=kind, target=target, value=value, text=line,
            source=source, binding=binding,
        ))
        return True

    # Fenced blocks are whole-snippet literals and are handled before the line
    # walk, which would otherwise see each of their lines as an instruction.
    body = text or ""
    for block in _FENCE_BLOCK_RE.findall(body):
        snippet = block.strip()
        if snippet and _CODEISH_RE.search(snippet):
            add(KIND_LITERAL, snippet, snippet)
    body = _FENCE_BLOCK_RE.sub(" ", body)

    # The order a standalone "[must appear: ...]" line qualifies.  The fix ports
    # put the hint on its own line under the instruction it belongs to, and a
    # miss must quote the instruction the user was given, not the machinery
    # around it.
    last_order = ""

    for line in _lines(body):
        decided = False

        hinted, line = _must_appear(line)
        for literal in hinted:
            decided |= add(KIND_LITERAL, literal, last_order or line or literal)
        if not line or _ATTRIBUTION_LINE_RE.match(line):
            # A hint-only or "Fixes:" line is not itself an instruction, so it
            # gets no prose fallback: it would print as a bullet saying nothing,
            # or worse, as an order aimed at the wrong artifact.
            continue
        last_order = line

        for clause, inverted in _clauses(line):
            # The verbatim LINE is what the user is shown, even when only one of
            # its clauses produced the directive: a fragment on `improvements`
            # sends them hunting for a sentence they never wrote.
            forbidding = bool(_FORBID_RE.search(clause)) != inverted

            for name, value in _PARAM_DECL_RE.findall(clause):
                decided |= add(KIND_PARAMETER, name, line, value)
            for name, value in _PARAM_PROSE_RE.findall(clause):
                decided |= add(KIND_PARAMETER, name, line, value)

            for name in _TEST_NAME_RE.findall(clause):
                decided |= add(KIND_TEST, name, line)

            for req in _REQ_ID_RE.findall(clause):
                decided |= add(KIND_REQUIREMENT, req, line)

            # A backticked snippet in a prohibition must be ABSENT, not present.
            for snippet in _snippets(clause):
                kind = KIND_FORBIDDEN if forbidding else KIND_LITERAL
                decided |= add(kind, snippet, line)

            # A port instruction is checked against the port list, so it is
            # matched before the generic require/forbid pass claims its token.
            if not forbidding and _ADD_SIGNAL_RE.search(clause):
                named = _SIGNAL_ALT_RE.findall(clause)
                for name in named or _tokens(clause)[:1]:
                    decided |= add(KIND_SIGNAL, name, line)
            elif forbidding:
                for token in _tokens(clause)[:2]:
                    decided |= add(KIND_FORBIDDEN, token, line)
            elif _REQUIRE_RE.search(clause):
                for token in _tokens(clause)[:2]:
                    decided |= add(KIND_IDENTIFIER, token, line)

        if not decided:
            add(KIND_PROSE, line, line)

    return directives


def parameter_directives_from_spec(spec: str) -> list[Directive]:
    """Parameter values stated numerically inside a spec, as NON-binding directives.

    ``code_hdl`` has no ``parameters`` input port: a resolved parameter table
    reaches it only inside the ``spec`` text. So a spec saying ``FIFO_DEPTH = 32``
    is the commonest way that value arrives, and the [create_code_hdl] prompt
    already promises to "parameterise widths and depths the SPEC states as
    numbers".

    Non-binding on purpose. This is the server READING prose rather than the user
    stating a rule, and a spec may legitimately name a value it does not intend as
    a parameter (a throughput figure, an example transaction). A miss is worth
    telling the user about; it is not worth rejecting the design over.
    """
    found = extract_directives(spec, source="spec", binding=False)
    return [d for d in found if d.kind == KIND_PARAMETER]


#: ``"Data width: 16"`` - a ``spec_hdl_parameters`` line whose value is a bare
#: number. The label is prose, so the number is the only checkable part.
_PINNED_VALUE_RE = re.compile(rf"^([^:]{{2,60}}):\s*({_PARAM_VALUE})\s*$")


def pinned_parameter_directives(lines: list[str]) -> list[Directive]:
    """Directives for the design parameters the user pinned on spec_hdl's ports.

    Takes ``router.spec_hdl_parameters`` output verbatim - ``"Data width: 16"``,
    ``"Reset style: async_active_low"``, ``"Further parameters: FIFO_DEPTH = 32"``
    - which has already dropped every port still sitting at its "you decide"
    default.

    Three outcomes, and the split is the point:

    * a numeric value gets :data:`KIND_VALUE`: the number must appear in the
      artifact, but the design may name the parameter whatever it likes;
    * an ``UPPER_SNAKE = value`` inside the free-form ``parameters`` field gets
      the ordinary :data:`KIND_PARAMETER` treatment, name included;
    * a stylistic value (``async_active_low``, a protocol name) stays ``prose``.
      A spec honouring it will say "asynchronous, active-low", not echo the token,
      so requiring the literal would fail every correct answer. The prompt and
      the model's ``compliance`` line carry these instead.
    """
    out: list[Directive] = []
    for line in lines or []:
        text = (line or "").strip()
        if not text:
            continue
        mined = extract_directives(text, source="parameters")
        # A line that yielded a real named parameter needs no label fallback.
        if any(d.kind != KIND_PROSE for d in mined):
            out.extend(mined)
            continue
        m = _PINNED_VALUE_RE.match(text)
        if m:
            out.append(Directive(
                kind=KIND_VALUE, target=m.group(1).strip(), value=m.group(2),
                text=text, source="parameters", binding=True,
            ))
        else:
            out.extend(mined)
    return out


def collect_directives(sources: dict[str, str]) -> list[Directive]:
    """Directives from several named instruction ports, in the given order.

    ``sources`` maps a port name to its text; empty entries are skipped. Order is
    the caller's — put the most authoritative port first, since that is the order
    the model reads them in.
    """
    out: list[Directive] = []
    for source, text in sources.items():
        if (text or "").strip():
            out.extend(extract_directives(text, source=source))
    return out


def has_interface_directive(directives: list[Directive]) -> bool:
    """True when any directive would change the module's port list.

    The router calls this to keep such an instruction off the [fix_rtl] prompt,
    whose frozen-interface rule ``hdl.validate_rtl_fix`` enforces — there, a
    design that obeyed the instruction would be REJECTED for interface drift.
    """
    return any(d.kind in INTERFACE_KINDS for d in directives)


# ── Rendering into the prompt ─────────────────────────────────────────────────

_INSTRUCTION_HEADER = (
    "MANDATORY INSTRUCTIONS - these override the spec, the defaults and your own "
    "judgement. Every one must be satisfied in the output, literally where a "
    "literal is given. Report each of them in the \"compliance\" key:"
)


def render_instructions(directives: list[Directive]) -> str:
    """The MANDATORY INSTRUCTIONS section of a user message (``""`` when there are none).

    One bullet per source LINE, not per directive: a line that yielded a
    parameter and a port is still one instruction to the user, and repeating it
    twice reads as two conflicting demands. The decidable parts of the line are
    appended as an explicit "must appear" hint, which is what turns "honour the
    feedback" into something the model can copy and the server can check.
    """
    if not directives:
        return ""
    order: list[str] = []
    hints: dict[str, list[str]] = {}
    for d in directives:
        line = d.text.strip()
        if not line:
            continue
        if line not in hints:
            order.append(line)
            hints[line] = []
        hint = d.evidence()
        if hint and hint not in hints[line]:
            hints[line].append(hint)

    bullets = []
    for line in order:
        shown = line if len(line) <= _HINT_MAX * 4 else line[: _HINT_MAX * 4] + " ..."
        detail = "; ".join(hints[line])
        if detail:
            shown = f"{shown}   [must appear: {detail}]"
        bullets.append(f"- {shown}")
    return _INSTRUCTION_HEADER + "\n" + "\n".join(bullets)


# ── Checking a generated artifact ─────────────────────────────────────────────

ARTIFACT_RTL = "rtl"
ARTIFACT_SPEC = "spec"
ARTIFACT_PYTHON = "python"


def _normalize(text: str) -> str:
    """Whitespace-collapsed text, for substring comparison of code snippets."""
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_for(artifact: str, kind: str) -> str:
    """``artifact`` with its comments removed, so a rule met only in a comment fails.

    A design that answers "use ``always_ff``" by writing ``// use always_ff`` has
    not answered it, and a design that answers "do not use ``$display``" is not
    condemned by the word appearing in a comment.
    """
    if kind == ARTIFACT_PYTHON:
        return re.sub(r"#[^\n]*", " ", artifact or "")
    if kind == ARTIFACT_SPEC:
        return artifact or ""
    stripped = _strip_comments(artifact or "")
    # VHDL uses "--"; stripping both is safe because "--" is not a Verilog token.
    return _strip_comments_vhdl(stripped)


def _parameter_met(body: str, name: str, value: str) -> bool:
    """A ``parameter``/``localparam`` declaration of ``name`` set to ``value``."""
    pattern = re.compile(
        r"\b(?:parameter|localparam|generic)\b[^;)]{0,200}?\b"
        + re.escape(name)
        + r"\b[^;)]{0,80}?=\s*"
        + re.escape(value)
        + r"\b"
    )
    if pattern.search(body):
        return True
    # A design may declare the parameter and assign it on the following line, or
    # inside a parameter list where the keyword appears only once.
    loose = re.compile(r"\b" + re.escape(name) + r"\b\s*=\s*" + re.escape(value) + r"\b")
    return bool(loose.search(body))


def _spec_parameter_met(body: str, name: str, value: str) -> bool:
    """``name`` and ``value`` on the same line of a specification."""
    for line in (body or "").splitlines():
        if re.search(r"\b" + re.escape(name) + r"\b", line) and re.search(
            r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", line
        ):
            return True
    return False


def unmet(
    artifact: str,
    directives: list[Directive],
    *,
    kind: str,
    top: str = "",
    signals: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` for the directives this artifact does not satisfy.

    ``problems`` are binding instructions that were missed and are worth a repair
    round; ``warnings`` are non-binding ones (see
    :func:`parameter_directives_from_spec`). ``prose`` directives are never in
    either list - see the honesty rule in this module's docstring.

    ``signals`` is the artifact's port list when the caller already knows it
    (spec_hdl parses its own ``interface``); for RTL it is derived here.
    """
    if not directives:
        return [], []

    body = _strip_for(artifact, kind)
    ports: list[str] = list(signals) if signals is not None else []
    if kind == ARTIFACT_RTL and signals is None and (artifact or "").strip():
        ports = module_ports(artifact, top or "")

    problems: list[str] = []
    warnings: list[str] = []

    for d in directives:
        missed = False
        if d.kind == KIND_PROSE:
            continue
        if d.kind == KIND_PARAMETER:
            if kind == ARTIFACT_SPEC:
                missed = not _spec_parameter_met(body, d.target, d.value)
            elif kind == ARTIFACT_RTL:
                missed = not _parameter_met(body, d.target, d.value)
            else:
                # A parameter instruction has no meaning for a testbench, which
                # drives ports rather than declaring parameters.
                continue
        elif d.kind == KIND_VALUE:
            missed = not re.search(
                r"(?<![\w.'])" + re.escape(d.value) + r"(?![\w.])", body
            )
        elif d.kind == KIND_LITERAL:
            missed = _normalize(d.target) not in _normalize(body)
        elif d.kind == KIND_IDENTIFIER:
            missed = not re.search(r"(?<![\w$])" + re.escape(d.target) + r"(?![\w$])", body)
        elif d.kind == KIND_FORBIDDEN:
            missed = bool(re.search(r"(?<![\w$])" + re.escape(d.target) + r"(?![\w$])", body))
        elif d.kind == KIND_SIGNAL:
            if kind == ARTIFACT_PYTHON:
                missed = f"dut.{d.target}" not in (artifact or "")
            elif ports:
                missed = d.target not in ports
            else:
                # An unparsable interface means "could not tell", never "missing".
                continue
        elif d.kind == KIND_TEST:
            if kind != ARTIFACT_PYTHON:
                continue
            missed = not re.search(
                r"\bdef\s+" + re.escape(d.target) + r"\s*\(", body
            )
        elif d.kind == KIND_REQUIREMENT:
            if kind != ARTIFACT_SPEC:
                continue
            missed = not re.search(r"\b" + re.escape(d.target) + r"\b", body)
        else:
            continue

        if not missed:
            continue
        message = _miss_message(d)
        (problems if d.binding else warnings).append(message)

    return problems, warnings


def _miss_message(d: Directive) -> str:
    """The sentence the model (and then the user) is shown for a missed directive."""
    where = f" (from the {d.source} port)" if d.source else ""
    if d.kind == KIND_PARAMETER:
        return (
            f"the instruction{where} \"{d.text}\" requires {d.target} = {d.value}, "
            f"and the output does not set it to that value"
        )
    if d.kind == KIND_VALUE:
        return (
            f"the user pinned {d.target} = {d.value}{where}, and {d.value} appears "
            f"nowhere in the output"
        )
    if d.kind == KIND_LITERAL:
        return (
            f"the instruction{where} \"{d.text}\" gives a literal that must appear "
            f"verbatim, and the output does not contain it: {d.target}"
        )
    if d.kind == KIND_IDENTIFIER:
        return (
            f"the instruction{where} \"{d.text}\" requires `{d.target}`, "
            f"which the output never uses"
        )
    if d.kind == KIND_FORBIDDEN:
        return (
            f"the instruction{where} \"{d.text}\" forbids `{d.target}`, "
            f"and the output still uses it"
        )
    if d.kind == KIND_SIGNAL:
        return (
            f"the instruction{where} \"{d.text}\" asks for a port named "
            f"`{d.target}`, which the output does not declare"
        )
    if d.kind == KIND_TEST:
        return (
            f"the instruction{where} \"{d.text}\" asks for a test named "
            f"`{d.target}`, which the output does not define"
        )
    if d.kind == KIND_REQUIREMENT:
        return (
            f"{d.target} was cited by the instruction{where} but no longer appears; "
            f"amend a requirement in place rather than dropping its number"
        )
    return f"the instruction{where} \"{d.text}\" was not applied"


# ── The model's own compliance answer ─────────────────────────────────────────

_NOT_APPLIED_RE = re.compile(r"\bNOT\s+APPLIED\b", re.IGNORECASE)

#: How much of an instruction has to appear in a `compliance` line for it to
#: count as "reported". Short enough to survive the model re-wrapping the text,
#: long enough not to match every line.
_COMPLIANCE_ECHO_CHARS = 24


def compliance_gaps(compliance: str, directives: list[Directive]) -> list[str]:
    """Instructions the model's own audit says it did not apply.

    This is the half of enforcement that :func:`unmet` cannot do: a ``prose``
    instruction has no regex, but the model can still be made to say, line by
    line, what it did with it.

    **An absent ``compliance`` key yields nothing.** A model that never filled the
    key has told us nothing about any individual instruction, and treating that
    silence as failure would charge a repair round for every prose constraint on
    every run - the checker's shortcoming billed to the user. The decidable
    directives have real checks (:func:`unmet`); this function only ever reports
    what the answer itself put in writing.

    A ``compliance`` that IS filled in but skips an instruction is different: the
    answer enumerated its work and that instruction was not in it, which is
    evidence rather than silence. That counts as a gap.
    """
    lines = [line.strip() for line in (compliance or "").splitlines() if line.strip()]
    if not lines:
        return []
    unique: list[Directive] = []
    seen: set[str] = set()
    for d in directives:
        text = d.text.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(d)
    if not unique:
        return []

    gaps: list[str] = []
    for d in unique:
        probe = _normalize(d.text)[:_COMPLIANCE_ECHO_CHARS].lower()
        matches = [line for line in lines if probe and probe in _normalize(line).lower()]
        if not matches:
            gaps.append(
                f"the answer never reported what it did with the instruction "
                f"\"{d.text}\"; every instruction must appear in \"compliance\""
            )
        elif all(_NOT_APPLIED_RE.search(line) for line in matches):
            gaps.append(
                f"the answer reports the instruction \"{d.text}\" as NOT APPLIED; "
                f"apply it, or state in \"improvements\" why it is impossible"
            )
    return gaps


# ── Reporting residual misses to the user ─────────────────────────────────────

#: Heading the app and the user look for on ``improvements``. Kept as a constant
#: because the tests and (later) any client-side rendering anchor on it.
UNMET_HEADING = (
    "UNMET INSTRUCTIONS (this run did not satisfy these; they are unchanged on "
    "the input port):"
)


def render_unmet(problems: list[str], warnings: list[str] = ()) -> str:
    """The ``improvements`` section listing what the run failed to honour.

    Returns ``""`` when everything was satisfied, so an ordinary run's
    ``improvements`` port is not decorated with a heading and nothing under it.
    """
    items = [*problems, *warnings]
    if not items:
        return ""
    return UNMET_HEADING + "\n- " + "\n- ".join(items)
