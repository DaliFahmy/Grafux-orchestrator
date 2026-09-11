"""
The instruction-compliance layer: what a directive is, and whether it was honoured.

These tests pin the two halves separately. Extraction is about mining checkable
demands out of prose without inventing any; checking is about deciding, from the
generated artifact alone, whether each demand was met. The tests that matter most
are the NEGATIVE ones — the phrasings that must NOT become directives — because a
false directive rejects a correct answer and burns the repair budget the real ones
need.
"""

from __future__ import annotations

from app.modules.blocks import directives as D

# ── The reported bug, as a fixture ───────────────────────────────────────────

FEEDBACK = "make  parameter FIFO_DEPTH = 32;"

RTL_WITHOUT = """
module sync_fifo #(parameter FIFO_DEPTH = 8) (
    input wire clk, input wire rst_n, output wire full
);
endmodule
"""

RTL_WITH = """
module sync_fifo #(parameter FIFO_DEPTH = 32) (
    input wire clk, input wire rst_n, output wire full
);
endmodule
"""


def test_the_reported_instruction_is_extracted_and_checked():
    found = D.extract_directives(FEEDBACK, source="feedback")
    assert [(d.kind, d.target, d.value) for d in found] == [
        (D.KIND_PARAMETER, "FIFO_DEPTH", "32")
    ]

    problems, warnings = D.unmet(RTL_WITHOUT, found, kind="rtl", top="sync_fifo")
    assert warnings == []
    assert len(problems) == 1
    # The user's own sentence, verbatim, so they can recognise it on the port.
    assert FEEDBACK in problems[0]
    assert "FIFO_DEPTH = 32" in problems[0]

    assert D.unmet(RTL_WITH, found, kind="rtl", top="sync_fifo") == ([], [])


def test_a_declaration_the_design_spells_differently_still_counts():
    found = D.extract_directives("FIFO_DEPTH must be 32", source="feedback")
    localparam = "module f(); localparam FIFO_DEPTH = 32; endmodule"
    assert D.unmet(localparam, found, kind="rtl", top="f") == ([], [])


# ── Extraction: the phrasings that must NOT become directives ───────────────


def test_a_depth_in_words_is_not_mined_as_a_parameter():
    # "it" is not a parameter, and a checker that thought so would reject every
    # design for not declaring one.
    found = D.extract_directives("make it 32 bits deep", source="feedback")
    assert [d.kind for d in found] == [D.KIND_PROSE]


def test_a_lowercase_name_is_not_a_parameter():
    found = D.extract_directives("set depth = 32", source="feedback")
    assert all(d.kind == D.KIND_PROSE for d in found)


def test_a_value_that_is_an_expression_is_not_decidable():
    # The design may legally spell $clog2(DEPTH) a different way, so there is
    # nothing to look for.
    found = D.extract_directives("ADDR_WIDTH = $clog2(DEPTH)", source="feedback")
    assert not [d for d in found if d.kind == D.KIND_PARAMETER]


def test_prose_is_carried_but_never_reported_as_unmet():
    """The honesty rule: an unverifiable instruction must not fail an artifact."""
    found = D.extract_directives(
        "the FIFO should be nicer and easier to read", source="feedback"
    )
    assert [d.kind for d in found] == [D.KIND_PROSE]
    assert D.unmet("module f(); endmodule", found, kind="rtl", top="f") == ([], [])
    # It still reaches the model, which is the only enforcement available for it.
    assert "nicer and easier to read" in D.render_instructions(found)


# ── Extraction: mixed-sign lines ────────────────────────────────────────────


def test_a_prohibition_and_a_requirement_in_one_line_keep_their_signs():
    line = "do not use `always @(*)`; use `always_comb` instead"
    found = D.extract_directives(line, source="feedback")
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_FORBIDDEN, "always @(*)"),
        (D.KIND_IDENTIFIER, "always_comb"),
    ]
    # Both are attributed to the whole line, not to the clause that produced them.
    assert all(d.text == line for d in found)


def test_a_forbidden_token_only_in_a_comment_counts_as_absent():
    found = D.extract_directives("never use `$display` in the design", source="feedback")
    commented = "module f();\n  // $display is banned here\nendmodule"
    assert D.unmet(commented, found, kind="rtl", top="f") == ([], [])
    live = "module f();\n  initial $display(\"x\");\nendmodule"
    problems, _ = D.unmet(live, found, kind="rtl", top="f")
    assert len(problems) == 1 and "still uses it" in problems[0]


def test_a_required_identifier_only_in_a_comment_is_not_satisfied():
    found = D.extract_directives("use `always_ff` for state", source="feedback")
    commented = "module f();\n  // one day, always_ff\nendmodule"
    problems, _ = D.unmet(commented, found, kind="rtl", top="f")
    assert len(problems) == 1 and "never uses" in problems[0]


def test_several_directives_come_out_of_one_line():
    found = D.extract_directives(
        "set DATA_WIDTH to 16 and add an output port called almost_full",
        source="feedback",
    )
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_PARAMETER, "DATA_WIDTH"),
        (D.KIND_SIGNAL, "almost_full"),
    ]


def test_a_literal_snippet_is_matched_whitespace_insensitively():
    found = D.extract_directives(
        "the reset must read `if (!rst_n) q <= 1'b0;`", source="constraints"
    )
    assert [d.kind for d in found] == [D.KIND_LITERAL]
    reflowed = "module f();\n  if   (!rst_n)\n      q <= 1'b0;\nendmodule"
    assert D.unmet(reflowed, found, kind="rtl", top="f") == ([], [])


def test_a_fenced_block_is_one_literal_not_a_list_of_instructions():
    text = "use this exactly:\n```verilog\nassign full = (count == DEPTH);\n```"
    found = D.extract_directives(text, source="feedback")
    literals = [d for d in found if d.kind == D.KIND_LITERAL]
    assert len(literals) == 1
    assert "assign full = (count == DEPTH);" in literals[0].target


def test_bullets_and_numbering_are_stripped_from_the_instruction_text():
    found = D.extract_directives(
        "- make parameter DEPTH_X = 4\n2) use `always_ff`", source="feedback"
    )
    assert [d.text for d in found] == [
        "make parameter DEPTH_X = 4", "use `always_ff`",
    ]


def test_headings_and_blank_lines_are_not_instructions():
    found = D.extract_directives("## Review\n\n----\n\n", source="feedback")
    assert found == []


# ── Interface-changing directives (the [fix_rtl] routing decision) ──────────


def test_only_a_port_change_counts_as_interface_changing():
    port = D.extract_directives("add an input port called almost_empty", source="feedback")
    assert D.has_interface_directive(port)

    # A width change alters no port NAME, and validate_rtl_fix compares names and
    # order only, so it is safe to leave on the repair prompt.
    width = D.extract_directives("make parameter DATA_WIDTH = 64", source="feedback")
    assert not D.has_interface_directive(width)


def test_a_simulator_failure_report_yields_no_interface_directive():
    """What the verify loop writes to `code.feedback` must stay on [fix_rtl]."""
    failures = (
        "## test_full_flag\n"
        "WHY: full asserted one cycle early (expected 0, got 1)\n"
        "WHERE: test_sync_fifo.py:42\n"
        "    assert int(dut.full.value) == 0\n"
        "LIKELY CAUSE: off-by-one in the pointer comparison\n"
    )
    found = D.extract_directives(failures, source="feedback")
    assert not D.has_interface_directive(found)


def test_an_unparsable_interface_means_could_not_tell_not_missing():
    found = D.extract_directives("add an output port called almost_full", source="feedback")
    # No module header to parse: reporting the port as missing would condemn a
    # design on the strength of a regex that failed.
    assert D.unmet("not verilog at all", found, kind="rtl", top="f") == ([], [])


# ── Pinned design parameters (spec_hdl) ────────────────────────────────────


def test_pinned_numeric_parameters_are_checked_by_their_value():
    ds = D.pinned_parameter_directives(["Data width: 16", "Address width: 3"])
    assert [(d.kind, d.target, d.value) for d in ds] == [
        (D.KIND_VALUE, "Data width", "16"),
        (D.KIND_VALUE, "Address width", "3"),
    ]
    problems, _ = D.unmet("An 8-bit FIFO.", ds, kind="spec")
    assert len(problems) == 2
    assert D.unmet("16 bits wide, 3 address bits.", ds, kind="spec") == ([], [])


def test_a_pinned_stylistic_value_stays_prose():
    # A spec honouring async_active_low writes "asynchronous, active-low"; demanding
    # the token back would fail every correct answer.
    ds = D.pinned_parameter_directives(["Reset style: async_active_low"])
    assert [d.kind for d in ds] == [D.KIND_PROSE]
    assert D.unmet("Reset is asynchronous and active-low.", ds, kind="spec") == ([], [])


def test_a_named_parameter_inside_the_free_form_field_is_checked_by_name():
    ds = D.pinned_parameter_directives(["Further parameters: FIFO_DEPTH = 32"])
    assert [(d.kind, d.target, d.value) for d in ds] == [
        (D.KIND_PARAMETER, "FIFO_DEPTH", "32")
    ]


def test_a_parameter_read_out_of_the_spec_is_a_warning_not_a_problem():
    """The server inferring a rule must not reject a design the user never faulted."""
    ds = D.parameter_directives_from_spec("The FIFO has FIFO_DEPTH = 32 entries.")
    assert [d.binding for d in ds] == [False]
    problems, warnings = D.unmet(RTL_WITHOUT, ds, kind="rtl", top="sync_fifo")
    assert problems == []
    assert len(warnings) == 1


# ── Requirement ids (spec_hdl revisions) ───────────────────────────────────


def test_a_cited_requirement_must_survive_a_revision():
    ds = D.extract_directives("REQ-4 never states the reset value", source="feedback")
    assert [(d.kind, d.target) for d in ds] == [(D.KIND_REQUIREMENT, "REQ-4")]
    problems, _ = D.unmet("REQ-1: it works.", ds, kind="spec")
    assert len(problems) == 1 and "REQ-4" in problems[0]
    assert D.unmet("REQ-4: reset drives q to 0.", ds, kind="spec") == ([], [])


# ── Test names (testbench) ─────────────────────────────────────────────────


def test_a_named_test_is_checked_against_the_python_source():
    ds = D.extract_directives("add a test named test_wrap_around", source="extra_tests")
    assert [(d.kind, d.target) for d in ds] == [(D.KIND_TEST, "test_wrap_around")]
    without = "import cocotb\n@cocotb.test()\nasync def test_reset(dut): pass\n"
    problems, _ = D.unmet(without, ds, kind="python")
    assert len(problems) == 1 and "test_wrap_around" in problems[0]
    with_it = without + "@cocotb.test()\nasync def test_wrap_around(dut): pass\n"
    assert D.unmet(with_it, ds, kind="python") == ([], [])


def test_a_parameter_instruction_is_meaningless_for_a_testbench():
    ds = D.extract_directives(FEEDBACK, source="feedback")
    # A testbench declares no parameters; checking it for one would fail every run.
    assert D.unmet("import cocotb\n", ds, kind="python") == ([], [])


def test_a_port_directive_is_checked_as_a_dut_reference_in_python():
    ds = D.extract_directives("add an output port called almost_full", source="feedback")
    problems, _ = D.unmet("assert int(dut.full.value) == 0", ds, kind="python")
    assert len(problems) == 1
    assert D.unmet("assert int(dut.almost_full.value) == 0", ds, kind="python") == ([], [])


# ── Rendering into the prompt ──────────────────────────────────────────────


def test_the_instruction_block_names_what_will_be_looked_for():
    ds = D.extract_directives(FEEDBACK, source="feedback")
    block = D.render_instructions(ds)
    assert "MANDATORY INSTRUCTIONS" in block
    assert FEEDBACK in block
    assert "[must appear: parameter FIFO_DEPTH = 32]" in block


def test_one_line_yielding_two_directives_is_still_one_bullet():
    ds = D.extract_directives(
        "set DATA_WIDTH to 16 and add an output port called almost_full",
        source="feedback",
    )
    block = D.render_instructions(ds)
    assert block.count("- set DATA_WIDTH") == 1
    # ...with both demands named on it, so nothing is lost by the de-duplication.
    assert "parameter DATA_WIDTH = 16" in block
    assert "a port named almost_full" in block


def test_no_directives_renders_nothing():
    assert D.render_instructions([]) == ""


# ── The model's own compliance answer ──────────────────────────────────────


def test_an_absent_compliance_key_reports_nothing():
    """Silence is not evidence, and charging a repair round for it is the checker's bug."""
    ds = D.collect_directives({"feedback": "make parameter DEPTH_X = 4\nrename busy"})
    assert D.compliance_gaps("", ds) == []
    assert D.compliance_gaps("   \n\n ", ds) == []


def test_a_filled_compliance_key_that_skips_an_instruction_is_a_gap():
    ds = D.collect_directives({"feedback": "make parameter DEPTH_X = 4\nrename busy"})
    answered = "make parameter DEPTH_X = 4 -> APPLIED: parameter list"
    gaps = D.compliance_gaps(answered, ds)
    assert len(gaps) == 1 and "rename busy" in gaps[0]


def test_a_declared_not_applied_is_a_gap():
    ds = D.collect_directives({"feedback": "rename busy to active"})
    gaps = D.compliance_gaps("rename busy to active -> NOT APPLIED: no such signal", ds)
    assert len(gaps) == 1 and "NOT APPLIED" in gaps[0]


def test_a_fully_answered_compliance_key_has_no_gaps():
    ds = D.collect_directives({"feedback": "make parameter DEPTH_X = 4\nrename busy"})
    answered = (
        "make parameter DEPTH_X = 4 -> APPLIED: parameter list of fifo\n"
        "rename busy -> APPLIED: busy is now active"
    )
    assert D.compliance_gaps(answered, ds) == []


def test_no_instructions_means_no_compliance_expectation():
    assert D.compliance_gaps("", []) == []
    assert D.compliance_gaps("anything at all", []) == []


# ── Reporting to the user ──────────────────────────────────────────────────


def test_the_unmet_section_is_absent_when_everything_was_honoured():
    assert D.render_unmet([], []) == ""


def test_the_unmet_section_lists_problems_and_warnings_under_one_heading():
    text = D.render_unmet(["a problem"], ["a warning"])
    assert text.startswith(D.UNMET_HEADING)
    assert "- a problem" in text and "- a warning" in text


# ── Collecting from several ports ──────────────────────────────────────────


def test_collect_keeps_port_order_and_skips_empty_ports():
    ds = D.collect_directives({
        "feedback": "make parameter DEPTH_X = 4",
        "constraints": "",
        "extra_tests": "add test_wrap",
    })
    assert [d.source for d in ds] == ["feedback", "extra_tests"]


def test_a_directive_is_not_duplicated_within_one_source():
    ds = D.extract_directives(
        "make parameter DEPTH_X = 4\nagain: DEPTH_X = 4", source="feedback"
    )
    assert len([d for d in ds if d.kind == D.KIND_PARAMETER]) == 1


def test_a_sentence_boundary_flips_the_sign_too():
    """"Use X. Never use Y." is a requirement and a prohibition, not two of one."""
    found = D.extract_directives(
        "Use `always_ff` for state. Never use `$display`.", source="constraints"
    )
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_IDENTIFIER, "always_ff"),
        (D.KIND_FORBIDDEN, "$display"),
    ]


def test_an_abbreviation_does_not_split_a_sentence():
    found = D.extract_directives(
        "set DEPTH_X to 4, e.g. four entries. Do not use `always @(*)`.",
        source="constraints",
    )
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_PARAMETER, "DEPTH_X"),
        (D.KIND_FORBIDDEN, "always @(*)"),
    ]


# ── The verilator fix ports' own wire format ─────────────────────────────────
#
# `fix_rtl` / `fix_tb` / `fix_spec` are wired into code_hdl.feedback,
# testbench.feedback and spec_hdl.feedback, so their text lands here.  These
# cases pin the three pieces of that format: the "[must appear: ...]" hint the
# triage writes about its own order, the "Fixes:" attribution line, and the
# "X, not Y" contrast that says which of two literals must survive.

FIX_RTL_PORT = """1. In `sync_fifo`, drive `full` from `count == DEPTH`, not `count == DEPTH-1`.
   [must appear: count == DEPTH]
   Fixes: test_full_flag_asserts_at_depth, test_write_when_full_is_ignored

2. Reset `rd_ptr` on `rst_n` low; it is currently only reset in the write branch.
   Fixes: test_reset_clears_pointers
"""


def test_a_must_appear_hint_becomes_a_checkable_literal():
    """The triage names its own evidence, and the server holds the answer to it.

    That is what makes a repair order enforceable rather than advisory.
    """
    found = D.extract_directives("[must appear: count == DEPTH]", source="feedback")
    assert [(d.kind, d.target) for d in found] == [(D.KIND_LITERAL, "count == DEPTH")]


def test_a_must_appear_hint_is_attributed_to_the_order_above_it():
    """A miss must quote the instruction, not the machinery around it."""
    found = D.extract_directives(
        "Drive `full` from the count comparison.\n[must appear: count == DEPTH]",
        source="feedback",
    )
    literals = [d for d in found if d.kind == D.KIND_LITERAL]
    assert literals and literals[0].text == "Drive `full` from the count comparison."


def test_a_must_appear_hint_is_not_rendered_twice():
    """render_instructions appends its own "[must appear: ...]" suffix.

    Leaving the bracket in the text prints the same demand as prose AND as a
    hint, which reads as two conflicting instructions.
    """
    found = D.extract_directives(
        "Drive `full` from the count comparison.\n[must appear: count == DEPTH]",
        source="feedback",
    )
    rendered = D.render_instructions(found)
    assert rendered.count("[must appear:") == 1


def test_an_undecidable_must_appear_hint_is_dropped():
    """A whole pasted line is not something substring presence can honestly judge.

    Same honesty rule as `prose`: a hint the checker cannot decide rejects a
    CORRECT answer and burns the rounds the real ones need.
    """
    long_hint = "x" * 200
    found = D.extract_directives(f"do it\n[must appear: {long_hint}]", source="feedback")
    assert not [d for d in found if d.kind == D.KIND_LITERAL]


def test_several_must_appear_literals_are_split_on_semicolons():
    found = D.extract_directives(
        "[must appear: count == DEPTH; almost_full]", source="feedback"
    )
    assert [d.target for d in found if d.kind == D.KIND_LITERAL] == [
        "count == DEPTH", "almost_full",
    ]


def test_a_fixes_line_is_metadata_not_an_instruction():
    """It says which failures the order clears; it is not an order of its own.

    Mined as one it reaches the RTL writer as "must appear: a test named
    test_wrap_around", and [create_code_hdl] forbids that block writing tests.
    """
    found = D.extract_directives(
        "Widen `wr_ptr` to 4 bits.\nFixes: test_wrap_around, test_reset",
        source="feedback",
    )
    assert not [d for d in found if d.kind == D.KIND_TEST]
    assert not any(d.text.startswith("Fixes:") for d in found)


def test_a_contrast_inverts_the_literal_that_follows_it():
    """"X, not Y" requires X and FORBIDS Y.

    Read as two requirements it demands the very expression the fix was written
    to remove, and then rejects the design that obeyed.
    """
    found = D.extract_directives(
        "compare `count == DEPTH`, not `count == DEPTH-1`", source="feedback"
    )
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_LITERAL, "count == DEPTH"),
        (D.KIND_FORBIDDEN, "count == DEPTH-1"),
    ]


def test_instead_of_inverts_its_tail_as_well():
    found = D.extract_directives(
        "use `always_comb` instead of `always @(*)`", source="constraints"
    )
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_IDENTIFIER, "always_comb"),
        (D.KIND_FORBIDDEN, "always @(*)"),
    ]


def test_a_bare_not_is_not_a_contrast_marker():
    """Splitting on " not " would cut "do not use X" into "do" and "use X"."""
    found = D.extract_directives("do not use `always @(*)`", source="constraints")
    assert [(d.kind, d.target) for d in found] == [
        (D.KIND_FORBIDDEN, "always @(*)"),
    ]


def test_a_whole_fix_rtl_port_reads_as_two_orders():
    found = D.extract_directives(FIX_RTL_PORT, source="feedback")
    kinds = [(d.kind, d.target) for d in found]
    assert (D.KIND_LITERAL, "count == DEPTH") in kinds
    assert (D.KIND_FORBIDDEN, "count == DEPTH-1") in kinds
    # No test-existence demand reaches the RTL writer.
    assert not [d for d in found if d.kind == D.KIND_TEST]
    # Two bullets, one per numbered order - not one per line of the port.
    assert D.render_instructions(found).count("\n- ") == 2


def test_the_fix_rtl_port_rejects_the_design_that_ignored_it():
    ignored = """module sync_fifo(input clk);
    assign full = (count == DEPTH-1);
endmodule"""
    applied = """module sync_fifo(input clk);
    assign full = (count == DEPTH);
    always @(posedge clk) if (!rst_n) rd_ptr <= 0;
endmodule"""
    found = D.extract_directives(FIX_RTL_PORT, source="feedback")
    problems, _ = D.unmet(ignored, found, kind=D.ARTIFACT_RTL, top="sync_fifo")
    assert problems
    problems, _ = D.unmet(applied, found, kind=D.ARTIFACT_RTL, top="sync_fifo")
    assert problems == []


# ── ARTIFACT_CODE: a program in an unknown language (the code_fix block) ──────


def test_the_comment_style_is_resolved_per_language_and_admits_ignorance():
    assert D.code_comment_style("Python") == D._CODE_COMMENT_HASH
    assert D.code_comment_style("C++") == D._CODE_COMMENT_SLASH
    assert D.code_comment_style("TypeScript") == D._CODE_COMMENT_SLASH
    assert D.code_comment_style("sql") == D._CODE_COMMENT_DASH
    # "" is a real answer: it means "cannot tell code from commentary".
    assert D.code_comment_style("brainfuck") == ""
    assert D.code_comment_style("") == ""


def test_python_floor_division_is_not_mistaken_for_a_comment():
    """The reason there is no language-agnostic stripper.

    ``a // b`` is a division in Python and a comment in C; ``#define`` is a
    directive in C and a comment in Python. A stripper that guessed would delete
    real code and then report a rule unmet that the program actually satisfies.
    """
    assert "b" in D._strip_for("a = x // b", D.ARTIFACT_CODE, "python")
    assert "eval" not in D._strip_for("int a; // eval", D.ARTIFACT_CODE, "c++")
    assert "MAX" in D._strip_for("#define MAX 5", D.ARTIFACT_CODE, "c")
    assert "MAX" not in D._strip_for("# define MAX 5", D.ARTIFACT_CODE, "python")
    # An unrecognised language strips nothing rather than guessing.
    assert D._strip_for("a // b # c -- d", D.ARTIFACT_CODE, "") == "a // b # c -- d"


def test_a_prohibition_is_not_broken_by_its_own_mention_in_a_comment():
    found = D.extract_directives("Do not use `eval`.", source="fix")
    mentioned = '# this no longer calls eval\nrows = parse(raw)\n'
    used = 'rows = eval(raw)\n'
    assert D.unmet(mentioned, found, kind=D.ARTIFACT_CODE, language="python")[0] == []
    assert D.unmet(used, found, kind=D.ARTIFACT_CODE, language="python")[0]


def test_a_prohibition_is_dropped_when_the_language_is_unknown():
    """Honesty rule. Without the comment boundary the check cannot be made.

    A prohibition is the one check that FAILS on a mention, so an unstripped comment
    would be a FALSE violation - and a forbidden directive is binding, so that costs
    a repair round and can end the run in a failure the program did not earn. A
    missed violation costs nothing, so ignorance resolves this way round.
    """
    found = D.extract_directives("Do not use `eval`.", source="fix")
    used = "rows = eval(raw)\n"
    assert D.unmet(used, found, kind=D.ARTIFACT_CODE, language="python")[0]
    assert D.unmet(used, found, kind=D.ARTIFACT_CODE, language="")[0] == []


def test_a_pinned_constant_is_checked_as_an_assignment_not_a_declaration():
    """``parameter`` is an HDL keyword; the assignment is what survives translation."""
    found = D.extract_directives("set MAX_ROWS to 500", source="fix")
    assert [d.kind for d in found] == [D.KIND_PARAMETER]
    for body in ("MAX_ROWS = 500", "const MAX_ROWS = 500;", "self.MAX_ROWS = 500"):
        assert D.unmet(body, found, kind=D.ARTIFACT_CODE, language="python")[0] == [], body
    assert D.unmet("MAX_ROWS := 500", found, kind=D.ARTIFACT_CODE, language="go")[0] == []
    assert D.unmet("#define MAX_ROWS 500", found, kind=D.ARTIFACT_CODE, language="c")[0] == []
    # ...and the near misses stay misses.
    for body in ("MAX_ROWS = 300", "MAX_ROWS = 5000", "FOO_MAX_ROWS = 500",
                 "print(MAX_ROWS)"):
        assert D.unmet(body, found, kind=D.ARTIFACT_CODE, language="python")[0], body


def test_hdl_and_cocotb_rules_are_never_checked_against_a_general_program():
    """A Python program must not be rejected for not being Verilog or a testbench."""
    assert D.KIND_SIGNAL in D._UNDECIDABLE_IN_CODE
    assert D.KIND_TEST in D._UNDECIDABLE_IN_CODE
    assert D.KIND_REQUIREMENT in D._UNDECIDABLE_IN_CODE
    # ...but a pinned constant IS decidable, so it is deliberately absent.
    assert D.KIND_PARAMETER not in D._UNDECIDABLE_IN_CODE

    signal = D.Directive(
        kind=D.KIND_SIGNAL, target="almost_full", value="", text="add an almost_full output",
        source="fix", binding=True,
    )
    test = D.Directive(
        kind=D.KIND_TEST, target="test_wrap", value="", text="add test_wrap",
        source="fix", binding=True,
    )
    body = "def main():\n    return 0\n"
    assert D.unmet(body, [signal, test], kind=D.ARTIFACT_CODE, language="python") == ([], [])


def test_the_rtl_checks_are_unaffected_by_the_new_kind():
    """The HDL paths must behave exactly as before; ARTIFACT_CODE is additive."""
    found = D.extract_directives(FEEDBACK, source="feedback")
    assert D.unmet(RTL_WITHOUT, found, kind=D.ARTIFACT_RTL, top="sync_fifo")[0]
    assert D.unmet(RTL_WITH, found, kind=D.ARTIFACT_RTL, top="sync_fifo")[0] == []
