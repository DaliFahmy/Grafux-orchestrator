"""
Pure helpers for checking a SPICE netlist an LLM wrote for the analogue_simulator
block, before a pod is rented to find out it does not load.

Sibling of ``hdl.py``: that module stops a testbench from driving signals the
design does not have; this one stops a netlist from naming devices the chosen PDK
does not have.  The device facts mirror ``EDA/ngspice.py`` in Grafux-devices,
which read them from the pinned open_pdks release:

* sky130A MOSFETs are SUBCIRCUITS -- ``XM1 d g s b sky130_fd_pr__nfet_01v8 W=1
  L=0.15`` (W/L in microns).  An ``M1 ... nmos`` line has no model to bind to.
* gf180mcuD MOSFETs are MODELS -- ``M1 d g s b nfet_03v3 W=1u L=0.28u``.
* The devices server adds the PDK's ``.lib`` line (with the corner) and the
  ``.control`` block itself, so a generated deck must carry neither: a second
  ``.lib`` loads the models twice, and a generated ``.control`` switches the
  server's result collection off.

The checks catch what makes ngspice REFUSE a deck (unknown subcircuit, missing
model, unbalanced ``.subckt``) or silently simulate something else (an element
line it drops).  They cannot catch a circuit that loads and is simply wrong; that
is what the run and its review are for.

Everything here is synchronous, side-effect free and unit-tested in isolation.
"""

from __future__ import annotations

import re

PDK_SKY130 = "sky130A"
PDK_GF180 = "gf180mcuD"
PDK_NONE = "none"

_PDK_ALIASES = {
    "": PDK_SKY130, "sky130": PDK_SKY130, "sky130a": PDK_SKY130,
    "sky130hd": PDK_SKY130, "sky130hs": PDK_SKY130,
    "gf180": PDK_GF180, "gf180mcu": PDK_GF180, "gf180mcud": PDK_GF180,
    "none": PDK_NONE, "generic": PDK_NONE,
}

# The MOSFET models gf180mcuD's sm141064.ngspice defines (binned, so the base
# name is what a netlist uses).
GF180_MOS_MODELS = frozenset({
    "nfet_03v3", "pfet_03v3", "nfet_06v0", "pfet_06v0", "nfet_06v0_nvt",
    "nfet_03v3_dss", "pfet_03v3_dss", "nfet_06v0_dss", "pfet_06v0_dss",
})

# gf180 passives are subcircuits too.
_GF180_SUBCKT_PREFIXES = ("ppolyf_", "npolyf_", "nplus_", "pplus_", "rm1", "rm2", "rm3",
                          "tm6k", "tm9k", "tm11k", "tm30k", "nwell", "cap_mim",
                          "cap_nmos", "cap_pmos", "diode_", "np_", "pn_", "vnpn", "vpnp")

_ANALYSIS_RE = re.compile(r"^\s*\.(tran|ac|dc|op|noise|disto|tf|sens|pz)\b", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"^\s*([RCLVIMXDQEGFHBJKSWTUOYPrclvimxdqegfhbjkswtuoyp])\S*\s+\S+")
_SUBCKT_RE = re.compile(r"^\s*\.subckt\s+(\S+)", re.IGNORECASE)
_ENDS_RE = re.compile(r"^\s*\.ends\b", re.IGNORECASE)
_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)", re.IGNORECASE)
_LIB_RE = re.compile(r"^\s*\.(lib|include|inc)\b(.*)$", re.IGNORECASE)
_CONTROL_RE = re.compile(r"^\s*\.control\b", re.IGNORECASE)
_PARAM_VDD_RE = re.compile(r"^\s*\.param\b[^\n]*\bvdd\s*=", re.IGNORECASE | re.MULTILINE)
_VDD_REF_RE = re.compile(r"\{\s*vdd\s*\}", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*$")


def normalize_pdk(text: str) -> str:
    """Canonical PDK name, or the input unchanged when it is not one we know."""
    raw = (text or "").strip()
    return _PDK_ALIASES.get(raw.lower(), raw)


def extract_spice(text: str) -> str:
    """The netlist from an LLM reply, without markdown fences and with LF endings."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln for ln in lines if not _FENCE_RE.match(ln)]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _logical_lines(netlist: str) -> list[str]:
    """Physical lines joined across ``+`` continuations, comments dropped.

    The first line is the TITLE and is excluded: SPICE ignores it, so checking it
    as an element would invent problems.
    """
    raw = (netlist or "").replace("\r\n", "\n").split("\n")[1:]
    out: list[str] = []
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        # Inline comments: ngspice accepts `;` and `$ ` in hsa mode.
        stripped = re.split(r"\s;|\s\$\s", stripped, maxsplit=1)[0].strip()
        if stripped.startswith("+") and out:
            out[-1] = out[-1] + " " + stripped[1:].strip()
        else:
            out.append(stripped)
    return out


def _instance_target(tokens: list[str]) -> str:
    """The model or subcircuit an X/M line names: the last token that is not k=v."""
    for tok in reversed(tokens[1:]):
        if "=" not in tok:
            return tok
    return ""


def validate_netlist(netlist: str, pdk: str = PDK_SKY130, *, analyses: str = "",
                     supply_voltage: str = "") -> list[str]:
    """
    Problems that would make ngspice refuse or mis-read ``netlist``.

    An empty list means "worth simulating", not "correct".  Every problem is
    phrased as an instruction the model can act on in a repair round.
    """
    problems: list[str] = []
    text = netlist or ""
    if not text.strip():
        return ["The netlist is empty."]
    first = text.replace("\r\n", "\n").split("\n", 1)[0].strip()
    # Only a dot-card or continuation is certainly not a title; "rc filter test"
    # and "R1 in out 1k" cannot be told apart, and a false alarm here would burn
    # a repair round on a correct deck.
    if first.startswith(".") or first.startswith("+"):
        problems.append(
            "The first line must be a plain title: SPICE discards it, so the "
            f"line '{first}' would be lost. Start with a one-line title.")

    pdk = normalize_pdk(pdk)
    lines = _logical_lines(text)
    elements = [ln for ln in lines if _ELEMENT_RE.match(ln) and not ln.startswith(".")]
    if not elements:
        problems.append("The netlist has no circuit elements.")

    defined_subckts = {m.group(1).lower() for ln in lines if (m := _SUBCKT_RE.match(ln))}
    defined_models = {m.group(1).lower() for ln in lines if (m := _MODEL_RE.match(ln))}
    has_includes = False

    depth = 0
    for ln in lines:
        if _SUBCKT_RE.match(ln):
            depth += 1
        elif _ENDS_RE.match(ln):
            depth -= 1
            if depth < 0:
                problems.append("A .ends has no matching .subckt.")
                depth = 0
        if _CONTROL_RE.match(ln):
            problems.append(
                "Remove the .control block: the simulator adds its own, and a generated "
                "one stops the results from being collected. Put analyses in the "
                "analyses field instead.")
        lib = _LIB_RE.match(ln)
        if lib:
            has_includes = True
            if pdk in (PDK_SKY130, PDK_GF180):
                problems.append(
                    f"Remove '{ln.strip()}': the server adds the {pdk} model library "
                    "(with the selected corner) itself.")
    if depth > 0:
        problems.append("A .subckt has no matching .ends.")

    if not any(_ANALYSIS_RE.match(ln) for ln in lines) and not (analyses or "").strip():
        problems.append(
            "There is no analysis. Add one to the analyses field (for example "
            "'tran 10p 5n' or 'ac dec 20 1 1G').")

    if _VDD_REF_RE.search(text) and not _PARAM_VDD_RE.search(text) \
            and not (supply_voltage or "").strip():
        problems.append("The netlist uses {vdd} but no supply voltage is set; add "
                        ".param vdd=<volts> or set supply_voltage.")

    for ln in elements:
        tokens = ln.split()
        letter = tokens[0][0].upper()
        if letter not in ("M", "X"):
            continue
        target = _instance_target(tokens).lower()
        if not target:
            continue
        if letter == "X":
            if target in defined_subckts:
                continue
            if pdk == PDK_SKY130 and target.startswith("sky130_fd_pr__"):
                continue
            if pdk == PDK_GF180 and (target in GF180_MOS_MODELS
                                     or target.startswith(_GF180_SUBCKT_PREFIXES)):
                if target in GF180_MOS_MODELS:
                    problems.append(
                        f"'{tokens[0]}' instantiates {target} as a subcircuit, but in "
                        f"gf180mcuD it is a MOSFET model: write M{tokens[0][1:]} ... "
                        f"{target} W=..u L=..u.")
                continue
            if pdk == PDK_GF180 and target.startswith("sky130_"):
                problems.append(f"'{tokens[0]}' uses the sky130 device {target} in a "
                                "gf180mcuD netlist; use nfet_03v3 / pfet_03v3.")
                continue
            if has_includes and pdk == PDK_NONE:
                continue
            problems.append(f"'{tokens[0]}' instantiates subcircuit {target}, which is "
                            "not defined in the netlist or the PDK.")
        else:  # M
            if target in defined_models:
                continue
            if pdk == PDK_SKY130:
                problems.append(
                    f"'{tokens[0]}' is an M-line, but sky130A MOSFETs are subcircuits: "
                    f"write X{tokens[0]} d g s b sky130_fd_pr__nfet_01v8 (or pfet_01v8) "
                    "W=<um> L=<um>.")
            elif pdk == PDK_GF180:
                if target not in GF180_MOS_MODELS:
                    problems.append(
                        f"'{tokens[0]}' uses model {target}, which gf180mcuD does not "
                        "define; use nfet_03v3 / pfet_03v3 (or the 6V0 variants).")
            elif not has_includes:
                problems.append(f"'{tokens[0]}' uses model {target}, which has no .model "
                                "card; add one (e.g. .model nmos1 nmos level=1).")
    # Keep the order, drop repeats (one bad model can be named on many lines).
    seen: list[str] = []
    for p in problems:
        if p not in seen:
            seen.append(p)
    return seen


def measure_names(text: str) -> list[str]:
    """Names declared by ``.meas`` lines, lower-cased, in order."""
    out: list[str] = []
    for m in re.finditer(r"^\s*\.?meas(?:ure)?\s+\w+\s+([A-Za-z_][\w.]*)", text or "",
                         re.IGNORECASE | re.MULTILINE):
        name = m.group(1).lower()
        if name not in out:
            out.append(name)
    return out
