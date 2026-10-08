#!/usr/bin/env python
"""Gate A (DSB spec §6): read the E2a HV-controls contrasts and freeze the DSB base arm.

    python scripts/dsb_gate_a.py --hv-numbers docs/results/hv_threshold_controls_20261007_verified.json \
        --out runs/analysis/dsb_gates_20261007/gate_a.json

Rules (frozen in the spec): with r = Dead(x1_hv8) - Dead(x1_hv30) and R = Dead(x2_hv30) -
Dead(x1_hv30) on the validation fold: R <= 0 -> base hv30, proceed; r >= 0.9*R AND the x1_hv8
bPQ tax >= -0.001 -> downgrade (HV support explains the complementarity; architecture premise
lost, back to the user); r >= 0.5*R -> base hv8, proceed; else base hv30, proceed.
"""
import argparse
import json
from pathlib import Path

SWITCH = 0.5
DOWNGRADE = 0.9
TAX_GUARD = -0.001
ARMS = ("x1_hv30", "x1_hv8", "x2_hv30")


def gate_a(numbers: dict) -> dict:
    dead = {a: float(numbers["arms"][a]["mean"]["Dead PQ"]) for a in ARMS}
    bpq = {a: float(numbers["arms"][a]["mean"]["bPQ"]) for a in ARMS}
    r, big = dead["x1_hv8"] - dead["x1_hv30"], dead["x2_hv30"] - dead["x1_hv30"]
    out = {"r_x1_hv8_minus_x1_hv30": r, "R_x2_hv30_minus_x1_hv30": big,
           "ratio": r / big if big > 0 else None,
           "bpq_tax_x1_hv8": bpq["x1_hv8"] - bpq["x1_hv30"]}
    if big <= 0:
        out.update({"verdict": "proceed", "base": "hv30",
                    "reason": "R<=0: HV support is not the complementarity driver"})
    elif r >= DOWNGRADE * big and out["bpq_tax_x1_hv8"] >= TAX_GUARD:
        out.update({"verdict": "downgrade", "base": None,
                    "reason": "x1 cutoff-lowering explains >=90% of the x2 Dead gain without tax"})
    else:
        out.update({"verdict": "proceed", "base": "hv8" if r >= SWITCH * big else "hv30",
                    "reason": f"support cutoff reproduces {out['ratio']:.1%} of the x2 Dead gain"})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hv-numbers", type=Path, required=True,
                   help="hv_control_numbers output JSON (arms.<arm>.mean endpoints)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    numbers = json.loads(a.hv_numbers.read_text())
    result = {"hv_numbers": str(a.hv_numbers), **gate_a(numbers)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
