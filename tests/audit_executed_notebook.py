"""Audit Executed Notebook Script.

Empirically verifies:
1. notebooks/01_EDA.ipynb exists and executed cleanly.
2. Every code cell has a non-null execution_count.
3. Zero code cells contain an error output (no ename, evalue, or traceback).
4. Final cell is Markdown and contains all required Gate 1 decision elements:
   - HORIZON_MIN = 10
   - REFRAME_RQ2
   - Non-random / serial structure
   - Phase 1 GO / APPROVED verdict.
"""

import json
import sys
from pathlib import Path

def audit_notebook(nb_path_str: str):
    nb_path = Path(nb_path_str)
    if not nb_path.exists():
        print(f"FAILED: Notebook {nb_path} does not exist.")
        sys.exit(1)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Loaded {nb_path_str}")
    print(f"Total cells: {len(cells)}")

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    print(f"Code cells: {len(code_cells)}, Markdown cells: {len(markdown_cells)}")

    errors = []
    audit_records = []

    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        if ctype == "code":
            exec_count = cell.get("execution_count")
            outputs = cell.get("outputs", [])

            if exec_count is None:
                errors.append(f"Cell index {idx}: execution_count is None")

            cell_errs = []
            for out in outputs:
                if out.get("output_type") == "error" or "ename" in out or "traceback" in out:
                    cell_errs.append({
                        "ename": out.get("ename"),
                        "evalue": out.get("evalue"),
                        "traceback": out.get("traceback")
                    })

            if cell_errs:
                errors.append(f"Cell index {idx} encountered runtime errors: {cell_errs}")

            audit_records.append({
                "cell_index": idx,
                "execution_count": exec_count,
                "outputs_count": len(outputs),
                "output_types": [o.get("output_type") for o in outputs],
                "has_error": len(cell_errs) > 0
            })

    print("\n--- Code Cell Execution Breakdown ---")
    for r in audit_records:
        print(f"Cell {r['cell_index']:2d} | ExecCount: {r['execution_count']} | Outputs: {r['outputs_count']:2d} | Types: {r['output_types']} | Error: {r['has_error']}")

    if errors:
        print(f"\nAUDIT FAILED: {len(errors)} errors found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n--> CODE CELL AUDIT PASSED: All code cells executed with zero exceptions/errors.")

    print("\n--- Detailed Cell Outputs Inspection ---")
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        if ctype == "code":
            ec = cell.get("execution_count")
            outputs = cell.get("outputs", [])
            print(f"\n[Cell {idx} (code, execution_count={ec})]")
            for o_idx, out in enumerate(outputs):
                otype = out.get("output_type")
                if otype == "stream":
                    lines = out.get("text", [])
                    print(f"  Out {o_idx} [stream/{out.get('name')}]: {len(lines)} lines")
                    if lines:
                        print(f"    Start: {lines[0].strip()[:80]}")
                        print(f"    End:   {lines[-1].strip()[:80]}")
                elif otype in ("display_data", "execute_result"):
                    data = out.get("data", {})
                    keys = list(data.keys())
                    print(f"  Out {o_idx} [{otype}]: mime-types: {keys}")
        else:
            first = cell.get("source", [""])[0].strip()[:80]
            print(f"\n[Cell {idx} (markdown)] Header: {first}")

    # Audit the final cell
    last_cell = cells[-1]
    if last_cell.get("cell_type") != "markdown":
        print(f"AUDIT FAILED: Expected last cell to be markdown, got {last_cell.get('cell_type')}")
        sys.exit(1)

    last_content = "".join(last_cell.get("source", []))
    print("\n--- Final Cell Gate 1 Decision Table Audit ---")

    checks = {
        "HORIZON_MIN = 10": "HORIZON_MIN = 10" in last_content or "HORIZON_MIN = 10" in last_content.replace("`", ""),
        "REFRAME_RQ2": "REFRAME_RQ2" in last_content,
        "Non-random / serial structure": any(kw in last_content for kw in ["Bartlett", "ACF lag-1", "autoregressive", "non-stationary", "serial"]),
        "GO / APPROVED verdict": "GO / APPROVED" in last_content or "APPROVED" in last_content
    }

    all_passed = True
    for requirement, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  - [{status}] Requirement: '{requirement}'")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\nAUDIT FAILED: Final cell decision table is missing one or more required components.")
        sys.exit(1)
    else:
        print("\n--> DECISION TABLE AUDIT PASSED: All required decision elements verified.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_EDA.ipynb"
    audit_notebook(target)

