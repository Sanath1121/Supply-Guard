#!/usr/bin/env python3
"""SupplyGuard Master Phase-Gated Test Runner.

Usage:
  python run_phase_tests.py --phase 0         # Run Gate 0 test
  python run_phase_tests.py --phase 2         # Run Gate 2 test
  python run_phase_tests.py --up-to 3         # Run Gates 0 through 3
  python run_phase_tests.py --all             # Run all Gates (0 to 8)
"""
import os
import sys
import argparse
import unittest
import time

# Ensure both current dir and parent repo root are in sys.path
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [PARENT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

PHASE_MAP = {
    0: ("Gate 0: Environment, Setup & Data Acquisition", "tests.test_phase0_setup"),
    1: ("Gate 1: EDA, ACF, Granger & Horizon Gating", "tests.test_phase1_eda"),
    2: ("Gate 2: Data Pipeline Hardening & Segmentation", "tests.test_phase2_dataset"),
    3: ("Gate 3: Models, Graph Convolutions & Baselines", "tests.test_phase3_models"),
    4: ("Gate 4: Training Pipeline & Colab Checkpoints", "tests.test_phase4_training"),
    5: ("Gate 5: Evaluation Pipeline, Metrics & Terciles", "tests.test_phase5_evaluation"),
    6: ("Gate 6: Explainability & Deletion Testing", "tests.test_phase6_explainability"),
    7: ("Gate 7: Streamlit Dashboard Components", "tests.test_phase7_app"),
    8: ("Gate 8: End-to-End Pipeline & Claims Audit", "tests.test_phase8_e2e"),
}


def run_phase_test(phase_num: int) -> bool:
    """Run tests for a single phase gate."""
    if phase_num not in PHASE_MAP:
        print(f"[ERROR] Invalid phase {phase_num}. Available: 0 to 8.")
        return False

    name, module_name = PHASE_MAP[phase_num]
    print("\n" + "=" * 70)
    print(f"RUNNING {name.upper()}")
    print("=" * 70)

    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception:
        short_name = module_name.split(".")[-1]
        try:
            suite = loader.loadTestsFromName(short_name)
        except Exception as e:
            print(f"[ERROR] Failed to load {module_name}: {e}")
            return False

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("-" * 70)
    if result.wasSuccessful():
        print(f"--> GATE {phase_num} [PASSED] in {elapsed:.2f}s ({result.testsRun} checks passed)")
        return True
    else:
        print(f"--> GATE {phase_num} [BLOCKED] in {elapsed:.2f}s ({len(result.failures)} failures, {len(result.errors)} errors)")
        return False


def main():
    parser = argparse.ArgumentParser(description="SupplyGuard Plan A Gate Verification Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=range(0, 9), help="Phase number to verify (0-8)")
    group.add_argument("--up-to", type=int, choices=range(0, 9), help="Verify all phases from 0 up to N")
    group.add_argument("--all", action="store_true", help="Verify all phases (0 to 8)")

    args = parser.parse_args()

    if args.phase is not None:
        target_phases = [args.phase]
    elif args.up_to is not None:
        target_phases = list(range(0, args.up_to + 1))
    elif args.all:
        target_phases = list(range(0, 9))

    print("\n+----------------------------------------------------------------+")
    print("|          SUPPLYGUARD PLAN A GATE VERIFICATION HARNESS          |")
    print("+----------------------------------------------------------------+")
    print(f"Executing checks for Phase(s): {target_phases}")

    results = {}
    for p in target_phases:
        success = run_phase_test(p)
        results[p] = success
        if not success:
            print(f"\n[HALT] Phase {p} did not pass. Halting further gate checks.")
            break

    # Summary
    print("\n" + "=" * 70)
    print("GATE VERIFICATION SUMMARY SCORECARD")
    print("=" * 70)
    all_passed = True
    for p in target_phases:
        status = "[PASSED]" if results.get(p, False) else "[FAILED/BLOCKED]"
        name, _ = PHASE_MAP[p]
        print(f"Phase {p:02d} | {name:<48} | {status}")
        if not results.get(p, False):
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL CHECKED PHASE GATES PASSED! READY TO PROCEED.")
        sys.exit(0)
    else:
        print("RESULT: ONE OR MORE GATES BLOCKED. REVIEW FAILURES BEFORE ADVANCING.")
        sys.exit(1)


if __name__ == "__main__":
    main()
