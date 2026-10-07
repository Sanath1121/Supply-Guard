"""SupplyGuard Checkpoint SHA-256 Manifest Generator (NEW-2).

Generates outputs/models/checkpoints.sha256 for all trained model weights,
scaler, and evaluation loss curves to ensure cryptographic reproducibility.

Usage:
    python scripts/generate_model_manifest.py [--verify]
"""
import argparse
import hashlib
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "checkpoints.sha256")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    """Scan outputs/models and outputs/results and write cryptographic manifest."""
    files_to_hash = []

    # Checkpoints and scaler
    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    if os.path.exists(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.endswith(".pt") or fname.endswith(".joblib"):
                rel_path = os.path.join("outputs", "models", fname).replace("\\", "/")
                files_to_hash.append((rel_path, os.path.join(models_dir, fname)))

    # Result CSVs
    results_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    if os.path.exists(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".csv"):
                rel_path = os.path.join("outputs", "results", fname).replace("\\", "/")
                files_to_hash.append((rel_path, os.path.join(results_dir, fname)))

    lines = []
    for rel_path, full_path in files_to_hash:
        sha = compute_sha256(full_path)
        lines.append(f"{sha}  {rel_path}\n")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[MANIFEST] Successfully generated {MANIFEST_PATH} with {len(lines)} file hashes.")
    return True


def verify_manifest() -> bool:
    """Verify all files listed in checkpoints.sha256."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        return False

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        entries = [line.strip().split(maxsplit=1) for line in f if line.strip()]

    all_matched = True
    print(f"[VERIFY] Verifying {len(entries)} cryptographic hashes...")
    for expected_hash, rel_path in entries:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [MISSING] {rel_path}")
            all_matched = False
            continue
        actual_hash = compute_sha256(full_path)
        if actual_hash == expected_hash:
            print(f"  [OK] {rel_path}")
        else:
            print(f"  [CORRUPTED] {rel_path} (expected {expected_hash[:8]}..., got {actual_hash[:8]}...)")
            all_matched = False

    if all_matched:
        print("\n[VERIFIED] ALL ARTIFACTS MATCH CRYPTOGRAPHIC MANIFEST!")
    else:
        print("\n[FAILURE] One or more artifacts failed cryptographic verification!")

    return all_matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptographic Manifest Tool")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifest against disk")
    args = parser.parse_args()

    if args.verify:
        success = verify_manifest()
        sys.exit(0 if success else 1)
    else:
        generate_manifest()
        sys.exit(0)
