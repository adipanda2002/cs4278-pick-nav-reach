"""Check a saved trial and write the complete audit report as JSON."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Saved trial directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON report path; existing files are preserved")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output file already exists; choose a new path")
    from evaluation.audit_i2 import audit
    report = audit(args.folder, Path(__file__).resolve().parent)
    # Bullet may write native diagnostics to stdout. Keep JSON in its own file.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Audit {}: {}".format(report["status"], args.output))


if __name__ == "__main__":
    main()
