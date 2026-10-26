"""Generate cross-model comparisons from saved evaluation and mechanism artifacts."""
import argparse
from pathlib import Path

from cross_model import summarize_behavioral
from mechanisms import summarize_mechanisms
from associations import run_associations

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("behavioral", "mechanistic", "all"), default="all")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--mech-root", type=Path, default=None,
                   help="Mechanistic outputs directory; defaults to mechanistic-interpretability/outputs.")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = p.parse_args()
    root = args.repo_root.resolve()
    mech = (args.mech_root or root / "mechanistic-interpretability" / "outputs").resolve()
    dest = args.output.resolve()
    if args.stage in ("behavioral", "all"):
        summary = summarize_behavioral(root, dest)
        print("Behavioral summary:", summary["common_fdr_discoveries"])
    if args.stage in ("mechanistic", "all"):
        summary = summarize_mechanisms(mech, dest)
        print("Mechanistic artifact availability:", summary["available_files"], "files")
        if not summary["available_files"]:
            print("No mechanistic outputs found; no mechanistic result is inferred.")
    if args.stage == "all":
        print("Cross-stage:", run_associations(dest))

if __name__ == "__main__":
    main()
