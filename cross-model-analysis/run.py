"""Reproduce cross-family tables, publication figures, and source-data exports."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import importlib.metadata
import platform

from cross_model import summarize_behavioral, write_json
from mechanisms import summarize_mechanisms
from associations import run_associations

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("behavioral","mechanistic","all"),default="all")
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--mech-root",type=Path,default=None,
                   help="Root of family-specific mechanistic-interpretability output folders")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"output")
    p.add_argument("--skip-figures",action="store_true",help="Only regenerate the statistical tables")
    p.add_argument("--figures-only",action="store_true",help="Plot from existing statistical tables")
    p.add_argument("--figure-dpi",type=int,default=600)
    args=p.parse_args()
    if args.skip_figures and args.figures_only:
        p.error("--skip-figures and --figures-only are mutually exclusive")
    if args.figure_dpi<72 or args.figure_dpi>1200:
        p.error("--figure-dpi must be between 72 and 1200")
    root=args.repo_root.resolve()
    mech=(args.mech_root or root/"mechanistic-interpretability"/"outputs").resolve()
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=True)

    if not args.figures_only:
        if args.stage in ("behavioral","all"):
            summary=summarize_behavioral(root,dest)
            print("Behavioral features significant in all four families:",
                  summary["common_fdr_discoveries"])
        if args.stage in ("mechanistic","all"):
            summary=summarize_mechanisms(mech,dest)
            print("Mechanistic files present:",summary["available_files"])
        if args.stage=="all":
            print("Cross-stage analysis:",run_associations(dest))
    if not args.skip_figures:
        from figures import FigurePublisher, generate_behavioral, style
        from mechanistic_figures import generate_mechanistic
        import matplotlib
        publisher=FigurePublisher(dest,dpi=args.figure_dpi)
        with matplotlib.rc_context(style()):
            if args.stage in ("behavioral","all"):
                generate_behavioral(publisher,dest,root)
            if args.stage in ("mechanistic","all"):
                generate_mechanistic(publisher,dest)
        manifest={
            "scope":args.stage,
            "formats":["pdf","svg","png","csv"],
            "png_dpi":args.figure_dpi,
            "packages":{p:importlib.metadata.version(p)
                        for p in ("matplotlib","numpy")},
            "python":platform.python_version(),
            "figures":publisher.entries,
            "generated":sum("exports" in x for x in publisher.entries),
            "unavailable":sum(x.get("status")=="not_generated" for x in publisher.entries),
            "note":"Figures are descriptive unless the caption explicitly identifies upstream within-family confidence intervals or FDR.",
        }
        write_json(dest/"figures"/"manifest.json",manifest)
        print(f"Figures exported: {manifest['generated']}; unavailable: {manifest['unavailable']}")
        print("Figure manifest:",dest/"figures"/"manifest.json")
    from report import generate_report
    print("Research report:",generate_report(dest,args.stage))

if __name__=="__main__":
    main()
