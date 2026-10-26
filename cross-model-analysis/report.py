"""Generate a concise, traceable results manuscript from numerical outputs."""
from __future__ import annotations
import json
from pathlib import Path
from cross_model import FAMILIES, CROSS_LANGUAGE, read_csv, number

def _cell(x,d=3):
    v=number(x)
    return "NA" if v is None else f"{v:+.{d}f}"
def generate_report(root:Path,stage="all"):
    lines=["# Cross-Model Results","",
           "## Scope and interpretation","",
           "Comparisons cover four model families evaluated on the same animal-welfare scenario design. "
           "The primary estimand is feature-adapter minus registered language-matched control "
           "on the 0–4 target-value-support scale. The upstream analysis provides clustered "
           "confidence intervals and per-family FDR-adjusted q-values. No independent-family "
           "meta-analytic inference or seed-level replication is assumed.",""]
    beh=root/"behavioral"
    if stage in ("all","behavioral") and (beh/"feature_replication.csv").exists():
        features=read_csv(beh/"feature_replication.csv")
        sig=[r["feature"] for r in features if r["same_sign_and_fdr_all_four"]=="True"]
        lines+=["## Behavioral replication","",
                f"**All-family sign-consistent, BH-FDR-positive features:** "
                +((", ".join(f.replace("_"," ") for f in sig)) if sig else "none")+".","",
                "| Linguistic feature | Gemma 2 | Llama 3.1 | Mistral | Qwen 3 | FDR-positive families |",
                "|---|---:|---:|---:|---:|---:|"]
        for r in features:
            desc=r["feature"].replace("_"," ")+(" †" if r["feature"] in CROSS_LANGUAGE else "")
            lines.append("| "+" | ".join([desc]+[_cell(r.get(m+"_effect")) for m in FAMILIES]
                                        +[r["fdr_significant_families"]])+" |")
        lines+=["","† Language-changing contrast; the grammatical intervention is confounded "
                "with a change in training language. Effects are not welfare opposition, "
                "but changes relative to their appropriate fine-tuning controls.","",
                "### Cross-family effect agreement","",
                "| Feature set | First family | Second family | Pearson r | Spearman ρ | Sign agreement |",
                "|---|---|---|---:|---:|---:|"]
        for r in read_csv(beh/"cross_family_correlations.csv"):
            lines.append(f"| {r['scope']} | {r['first']} | {r['second']} | "
                         f"{_cell(r['pearson'])} | {_cell(r['spearman'])} | "
                         f"{number(r['sign_agreement']):.2f} |")
        lines+=["","### Baseline gains","",
                "| Model | Contrast | Weighted mean gain |",
                "|---|---|---:|"]
        for r in read_csv(beh/"baseline_gains.csv"):
            lines.append(f"| {r['model']} | {r['contrast']} | {_cell(r['weighted_category_mean'])} |")
        lines+=["","### Robustness and heterogeneity","",
                "Original category-specific, held-out-dimension, tradeoff, coherence, "
                "judge-reliability, and sensitivity tables are retained as source-data exports. "
                "Feature-level lexical correlations are exploratory and do not establish mediation.",""]
    if stage in ("all","mechanistic"):
        manifest=root/"mechanistic"/"summary.json"
        lines+=["## Mechanistic comparison",""]
        if manifest.exists():
            x=json.loads(manifest.read_text(encoding="utf-8"))
            if x["available_files"]:
                lines.append(f"Analyzed {x['available_files']} mechanistic scalar files "
                             f"and {x['normalized_measurements']} individual scalar records. "
                             "Representational, probe, LoRA, and steering comparisons are descriptive "
                             "unless the original experiment provides an independently judged held-out contrast.")
            else:
                lines.append("Mechanistic scalar results are not available in the selected output directory. "
                             "No mechanistic conclusions are reported.")
        else:lines.append("Mechanistic analysis has not been generated.")
        lines+=[""]
    f=root/"figures"/"manifest.json"
    if f.exists():
        manifest=json.loads(f.read_text(encoding="utf-8"))
        lines+=["## Figure index",""]
        for item in manifest["figures"]:
            if item.get("exports"):
                path=item["exports"][0]
                lines.append(f"- **{item['figure']}** — {item['caption']} "
                             f"([vector PDF]({path}), [source data]({item['exports'][-1]}))")
            else:
                lines.append(f"- **{item['figure']}** — unavailable: {item['reason']}.")
        lines+=[""]
    lines+=["## Limitations","",
            "Shared benchmark and fine-tuning infrastructure limit the independence of model replications. "
            "Only the sampled model architectures and training seeds are covered. Language-changing "
            "contrasts cannot isolate syntax or morphology from language identity. "
            "Agreement among automated judges is not equivalent to human validation. "
            "Representation similarity and probe performance are descriptive; steering is an "
            "intervention-specific test, not proof of a unique causal mechanism. "
            "Cross-stage correlations are exploratory and potentially sensitive to "
            "feature choice, module aggregation, and response quality.","",
            "## Reproducibility","",
            "The per-family source files, original uncertainty estimates, "
            "figure source-data CSVs, and figure manifest are the provenance record. "
            "Plots never introduce inferential statistics beyond the source analysis.",""]
    out=root/"research_results.md"
    out.write_text("\n".join(lines),encoding="utf-8")
    return out
