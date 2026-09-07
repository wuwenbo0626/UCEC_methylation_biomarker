#!/usr/bin/env python3
"""Run the UCEC methylation biomarker workflow.

This lightweight orchestrator intentionally uses only the Python standard
library. It calls the numbered scripts in order, prints a clear execution plan,
supports resume/dry-run modes, and avoids hiding the underlying commands.

Examples
--------
Preview the core pipeline:

    python scripts/run_pipeline.py --dry-run

Run the core discovery workflow from download through final evaluation:

    python scripts/run_pipeline.py --stages core

Run only downstream steps after preprocessing:

    python scripts/run_pipeline.py --from differential

Run core workflow plus selected supplemental analyses:

    python scripts/run_pipeline.py --stages core literature_panel external_validation
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class Step:
    """A runnable pipeline step."""

    name: str
    script: str
    description: str
    outputs: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    accepts_config: bool = True
    is_heavy: bool = False
    needs_network: bool = False

    @property
    def script_path(self) -> Path:
        return ROOT / "scripts" / self.script

    def output_paths(self) -> tuple[Path, ...]:
        return tuple(ROOT / item for item in self.outputs)


STEPS: dict[str, Step] = {
    "download": Step(
        name="download",
        script="01_download_data.py",
        description="Download/query TCGA-UCEC 450K methylation data and clinical metadata.",
        outputs=(
            "data/processed/gdc_hm450_file_metadata.parquet",
            "data/processed/ucec_clinical.parquet",
            "data/processed/ucec_hm450_beta_samples_by_probes.npy",
            "data/processed/hm450_probe_metadata.parquet",
        ),
        needs_network=True,
        is_heavy=True,
    ),
    "preprocess": Step(
        name="preprocess",
        script="02_preprocess.py",
        description="Clean beta matrix, filter probes/samples, and impute remaining missing values.",
        outputs=(
            "data/processed/ucec_hm450_beta_clean_probes_by_samples.npy",
            "data/processed/ucec_hm450_clean_sample_groups.parquet",
            "data/processed/ucec_hm450_clean_probe_metadata.parquet",
        ),
        is_heavy=True,
    ),
    "differential": Step(
        name="differential",
        script="03_differential_analysis.py",
        description="Run tumor-vs-normal differential methylation analysis.",
        outputs=(
            "data/results/differential_methylation_all.parquet",
            "data/results/differential_methylation_significant.parquet",
        ),
        is_heavy=True,
    ),
    "annotate": Step(
        name="annotate",
        script="04_annotate_probes.py",
        description="Annotate significant 450K probes to genes and promoter regions.",
        outputs=(
            "data/results/dmp_annotated_to_genes.parquet",
            "data/results/dmp_promoter_TSS200_TSS1500.parquet",
        ),
    ),
    "enrichment": Step(
        name="enrichment",
        script="05_enrichment_analysis.py",
        description="Create candidate-gene tables and run GO/KEGG enrichment.",
        outputs=(
            "data/results/promoter_hypermethylated_probe_gene_pairs.parquet",
            "data/results/candidate_genes_dedup.parquet",
            "outputs/tables/known_marker_overlaps_with_candidate_genes.parquet",
        ),
        needs_network=True,
    ),
    "feature_selection": Step(
        name="feature_selection",
        script="06_feature_selection.py",
        description="Run LASSO feature selection and create a 10-marker data-driven panel.",
        outputs=("outputs/tables/final_10_methylation_markers.parquet",),
        is_heavy=True,
    ),
    "train_model": Step(
        name="train_model",
        script="07_train_model.py",
        description="Train and compare core classification models.",
        outputs=(
            "outputs/tables/model_test_metrics.parquet",
            "outputs/tables/model_5fold_cv_auc_summary.parquet",
        ),
    ),
    "evaluate_model": Step(
        name="evaluate_model",
        script="08_evaluate_model.py",
        description="Evaluate final 10-marker model against PAX1/JAM3 baseline and export main figures/tables.",
        outputs=(
            "outputs/tables/key_results_summary.parquet",
            "outputs/tables/delong_auc_test_10marker_vs_pax1_jam3.parquet",
            "outputs/figures/fig2_volcano_plot.png",
            "outputs/figures/fig4_roc_10marker_vs_baseline.png",
        ),
    ),
    "batch_effects": Step(
        name="batch_effects",
        script="09_assess_batch_effects_combat.py",
        description="Supplemental PCA and ComBat batch-effect sensitivity analysis.",
        outputs=(),
        accepts_config=False,
        is_heavy=True,
    ),
    "nested_cv": Step(
        name="nested_cv",
        script="10_nested_cv_lasso_lr_promoter_hyper.py",
        description="Supplemental nested CV using precomputed promoter hypermethylated probes.",
        outputs=(),
        accepts_config=False,
        is_heavy=True,
    ),
    "literature_panel": Step(
        name="literature_panel",
        script="11_literature_constrained_10gene_panel.py",
        description="Build and evaluate the literature-constrained plus data-driven 10-gene panel.",
        outputs=(),
        accepts_config=False,
        is_heavy=True,
    ),
    "fully_nested_cv": Step(
        name="fully_nested_cv",
        script="12_fully_nested_cv_dmp_lasso_lr.py",
        description="Fully nested DMP + LASSO + logistic regression validation.",
        outputs=(),
        accepts_config=False,
        is_heavy=True,
    ),
    "external_validation": Step(
        name="external_validation",
        script="13_external_validate_geo_gse155760.py",
        description="Validate the TCGA-trained panel on GEO GSE155760.",
        outputs=(),
        accepts_config=False,
        needs_network=True,
        is_heavy=True,
    ),
}


GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "download",
        "preprocess",
        "differential",
        "annotate",
        "enrichment",
        "feature_selection",
        "train_model",
        "evaluate_model",
    ),
    "supplemental": (
        "batch_effects",
        "nested_cv",
        "literature_panel",
        "fully_nested_cv",
        "external_validation",
    ),
    "validation": (
        "literature_panel",
        "fully_nested_cv",
        "external_validation",
    ),
    "all": tuple(STEPS.keys()),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def expand_stage_tokens(tokens: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        if token in GROUPS:
            expanded.extend(GROUPS[token])
        elif token in STEPS:
            expanded.append(token)
        else:
            known = ", ".join(sorted(set(STEPS) | set(GROUPS)))
            raise SystemExit(f"Unknown stage/group '{token}'. Available: {known}")
    return unique_preserve_order(expanded)


def slice_steps(step_names: list[str], start: str | None, end: str | None) -> list[str]:
    if start is not None:
        if start not in step_names:
            raise SystemExit(f"--from stage '{start}' is not present in the selected stages.")
        step_names = step_names[step_names.index(start) :]
    if end is not None:
        if end not in step_names:
            raise SystemExit(f"--until stage '{end}' is not present in the selected stages.")
        step_names = step_names[: step_names.index(end) + 1]
    return step_names


def output_status(step: Step) -> str:
    outputs = step.output_paths()
    if not outputs:
        return "no declared outputs"
    existing = sum(path.exists() for path in outputs)
    if existing == len(outputs):
        return "complete"
    if existing:
        return f"partial ({existing}/{len(outputs)} outputs exist)"
    return "not started"


def should_skip(step: Step, resume: bool, force: bool) -> bool:
    outputs = step.output_paths()
    return resume and not force and bool(outputs) and all(path.exists() for path in outputs)


def command_for_step(step: Step, args: argparse.Namespace) -> list[str]:
    command = [str(args.python), str(step.script_path)]
    if step.accepts_config:
        command.extend(["--config", str(args.config)])
    command.extend(step.extra_args)
    if step.name == "download" and args.skip_download:
        command.append("--skip-download")
    return command


def write_run_report(
    report_path: Path,
    selected_steps: list[str],
    results: list[dict[str, object]],
    started_at: float,
    args: argparse.Namespace,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "UCEC_methylation_biomarker",
        "started_at_epoch": started_at,
        "finished_at_epoch": time.time(),
        "duration_seconds": round(time.time() - started_at, 3),
        "selected_steps": selected_steps,
        "config": str(args.config),
        "python": str(args.python),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "skip_download": bool(args.skip_download),
        "results": results,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible UCEC methylation biomarker pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["core"],
        help=(
            "Pipeline stages or groups to run. Groups: core, validation, supplemental, all. "
            "Individual stages include download, preprocess, differential, annotate, "
            "enrichment, feature_selection, train_model, evaluate_model, batch_effects, "
            "nested_cv, literature_panel, fully_nested_cv, external_validation."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter used to run scripts.")
    parser.add_argument("--from", dest="start", choices=list(STEPS), default=None, help="Start at this stage.")
    parser.add_argument("--until", dest="end", choices=list(STEPS), default=None, help="Stop after this stage.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip stages whose declared outputs already exist.")
    parser.add_argument("--force", action="store_true", help="Run selected stages even when declared outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-download", action="store_true", help="Pass --skip-download to 01_download_data.py.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed stage instead of stopping immediately.")
    parser.add_argument(
        "--run-report",
        type=Path,
        default=ROOT / "outputs" / "reports" / "pipeline_run_report.json",
        help="Where to write the pipeline run report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.python = args.python.resolve()
    selected = slice_steps(expand_stage_tokens(args.stages), args.start, args.end)

    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    log(f"Repository root: {ROOT}")
    log(f"Selected stages: {', '.join(selected)}")
    if any(STEPS[name].is_heavy for name in selected):
        log("Note: selected stages include memory/time-heavy steps.")
    if any(STEPS[name].needs_network for name in selected):
        log("Note: selected stages include network-dependent downloads/API calls.")

    log("Execution plan:")
    for i, name in enumerate(selected, start=1):
        step = STEPS[name]
        command = " ".join(shlex.quote(part) for part in command_for_step(step, args))
        log(f"  {i:02d}. {name}: {step.description} [{output_status(step)}]")
        log(f"      {command}")

    started_at = time.time()
    results: list[dict[str, object]] = []
    if args.dry_run:
        write_run_report(args.run_report, selected, results, started_at, args)
        log(f"Dry run complete. Report written to {args.run_report}")
        return 0

    os.chdir(ROOT)
    for name in selected:
        step = STEPS[name]
        if should_skip(step, args.resume, args.force):
            log(f"Skipping {name}; declared outputs already exist. Use --force to rerun.")
            results.append({"stage": name, "status": "skipped", "duration_seconds": 0})
            continue

        command = command_for_step(step, args)
        log(f"Starting {name}")
        t0 = time.time()
        completed = subprocess.run(command, cwd=ROOT)
        duration = round(time.time() - t0, 3)
        status = "completed" if completed.returncode == 0 else "failed"
        results.append(
            {
                "stage": name,
                "status": status,
                "returncode": completed.returncode,
                "duration_seconds": duration,
            }
        )
        log(f"Finished {name} with status={status}, returncode={completed.returncode}, duration={duration}s")
        write_run_report(args.run_report, selected, results, started_at, args)
        if completed.returncode != 0 and not args.keep_going:
            log(f"Stopping after failed stage: {name}")
            return completed.returncode

    write_run_report(args.run_report, selected, results, started_at, args)
    log(f"Pipeline finished. Report written to {args.run_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

