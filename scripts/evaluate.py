#!/usr/bin/env python3
"""Evaluate any registered retrieval method on a Flickr8k split.

Methods are discovered from ``vision_retrieval.methods``; run
``python scripts/evaluate.py --help`` to see the current list.

Two modes, selected with ``--split``:

* ``test`` (default): the benchmark. Writes the complete-ranking predictions and metrics
  and, when the method's validation-selected threshold file exists, the thresholded
  metrics next to them. ``--require-threshold`` turns a missing file into an error.
* ``validation``: writes the same artefacts under ``predictions/validation`` and
  ``reports/validation`` and, for a method that owns its threshold, selects the cutoff
  that keeps ``threshold_target_recall`` of the relevant-image scores and freezes it in
  ``artifacts/parameters/<method>_threshold.json``. Methods that reuse another method's
  threshold (FAISS -> global) only print what their own selection would have been.

Which threshold applies to which method is declared on the ``MethodSpec``
(``threshold_source``); see ``src/vision_retrieval/evaluation/thresholds.py``.
"""

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.data.cache import (
    CacheMetadata,
    load_embedding_cache,
    read_cache_metadata,
)
from vision_retrieval.config import load_config
from vision_retrieval.data.flickr8k import load_split, manifest_sha256
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.evaluation.metrics import (
    EvaluatedQuery,
    aggregate_metrics,
    evaluate_ranking,
    evaluate_score_threshold,
)
from vision_retrieval.evaluation.predictions import (
    MethodMetadata,
    PredictionRecord,
    validate_predictions,
    write_method_metadata,
    write_predictions,
)
from vision_retrieval.methods import BuildContext, build_method, list_methods
from vision_retrieval.evaluation.thresholds import (
    load_threshold,
    reachable_fraction,
    select_threshold_from_records,
    threshold_file_contents,
    threshold_path,
    write_threshold_file,
)

SPLITS = ("test", "validation")


def _synchronize(encoder) -> None:
    torch = encoder._torch
    if str(encoder.device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(encoder.device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_retriever(config, method, encoder, images, queries, split):
    cache_path = config.output_root / "embeddings" / split
    actual = read_cache_metadata(cache_path)
    expected = CacheMetadata(
        split=split,
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=manifest_sha256(images, queries),
        crop_ratio=config.crop_ratio,
        crop_names=tuple(config.crop_names),
        image_ids=tuple(image.image_id for image in images),
        vector_dimension=actual.vector_dimension,
    )
    cache = load_embedding_cache(cache_path, expected)
    # Every method (exact, five-crop, FAISS variants) is built through the registry so
    # this script never needs to know about a specific retriever class.
    return build_method(method, BuildContext(config, cache, encoder.checkpoint_revision))


def _timed_search(encoder, retriever, text: str, timing_runs: int):
    text_times = []
    search_times = []
    end_times = []
    results = None
    for _ in range(timing_runs):
        _synchronize(encoder)
        started = time.perf_counter_ns()
        query_vector = encoder.encode_texts([text])[0]
        _synchronize(encoder)
        encoded = time.perf_counter_ns()
        current_results = retriever.search(query_vector)
        _synchronize(encoder)
        finished = time.perf_counter_ns()
        if results is None:
            results = current_results
        elif [row.image_id for row in results] != [
            row.image_id for row in current_results
        ]:
            raise RuntimeError("retrieval ranking changed between timing runs")
        text_times.append((encoded - started) / 1_000_000.0)
        search_times.append((finished - encoded) / 1_000_000.0)
        end_times.append((finished - started) / 1_000_000.0)
    return (
        results,
        statistics.fmean(text_times),
        statistics.fmean(search_times),
        statistics.fmean(end_times),
    )


def _write_reports(
    report_dir: Path,
    method: str,
    split: str,
    evaluated,
    report,
    threshold_report=None,
    threshold_source=None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = report_dir / f"{method}_per_query.csv"
    temporary_csv = per_query_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "query_id",
            "relevant_rank",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "reciprocal_rank",
            "ndcg_at_10",
            "text_encoding_ms",
            "search_ms",
            "end_to_end_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluated:
            writer.writerow(
                {
                    "query_id": row.query_id,
                    **asdict(row.metrics),
                    "text_encoding_ms": row.text_encoding_ms,
                    "search_ms": row.search_ms,
                    "end_to_end_ms": row.end_to_end_ms,
                }
            )
    temporary_csv.replace(per_query_path)

    metrics_path = report_dir / f"{method}_metrics.json"
    temporary_json = metrics_path.with_suffix(".json.tmp")
    report_data = asdict(report)
    report_data["split"] = split
    if threshold_report is not None:
        report_data["thresholded"] = asdict(threshold_report)
        report_data["threshold_source_method"] = threshold_source
    temporary_json.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_json.replace(metrics_path)

    markdown_path = report_dir / f"{method}_metrics.md"
    markdown = (
        f"# {method.replace('_', ' ').title()} Results ({split} split)\n\n"
        "| Queries | R@1 | R@5 | R@10 | MRR | nDCG@10 | "
        "Mean latency (ms) | p50 (ms) | p95 (ms) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {report.query_count} | {report.recall_at_1:.6f} | "
        f"{report.recall_at_5:.6f} | {report.recall_at_10:.6f} | "
        f"{report.mrr:.6f} | {report.ndcg_at_10:.6f} | "
        f"{report.end_to_end.mean_ms:.3f} | {report.end_to_end.p50_ms:.3f} | "
        f"{report.end_to_end.p95_ms:.3f} |\n"
    )
    if threshold_report is not None:
        source_note = (
            "selected on this method's own validation scores"
            if threshold_source == method
            else f"selected on the validation scores of `{threshold_source}` (same score scale)"
        )
        markdown += (
            "\n## Validation-selected score threshold\n\n"
            f"Scores below `{threshold_report.threshold:.9f}` are excluded ({source_note}). "
            "The complete ranking above remains the primary benchmark result.\n\n"
            "| Candidate precision | Candidate recall | Candidate F1 | "
            "Retained fraction | Average results/query | Query coverage | "
            "Thresholded R@1 | R@5 | R@10 | MRR | nDCG@10 |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {threshold_report.candidate_precision:.6f} | "
            f"{threshold_report.candidate_recall:.6f} | "
            f"{threshold_report.candidate_f1:.6f} | "
            f"{threshold_report.retained_candidate_fraction:.6f} | "
            f"{threshold_report.average_results_per_query:.3f} | "
            f"{threshold_report.query_coverage:.6f} | "
            f"{threshold_report.recall_at_1:.6f} | "
            f"{threshold_report.recall_at_5:.6f} | "
            f"{threshold_report.recall_at_10:.6f} | "
            f"{threshold_report.mrr:.6f} | "
            f"{threshold_report.ndcg_at_10:.6f} |\n"
        )
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    temporary_markdown.write_text(markdown, encoding="utf-8")
    temporary_markdown.replace(markdown_path)


def _select_validation_threshold(config, built, records, images, encoder, split_hash):
    """Validation split: select (and freeze, for owners) the score threshold."""
    spec = built.spec
    gallery_ids = tuple(image.image_id for image in images)
    selection = select_threshold_from_records(records, gallery_ids, config.threshold_target_recall)
    reachable = reachable_fraction(records)
    print(
        f"{spec.name}: own validation selection would be threshold={selection.threshold:.6f} "
        f"(recall {selection.recall:.4f}, precision {selection.precision:.4f}, "
        f"{selection.retained_candidate_count / len(records):.2f} results/query); "
        f"relevant image reachable for {reachable:.2%} of queries",
        flush=True,
    )
    if spec.threshold_method != spec.name:
        print(
            f"{spec.name}: threshold_source is {spec.threshold_method!r}; nothing written. "
            f"Run `--method {spec.threshold_method} --split validation` to (re)select it.",
            flush=True,
        )
        return None
    if reachable < config.threshold_target_recall:
        print(
            f"{spec.name}: WARNING only {reachable:.2%} of relevant images are reachable, below the "
            f"{config.threshold_target_recall:.0%} target; the selected cutoff sits at the filler level",
            flush=True,
        )
    path = threshold_path(config, spec)
    contents = threshold_file_contents(
        selection,
        method_name=spec.name,
        method_parameters=built.parameters,
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        validation_manifest_sha256=split_hash,
        query_count=len(records),
        gallery_size=len(gallery_ids),
    )
    write_threshold_file(path, contents)
    print(f"{spec.name}: wrote {path}", flush=True)
    return path


def _apply_test_threshold(config, built, records, encoder, alpha, require_threshold):
    """Test split: load the frozen threshold for this method (own or source) and apply it."""
    spec = built.spec
    path = threshold_path(config, spec)
    validation_images, validation_queries = load_split(config, "validation")
    validation_hash = manifest_sha256(validation_images, validation_queries)
    try:
        threshold = load_threshold(
            path,
            validation_manifest_sha256=validation_hash,
            checkpoint_revision=encoder.checkpoint_revision,
            target_recall=config.threshold_target_recall,
            alpha=alpha,
        )
    except FileNotFoundError as exc:
        if require_threshold:
            raise
        print(f"{spec.name}: no thresholded metrics ({exc})", flush=True)
        return None, None, None
    report = evaluate_score_threshold(
        [record.ranked_image_ids for record in records],
        [record.ranked_scores for record in records],
        [record.relevant_image_ids[0] for record in records],
        threshold,
    )
    print(
        f"{spec.name}: applied threshold {threshold:.6f} from {path.name} "
        f"(source method {spec.threshold_method!r})",
        flush=True,
    )
    return threshold, report, path


def evaluate(
    config,
    method: str,
    device: str,
    timing_runs: int,
    warmup: int,
    split: str = "test",
    require_threshold: bool = False,
):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    images, queries = load_split(config, split)
    split_hash = manifest_sha256(images, queries)
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=device,
        cache_dir=config.checkpoint_cache_dir,
    )
    built = _load_retriever(config, method, encoder, images, queries, split)
    retriever = built.retriever
    alpha = float(built.parameters.get("alpha", 0.0))
    print(
        f"{method}: built {built.spec.label!r} in {built.build_seconds:.3f}s "
        f"(family={built.spec.family}, split={split})",
        flush=True,
    )

    for query in queries[: min(warmup, len(queries))]:
        query_vector = encoder.encode_texts([query.text])[0]
        retriever.search(query_vector)
    _synchronize(encoder)

    records = []
    evaluated = []
    for index, query in enumerate(queries, start=1):
        results, text_ms, search_ms, end_ms = _timed_search(
            encoder, retriever, query.text, timing_runs
        )
        record = PredictionRecord(
            query_id=query.query_id,
            relevant_image_ids=(query.relevant_image_id,),
            ranked_image_ids=tuple(result.image_id for result in results),
            ranked_scores=tuple(result.score for result in results),
            text_encoding_ms=text_ms,
            search_ms=search_ms,
            end_to_end_ms=end_ms,
        )
        records.append(record)
        evaluated.append(
            EvaluatedQuery(
                query_id=query.query_id,
                metrics=evaluate_ranking(
                    record.ranked_image_ids, set(record.relevant_image_ids)
                ),
                text_encoding_ms=text_ms,
                search_ms=search_ms,
                end_to_end_ms=end_ms,
            )
        )
        if index % 100 == 0 or index == len(queries):
            print(f"{method}: evaluated {index}/{len(queries)} queries", flush=True)

    expected_relevance = {
        query.query_id: (query.relevant_image_id,) for query in queries
    }
    gallery_ids = tuple(image.image_id for image in images)
    validate_predictions(records, expected_relevance, gallery_ids)
    report = aggregate_metrics(evaluated)

    threshold = None
    threshold_report = None
    threshold_file = None
    if split == "validation":
        # Thresholds are *selected* here and never applied here.
        threshold_file = _select_validation_threshold(
            config, built, records, images, encoder, split_hash
        )
        predictions_dir = config.output_root / "predictions" / "validation"
        report_dir = config.output_root / "reports" / "validation"
    else:
        threshold, threshold_report, threshold_file = _apply_test_threshold(
            config, built, records, encoder, alpha, require_threshold
        )
        predictions_dir = config.output_root / "predictions"
        report_dir = config.output_root / "reports"

    prediction_path = predictions_dir / f"{method}.jsonl.gz"
    write_predictions(prediction_path, records)
    metadata = MethodMetadata(
        method_name=method,
        method_version="1",
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=split_hash,
        parameters={
            **built.parameters,
            "split": split,
            "method_family": built.spec.family,
            "method_label": built.spec.label,
            "method_build_seconds": built.build_seconds,
            "alpha": alpha,
            "score_threshold": threshold,
            "threshold_source_method": built.spec.threshold_method,
            "threshold_file": str(threshold_file) if threshold_file else None,
            "threshold_selection_metric": (
                "validation_target_relevant_recall" if threshold is not None else None
            ),
            "threshold_target_recall": (
                config.threshold_target_recall if threshold is not None else None
            ),
            "crop_ratio": config.crop_ratio,
            "timing_runs": timing_runs,
            "warmup_queries": min(warmup, len(queries)),
            "batch_size": 1,
        },
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(encoder.device),
        },
        software={
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    )
    write_method_metadata(predictions_dir / f"{method}.metadata.json", metadata)
    _write_reports(
        report_dir,
        method,
        split,
        evaluated,
        report,
        threshold_report,
        built.spec.threshold_method,
    )
    printed_report = asdict(report)
    if threshold_report is not None:
        printed_report["thresholded"] = asdict(threshold_report)
    print(json.dumps(printed_report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument(
        "--method",
        choices=list_methods(),
        required=True,
        help="registered retrieval method; see src/vision_retrieval/methods/",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="test = benchmark (applies the frozen threshold); validation = select the threshold",
    )
    parser.add_argument(
        "--require-threshold",
        action="store_true",
        help="test split only: fail instead of skipping when the threshold file is missing",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    if args.timing_runs <= 0:
        parser.error("--timing-runs must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    evaluate(
        load_config(args.config),
        args.method,
        args.device,
        args.timing_runs,
        args.warmup,
        split=args.split,
        require_threshold=args.require_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
