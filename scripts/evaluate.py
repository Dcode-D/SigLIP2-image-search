#!/usr/bin/env python3
"""Evaluate global or five-crop retrieval on the Flickr8k test split."""

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

from vision_retrieval.cache import (
    CacheMetadata,
    load_embedding_cache,
    read_cache_metadata,
)
from vision_retrieval.config import load_config
from vision_retrieval.data import load_split, manifest_sha256
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.metrics import (
    EvaluatedQuery,
    aggregate_metrics,
    evaluate_ranking,
    evaluate_score_threshold,
)
from vision_retrieval.predictions import (
    MethodMetadata,
    PredictionRecord,
    validate_predictions,
    write_method_metadata,
    write_predictions,
)
from vision_retrieval.retrieval import FiveCropRetriever, GlobalRetriever


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


def _load_retriever(config, method, encoder, images, queries):
    cache_path = config.output_root / "embeddings" / "test"
    actual = read_cache_metadata(cache_path)
    expected = CacheMetadata(
        split="test",
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=manifest_sha256(images, queries),
        crop_ratio=config.crop_ratio,
        crop_names=tuple(config.crop_names),
        image_ids=tuple(image.image_id for image in images),
        vector_dimension=actual.vector_dimension,
    )
    cache = load_embedding_cache(cache_path, expected)
    if method == "global":
        return GlobalRetriever(cache.metadata.image_ids, cache.global_embeddings), 0.0

    alpha_path = config.output_root / "parameters" / "five_crop_alpha.json"
    if not alpha_path.is_file():
        raise FileNotFoundError(
            f"selected alpha file not found: {alpha_path}; run scripts/tune_alpha.py first"
        )
    alpha_data = json.loads(alpha_path.read_text(encoding="utf-8"))
    validation_images, validation_queries = load_split(config, "validation")
    validation_hash = manifest_sha256(validation_images, validation_queries)
    if alpha_data.get("validation_manifest_sha256") != validation_hash:
        raise ValueError(
            "selected alpha was not tuned on the configured validation manifest"
        )
    if alpha_data.get("checkpoint_revision") != cache.metadata.checkpoint_revision:
        raise ValueError("selected alpha and test cache use different checkpoint revisions")
    alpha = float(alpha_data["alpha"])
    return (
        FiveCropRetriever(
            cache.metadata.image_ids,
            cache.global_embeddings,
            cache.crop_embeddings,
            cache.metadata.crop_names,
            alpha,
        ),
        alpha,
    )


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


def _write_reports(config, method: str, evaluated, report, threshold_report=None) -> None:
    report_dir = config.output_root / "reports"
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
    if threshold_report is not None:
        report_data["thresholded"] = asdict(threshold_report)
    temporary_json.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_json.replace(metrics_path)

    markdown_path = report_dir / f"{method}_metrics.md"
    markdown = (
        f"# {method.replace('_', ' ').title()} Results\n\n"
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
        markdown += (
            "\n## Validation-selected score threshold\n\n"
            f"Scores below `{threshold_report.threshold:.9f}` are excluded. "
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


def evaluate(config, method: str, device: str, timing_runs: int, warmup: int):
    images, queries = load_split(config, "test")
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=device,
        cache_dir=config.checkpoint_cache_dir,
    )
    retriever, alpha = _load_retriever(
        config, method, encoder, images, queries
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
    threshold_report = None
    threshold = None
    if method == "five_crop":
        threshold_path = config.output_root / "parameters" / "five_crop_threshold.json"
        if not threshold_path.is_file():
            raise FileNotFoundError(
                f"selected threshold not found: {threshold_path}; run scripts/tune_alpha.py first"
            )
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        validation_images, validation_queries = load_split(config, "validation")
        validation_hash = manifest_sha256(validation_images, validation_queries)
        if threshold_data.get("validation_manifest_sha256") != validation_hash:
            raise ValueError("selected threshold does not match the validation manifest")
        if threshold_data.get("checkpoint_revision") != encoder.checkpoint_revision:
            raise ValueError("selected threshold and evaluation use different checkpoints")
        if float(threshold_data.get("alpha")) != alpha:
            raise ValueError("selected threshold and retriever use different alpha values")
        threshold = float(threshold_data["threshold"])
        threshold_report = evaluate_score_threshold(
            [record.ranked_image_ids for record in records],
            [record.ranked_scores for record in records],
            [record.relevant_image_ids[0] for record in records],
            threshold,
        )

    predictions_dir = config.output_root / "predictions"
    prediction_path = predictions_dir / f"{method}.jsonl.gz"
    write_predictions(prediction_path, records)
    metadata = MethodMetadata(
        method_name=method,
        method_version="1",
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=manifest_sha256(images, queries),
        parameters={
            "alpha": alpha,
            "score_threshold": threshold,
            "threshold_selection_metric": (
                "validation_image_pair_f1" if threshold is not None else None
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
    _write_reports(config, method, evaluated, report, threshold_report)
    printed_report = asdict(report)
    if threshold_report is not None:
        printed_report["thresholded"] = asdict(threshold_report)
    print(json.dumps(printed_report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "five_crop.yaml",
    )
    parser.add_argument("--method", choices=("global", "five_crop"), required=True)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
