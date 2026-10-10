# Flickr8k Shared Dataset, Metrics, and Timing Guide

This is the single guide to give every participant. Each project may use a different retrieval method, but everyone must use the dataset, ground truth, prediction contract, timing protocol, and evaluator defined here.

Share these four files together:

```text
docs/SHARED_BENCHMARK_GUIDE.md
scripts/score_predictions.py
scripts/select_threshold.py
scripts/time_operation.py
```

## What everyone must keep identical

| Item | Fixed value |
|---|---|
| Dataset | Flickr8k mirror `jxie/flickr8k` |
| Dataset revision | `56f58c967835f7c508d684f36bd7897cca9d7634` |
| Validation set | 1,000 images and 5,000 caption queries |
| Test set | 1,000 images and 5,000 caption queries |
| Relevant result | The caption's source image only |
| Primary metric | Recall@1 |
| Other metrics | Recall@5, Recall@10, MRR, nDCG@10 |
| Test manifest SHA-256 | `2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce` |

## Step 1: Download the exact dataset files

Source page:

<https://huggingface.co/datasets/jxie/flickr8k/tree/56f58c967835f7c508d684f36bd7897cca9d7634/data>

Required files:

- [Validation Parquet](https://huggingface.co/datasets/jxie/flickr8k/resolve/56f58c967835f7c508d684f36bd7897cca9d7634/data/validation-00000-of-00001-7025a2b596f14b7b.parquet)
- [Test Parquet](https://huggingface.co/datasets/jxie/flickr8k/resolve/56f58c967835f7c508d684f36bd7897cca9d7634/data/test-00000-of-00001-42a2661d12c73e48.parquet)

Do not use files from `main`; use the pinned revision above.

Install the two preparation dependencies:

```bash
python -m pip install huggingface-hub pyarrow
```

Save this as `prepare_flickr8k_benchmark.py` in any project:

```python
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "jxie/flickr8k"
REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
FILES = {
    "validation": "data/validation-00000-of-00001-7025a2b596f14b7b.parquet",
    "test": "data/test-00000-of-00001-42a2661d12c73e48.parquet",
}
EXPECTED_PARQUET_SHA256 = {
    "validation": "2470e49d96fb37e19c8a1d29056623c84ed41c3760a12747e120695aa0c44ad4",
    "test": "f3bc4c3548ec1d9bb3c445c871cfcd3aca6003e3d4786e404fc9356b5215410d",
}
EXPECTED_MANIFEST_SHA256 = {
    "validation": "d3259e317aba8ae64a3e9ef152253f0fe4031289371bb214ace409a1ea0685b2",
    "test": "2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce",
}

root = Path("flickr8k_benchmark")
images_dir = root / "images"
images_dir.mkdir(parents=True, exist_ok=True)

for split, filename in FILES.items():
    parquet_path = Path(hf_hub_download(
        repo_id=REPO,
        filename=filename,
        repo_type="dataset",
        revision=REVISION,
    ))
    file_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    assert file_hash == EXPECTED_PARQUET_SHA256[split], (split, file_hash)

    rows = pq.read_table(parquet_path).to_pylist()
    assert len(rows) == 1000
    images = []
    queries = []

    for row in rows:
        image_id = Path(row["image"]["path"]).name
        (images_dir / image_id).write_bytes(row["image"]["bytes"])
        images.append({"split": split, "image_id": image_id})

        for caption_index in range(5):
            text = row[f"caption_{caption_index}"].strip()
            queries.append({
                "split": split,
                "query_id": f"{split}:{image_id}:{caption_index}",
                "relevant_image_id": image_id,
                "caption_index": caption_index,
                "text": text,
            })

    (root / f"{split}_images.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in images),
        encoding="utf-8",
    )
    (root / f"{split}_queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queries),
        encoding="utf-8",
    )

    canonical = json.dumps(
        {"images": images, "queries": queries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    assert manifest_hash == EXPECTED_MANIFEST_SHA256[split], (split, manifest_hash)
    print(split, len(images), len(queries), manifest_hash)
```

Run it:

```bash
python prepare_flickr8k_benchmark.py
```

Expected output:

```text
validation 1000 5000 d3259e317aba8ae64a3e9ef152253f0fe4031289371bb214ace409a1ea0685b2
test 1000 5000 2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce
```

If either assertion fails, stop. That project is not using the same benchmark data.

## Step 2: Use validation and test correctly

### Validation

- Gallery: the 1,000 images in `validation_images.jsonl`.
- Queries: all 5,000 rows in `validation_queries.jsonl`.
- Use validation to choose parameters, prompts, weights, thresholds, and method variants.

### Test

- Gallery: the 1,000 images in `test_images.jsonl`.
- Queries: all 5,000 rows in `test_queries.jsonl`.
- Run after every parameter and method choice has been frozen.
- Do not use test captions to train, tune, construct the image index, normalize scores, or select prompts.

For every query, `relevant_image_id` is the only relevant result. A similar image still counts as incorrect if it is not the caption's source image.

## Step 3: Export predictions in the same format

Each method must create `test_predictions.jsonl` with one row for every test query:

```json
{"query_id":"test:123.jpg:0","relevant_image_ids":["123.jpg"],"ranked_image_ids":["123.jpg","456.jpg"],"ranked_scores":[0.81,0.72],"text_encoding_ms":12.1,"search_ms":0.4,"end_to_end_ms":12.5}
```

Requirements:

- Include all 5,000 test query IDs exactly once.
- Copy `relevant_image_ids` from the benchmark ground truth; it must contain exactly the source image.
- Each `ranked_image_ids` array must contain all 1,000 test image IDs exactly once.
- Sort by descending score.
- Break exact score ties by ascending image ID.
- `ranked_scores` must align with `ranked_image_ids`.
- Record all three latency fields using Step 5. The shared evaluator requires them so every result has the same report shape.

## Step 4: Calculate the metrics identically

For a query, let `r` be the one-based rank of its relevant image.

```text
Recall@K(query) = 1 if r <= K, otherwise 0
MRR(query)      = 1 / r
nDCG@10(query)  = 1 / log2(r + 1) if r <= 10, otherwise 0
```

Final metrics are the arithmetic mean across all 5,000 caption queries. Report values between 0 and 1. Do not multiply them by 100 unless every participant agrees to report percentages.

Use the shared evaluator rather than implementing the formulas separately:

```bash
python scripts/score_predictions.py \
  --images flickr8k_benchmark/test_images.jsonl \
  --queries flickr8k_benchmark/test_queries.jsonl \
  --predictions test_predictions.jsonl \
  --output test_metrics.json
```

Both plain JSONL and gzip-compressed `.jsonl.gz` predictions are accepted. The evaluator rejects:

- Missing, extra, or duplicate queries.
- A gallery other than the canonical 1,000 images.
- Duplicate or missing ranked images.
- Incorrect relevance labels.
- Misaligned, non-finite, or ascending scores.
- Incorrect image-ID tie ordering.
- Missing, negative, or non-finite latency values.

It produces the same report fields as this project:

```text
query_count
recall_at_1
recall_at_5
recall_at_10
mrr
ndcg_at_10
text_encoding: mean_ms, p50_ms, p95_ms
search: mean_ms, p50_ms, p95_ms
end_to_end: mean_ms, p50_ms, p95_ms
```

### Optional shared score-threshold protocol

Use this only when a method filters low-scoring images. Do not copy another method's numeric cutoff because score scales differ between models and scoring functions.

First export complete rankings for all validation queries using the same prediction schema. Select the threshold from validation only:

```bash
python scripts/select_threshold.py \
  --images flickr8k_benchmark/validation_images.jsonl \
  --queries flickr8k_benchmark/validation_queries.jsonl \
  --predictions validation_predictions.jsonl \
  --output selected_threshold.json
```

The selector treats all 5,000 × 1,000 query–image pairs as binary examples. Each caption's source image is positive and the other 999 images are negative. It chooses the cutoff that maximizes image-pair F1 and chooses the higher cutoff when F1 ties. Freeze this file before evaluating test predictions.

Apply the frozen cutoff during test scoring:

```bash
python scripts/score_predictions.py \
  --images flickr8k_benchmark/test_images.jsonl \
  --queries flickr8k_benchmark/test_queries.jsonl \
  --predictions test_predictions.jsonl \
  --threshold-file selected_threshold.json \
  --output test_metrics.json
```

Keep all 1,000 ranked images and scores in every prediction row. The evaluator applies the cutoff without changing the prediction contract. It adds:

```text
threshold
candidate_precision
candidate_recall
candidate_f1
retained_candidate_count and retained_candidate_fraction
queries_with_results and query_coverage
average_results_per_query
thresholded recall_at_1, recall_at_5, recall_at_10, mrr, ndcg_at_10
```

A relevant image below the cutoff counts as a miss. Report these thresholded results separately from the primary complete-ranking metrics.

For a fair comparison, every method must select its own cutoff from its own validation scores with this exact rule. Use the same validation and test manifests for all methods. Do not compare the numeric cutoff itself across models; compare candidate precision, recall, F1, retained fraction, coverage, and thresholded ranking metrics.

MINER does not define a retrieval-result cutoff. Its confidence-gated ablation controls whether crop fusion is applied and was not adopted by the paper; its 5% object-area threshold is a ROCS dataset-construction rule. Any result filtering must therefore be labelled as a method-specific extension.

## Step 5: Measure latency fairly

Use this procedure only when comparing latency:

1. Use the same machine, device, precision, and software environment.
2. Load the model and image index before timing.
3. Use batch size 1.
4. Warm up on the first 50 queries.
5. Run each query three times and verify that its ranking does not change.
6. Synchronize CUDA or MPS before reading each timer.
7. Average the three runs for each query.
8. Across the 5,000 query averages, report mean, p50, and p95 milliseconds.

Definitions:

- Text encoding: tokenization, device transfer, and text model inference.
- Search: similarity calculation, method-specific reranking, and sorting.
- End to end: text encoding plus search.
- Exclude model loading, index loading, offline image encoding, and UI rendering.

Use linear percentile interpolation at position `(N - 1) × p`. Do not directly compare latency measured on different hardware.

### Offline preparation timing

Record these separately from query latency:

- Dataset download time.
- Parquet reading and image extraction time.
- Manifest construction time.
- Model download time.
- Image preprocessing and embedding/index construction time.
- Index storage size and peak memory when available.

State whether dataset and model files were already cached. Compare download time only for clean caches under similar network conditions. For method comparisons, image embedding/index construction time is more informative than the shared dataset download time.

Every participant must record preparation time. Use the shared timer around each preparation command:

```bash
python scripts/time_operation.py \
  --operation dataset_download_and_extraction \
  --cache-state clean \
  --output timing/dataset_preparation.json \
  -- python prepare_flickr8k_benchmark.py

python scripts/time_operation.py \
  --operation model_download \
  --cache-state clean \
  --output timing/model_download.json \
  -- python download_model.py

python scripts/time_operation.py \
  --operation image_index_construction \
  --cache-state clean \
  --output timing/image_index.json \
  -- python build_image_index.py
```

Replace the example commands with the actual commands in that project. Use `cached` when all inputs were already cached and `mixed` when only some were cached. Keep each stage separate; do not report one combined preparation number without its breakdown.

## Step 6: Compare methods statistically

Use a paired, image-clustered bootstrap because the five captions belonging to one image are correlated:

1. Pair methods by identical query ID.
2. Group the five queries for each source image.
3. Sample 1,000 image groups with replacement.
4. Keep all five captions whenever an image is sampled.
5. Calculate `candidate - baseline` for each metric.
6. Repeat 10,000 times with seed `20261009`.
7. Report the 2.5th and 97.5th percentiles as the 95% confidence interval.

Treat a Recall@1 improvement as supported when its delta is positive and the full 95% interval is above zero.

## Step 7: Report method details

Alongside the result table, every participant must report:

- Method name and version.
- Dataset test manifest hash.
- Model names and immutable checkpoint revisions.
- Parameters and how validation selected them.
- Score threshold, selection objective, and validation selection file when filtering is used.
- Whether the method uses training, generated text, or additional models.
- Hardware, device, numeric precision, and software versions.
- Latency batch size, warmup count, and timing runs.
- Dataset, model, and index preparation timing files, including cache state.

Use this table:

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | | | |
| Candidate | | | | | | | | |

## Reference result from this project

These numbers are a sanity check for the frozen SigLIP 2 implementation, not expected results for different methods:

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Global SigLIP 2 | 0.7220 | 0.9174 | 0.9606 | 0.8077 | 0.843356 |
| Five-crop fusion | 0.7368 | 0.9206 | 0.9658 | 0.818594 | 0.853057 |

Checkpoint: `google/siglip2-base-patch16-224` at revision `0ad8c6e0ff16615356a08a1ad8c8bbc8930c434e`.
