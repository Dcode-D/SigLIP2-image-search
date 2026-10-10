# Five-Crop SigLIP 2 Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a frozen SigLIP 2 text-to-image retrieval system that combines one global image embedding with five fixed crop embeddings, and provide a shared output contract for fair comparison with other retrieval methods.

**Architecture:** Precompute normalized global and crop embeddings for each Flickr8k image using one frozen SigLIP 2 checkpoint. At query time, encode the text once, score every image globally and by its strongest crop, then combine those scores with a validation-selected weight. A standalone evaluator consumes method-independent prediction files so this method and independently implemented methods use identical queries, relevance labels, metrics, and comparison rules.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, Pillow, NumPy, pandas, PyYAML, Gradio, pytest; FAISS is optional and used only for exact-ranking parity checks.

**Spec:** `project.md`

## Global Constraints

- Keep `google/siglip2-base-patch16-224` frozen and in evaluation mode for all primary experiments.
- Resolve and record the checkpoint revision before producing final embeddings.
- Use the documented Flickr8k 6,000 train / 1,000 validation / 1,000 test image split.
- Use the five test captions per test image as queries and the source image as the single relevant image.
- Use only validation data to select the crop-fusion weight or any other method parameter.
- Do not use test captions in embedding construction, parameter selection, prompt selection, or training.
- Use `padding="max_length"` and the checkpoint's own processor for text and image preprocessing.
- L2-normalize every cached image vector and every query vector.
- Cache vectors as float32 and perform similarity scoring in float32.
- Keep datasets, model weights, cached embeddings, indexes, predictions, and generated reports out of version control.
- Report R@1, R@5, R@10, MRR, nDCG@10, mean latency, p50 latency, and p95 latency.
- Label every comparison method by whether it uses extra pretrained models, generated text, or learned parameters.

## Review Focus

- Missing or unreadable Flickr8k images must fail with the image ID and path, rather than silently changing the gallery.
- Cache loading must reject mismatched checkpoint revisions, split hashes, crop definitions, or normalization settings.
- Crop coordinates must remain valid for portrait, landscape, and very small images.
- Prediction files must be rejected when queries, gallery IDs, ranks, or relevance labels differ from the canonical benchmark manifest.
- Latency measurement must synchronize accelerator work and exclude offline image embedding construction.

## Fixed Research Protocol

### Data

- Validation experiment: 1,000 validation images, 5,000 validation-caption queries.
- Test experiment: 1,000 test images, 5,000 test-caption queries.
- Relevance: the image from which a caption originated is the only relevant image for that query.
- Primary metric: text-to-image Recall@1.
- Secondary metrics: Recall@5, Recall@10, MRR, and nDCG@10.
- Efficiency metrics: mean, p50, and p95 batch-size-one query latency.

### Five crops

For image width `W` and height `H`, use crop width `round(0.6 * W)` and crop height `round(0.6 * H)`. Clamp both to at least one pixel. Extract these crops in this stable order:

1. Top left: `(0, 0)`
2. Top right: `(W - crop_width, 0)`
3. Bottom left: `(0, H - crop_height)`
4. Bottom right: `(W - crop_width, H - crop_height)`
5. Center: `((W - crop_width) // 2, (H - crop_height) // 2)`

The checkpoint processor resizes each crop to its required model input after extraction.

### Scoring

For normalized query vector `q`, global image vector `g_i`, and crop vectors `c_i1...c_i5`:

```text
global_score(i) = q dot g_i
crop_score(i)   = max_j(q dot c_ij)
final_score(i)  = (1 - alpha) * global_score(i) + alpha * crop_score(i)
```

Select `alpha` from `{0.0, 0.1, ..., 1.0}` by validation R@1. Break ties using validation MRR, then choose the smaller alpha. Freeze that value before test evaluation. `alpha=0.0` is the plain global baseline.

### Comparison rule

- A method wins on retrieval quality when its R@1 is higher.
- Report the paired R@1 difference and a 95% confidence interval from 10,000 image-clustered bootstrap samples with seed `20261009`.
- Treat an improvement as supported only when that interval excludes zero.
- Use R@5, R@10, MRR, and nDCG@10 to describe ranking behavior, not to override the predeclared R@1 result.
- Compare p95 end-to-end latency separately. Do not combine accuracy and latency into an invented single score.
- For practical selection, present the quality/latency tradeoff as `(R@1, p95 milliseconds)`.
- Compare latency directly only when both methods run on the same machine, device, precision, query order, and batch size. Otherwise report latency as descriptive metadata.

---

### Task 1: Project configuration and canonical benchmark manifests

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/five_crop.yaml`
- Create: `src/vision_retrieval/__init__.py`
- Create: `src/vision_retrieval/config.py`
- Create: `src/vision_retrieval/data.py`
- Create: `scripts/prepare_manifest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `ImageRecord`, `QueryRecord`, `load_split()`, and canonical JSONL image/query manifests used by every later task.

- [ ] **Step 1: Define the package, dependencies, ignored artifact directories, and `configs/five_crop.yaml`.**

  Pin the Python version range and minimum compatible library versions. The configuration must include checkpoint ID/revision, dataset root, split files, output root, crop ratio `0.6`, five crop names, alpha grid, seed, and evaluation cutoffs.

- [ ] **Step 2: Write manifest validation checks in `tests/test_data.py`.**

  Cover stable image/query IDs, exactly five captions per image, duplicate IDs, missing files, split overlap, and deterministic manifest hashing.

- [ ] **Step 3: Implement the configuration and data interfaces.**

  Use these signatures:

  ```python
  def load_config(path: Path) -> ExperimentConfig: ...
  def load_split(config: ExperimentConfig, split: Literal["train", "validation", "test"]) -> tuple[list[ImageRecord], list[QueryRecord]]: ...
  def manifest_sha256(images: Sequence[ImageRecord], queries: Sequence[QueryRecord]) -> str: ...
  ```

- [ ] **Step 4: Implement `scripts/prepare_manifest.py`.**

  Write canonical image and query JSONL manifests for validation and test, plus a metadata JSON file containing counts and hashes. Assert 1,000 images and 5,000 queries per evaluation split.

- [ ] **Step 5: Verify the task.**

  Run `pytest tests/test_data.py -v` and `python scripts/prepare_manifest.py --config configs/five_crop.yaml`. Expect all checks to pass and both manifests to have the required counts.

- [ ] **Step 6: Commit the independently usable manifest builder once the directory is under Git.**

### Task 2: Frozen SigLIP 2 encoder and reproducible embedding cache

**Files:**
- Create: `src/vision_retrieval/encoder.py`
- Create: `src/vision_retrieval/crops.py`
- Create: `src/vision_retrieval/cache.py`
- Create: `scripts/build_image_cache.py`
- Test: `tests/test_crops.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `ExperimentConfig` and image manifests from Task 1.
- Produces: `FrozenSiglip2Encoder`, `five_crops()`, `EmbeddingCache`, and cached arrays shaped `[N, D]` for global vectors and `[N, 5, D]` for crop vectors.

- [ ] **Step 1: Write crop geometry checks.**

  Assert exact crop coordinates for landscape, portrait, square, and one-pixel images; assert all returned crops are non-empty and remain within the original image.

- [ ] **Step 2: Implement deterministic crop extraction.**

  ```python
  def five_crops(image: Image.Image, ratio: float = 0.6) -> list[tuple[str, Image.Image]]: ...
  ```

  Return crops in the fixed order from the research protocol.

- [ ] **Step 3: Implement the frozen encoder.**

  ```python
  class FrozenSiglip2Encoder:
      def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...
      def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
  ```

  Load the pinned checkpoint revision, call `eval()`, disable gradients, use `padding="max_length"`, and return L2-normalized float32 arrays.

- [ ] **Step 4: Write cache compatibility checks.**

  Assert rejection of changed checkpoint revision, manifest hash, crop ratio/order, vector dimension, image order, dtype, and normalization setting.

- [ ] **Step 5: Implement cache read/write and metadata.**

  ```python
  def save_embedding_cache(path: Path, cache: EmbeddingCache) -> None: ...
  def load_embedding_cache(path: Path, expected: CacheMetadata) -> EmbeddingCache: ...
  ```

  Store vectors separately from a small JSON metadata file. Write through a temporary path and rename only after all images succeed.

- [ ] **Step 6: Implement `scripts/build_image_cache.py`.**

  Batch global images and crops, preserve manifest order, and print progress plus final shapes. Build separate validation and test caches.

- [ ] **Step 7: Verify the task.**

  Run crop and cache checks, then build a small fixture cache. Confirm all vector norms are within `1e-5` of one and global/crop shapes match their manifests.

- [ ] **Step 8: Commit the encoder and cache once the directory is under Git.**

### Task 3: Exact global and five-crop retrieval

**Files:**
- Create: `src/vision_retrieval/retrieval.py`
- Create: `scripts/tune_alpha.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `FrozenSiglip2Encoder` and `EmbeddingCache` from Task 2.
- Produces: `SearchResult`, `GlobalRetriever`, `FiveCropRetriever`, and a frozen alpha parameter file.

- [ ] **Step 1: Write retrieval behavior checks.**

  Use hand-constructed normalized vectors to verify descending global ranking, crop max aggregation, fusion scores, stable image-ID mapping, deterministic tie breaking by image ID, and full-gallery output.

- [ ] **Step 2: Implement the shared result and retriever interfaces.**

  ```python
  @dataclass(frozen=True)
  class SearchResult:
      image_id: str
      score: float
      global_score: float
      crop_score: float | None
      winning_crop: str | None

  class GlobalRetriever:
      def search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[SearchResult]: ...

  class FiveCropRetriever:
      def search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[SearchResult]: ...
  ```

- [ ] **Step 3: Implement exact vectorized scoring.**

  Use NumPy or PyTorch matrix multiplication across the full gallery. Do not add approximate indexing to this task.

- [ ] **Step 4: Implement validation-only alpha selection.**

  `scripts/tune_alpha.py` must evaluate every grid value, apply the declared tie rules, and save alpha with the validation manifest hash, checkpoint revision, and timestamp.

- [ ] **Step 5: Verify the task.**

  Run retrieval checks and tune alpha on the validation split. Re-running with the same cache and seed must produce the identical alpha and metrics table.

- [ ] **Step 6: Commit the two retrieval methods once the directory is under Git.**

### Task 4: Method-independent metrics and prediction contract

**Files:**
- Create: `src/vision_retrieval/predictions.py`
- Create: `src/vision_retrieval/metrics.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_predictions.py`

**Interfaces:**
- Consumes: canonical manifests from Task 1 and any method's prediction JSONL file.
- Produces: validated prediction files, `MetricReport`, per-query metrics, aggregate metrics, and latency summaries.

- [ ] **Step 1: Fix the external prediction schema.**

  Every method writes one gzip-compressed JSONL row per query:

  ```json
  {
    "query_id": "test:<image_id>:<caption_index>",
    "relevant_image_ids": ["<image_id>"],
    "ranked_image_ids": ["image_0001", "image_0002"],
    "ranked_scores": [0.8123, 0.7901],
    "text_encoding_ms": 0.0,
    "search_ms": 0.0,
    "end_to_end_ms": 0.0
  }
  ```

  The two example ranking arrays stand for all 1,000 gallery entries. Store method metadata separately: method name/version, checkpoint/revision, manifest hashes, parameters, hardware, software versions, and flags for extra model, training, or generated text.

- [ ] **Step 2: Write metric checks using tiny known rankings.**

  Assert exact values for R@1/5/10, reciprocal rank, single-positive nDCG@10, missing relevant images, duplicate ranked IDs, non-finite scores, incomplete galleries, and mismatched query manifests.

- [ ] **Step 3: Implement per-query and aggregate metrics.**

  ```python
  def evaluate_ranking(ranked_ids: Sequence[str], relevant_ids: set[str]) -> QueryMetrics: ...
  def aggregate_metrics(rows: Sequence[EvaluatedQuery]) -> MetricReport: ...
  ```

  Use binary relevance. For a relevant result at one-indexed rank `r <= 10`, nDCG@10 is `1 / log2(r + 1)`; otherwise it is zero.

- [ ] **Step 4: Implement robust latency measurement.**

  Warm up with 50 unreported queries. Measure batch size one, synchronize CUDA/MPS where supported, exclude cache loading and image encoding, and report three complete runs plus their pooled mean, p50, and p95.

- [ ] **Step 5: Implement `scripts/evaluate.py`.**

  Run the chosen method over the canonical manifest, write predictions, per-query metrics CSV, aggregate metrics JSON, and a human-readable Markdown table.

- [ ] **Step 6: Verify the task.**

  Run the metric and schema checks. Evaluate `alpha=0.0` and the selected five-crop alpha on the test split without changing the saved alpha.

- [ ] **Step 7: Commit the reusable evaluator once the directory is under Git.**

### Task 5: Paired comparison with independently implemented methods

**Files:**
- Create: `src/vision_retrieval/compare.py`
- Create: `scripts/compare_methods.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: two or more prediction files conforming to Task 4's schema.
- Produces: paired metric differences, image-clustered bootstrap confidence intervals, latency comparisons, and a comparison Markdown/CSV report.

- [ ] **Step 1: Write comparison validation checks.**

  Reject different query sets, gallery sets, relevance labels, checkpoint/split metadata for same-backbone comparisons, and duplicated method names. Permit extra-model or trained methods only when their metadata declares those differences.

- [ ] **Step 2: Write deterministic bootstrap checks.**

  Construct a small example with multiple captions per image and assert that sampling occurs by source image, carries all five associated queries together, and reproduces identical intervals with seed `20261009`.

- [ ] **Step 3: Implement paired image-clustered bootstrap comparison.**

  ```python
  def paired_bootstrap(
      baseline: Sequence[EvaluatedQuery],
      candidate: Sequence[EvaluatedQuery],
      metric: str,
      samples: int = 10_000,
      seed: int = 20_261_009,
  ) -> ConfidenceInterval: ...
  ```

  Compute candidate-minus-baseline differences for R@1, R@5, R@10, MRR, and nDCG@10.

- [ ] **Step 4: Implement `scripts/compare_methods.py`.**

  Produce a table containing absolute metrics, absolute differences, relative R@1 difference, 95% intervals, significance flag, and latency mean/p50/p95. Include the `(R@1, p95 latency)` pair for each method. Mark latency comparisons as non-comparable when hardware, device, precision, query order, or batch size differs.

- [ ] **Step 5: Verify friend-method interoperability.**

  Create a synthetic external-method prediction file without importing retrieval code. Confirm the comparison script can validate and compare it solely through the documented schema.

- [ ] **Step 6: Commit the external comparison tooling once the directory is under Git.**

### Task 6: Gradio demo

**Files:**
- Create: `app.py`
- Create: `src/vision_retrieval/demo.py`

**Interfaces:**
- Consumes: encoder, test or demo gallery cache, global/five-crop retrievers, and selected alpha.
- Produces: a free-form text search UI with method selection and interpretable ranked results.

- [ ] **Step 1: Implement lazy application loading.**

  Load the model and cache once at startup. Validate metadata before enabling search and show a clear asset/setup error when required files are absent.

- [ ] **Step 2: Implement query search.**

  Accept non-empty text, encode it once, search using `Global` or `Five-crop fusion`, and return the top 10 images with rank, final score, global score, best-crop score, and winning crop name.

- [ ] **Step 3: Add result interpretation.**

  For five-crop results, overlay or separately display the winning crop. State the fixed alpha and checkpoint in the interface.

- [ ] **Step 4: Manually verify the demo.**

  Check an empty query, a normal Flickr-style description, repeated searches, method switching, portrait/landscape images, and missing-cache startup behavior.

- [ ] **Step 5: Commit the demo once the directory is under Git.**

### Task 7: Reproducibility documentation and final run

**Files:**
- Create: `README.md`
- Modify: `project.md`

**Interfaces:**
- Consumes: every command and artifact from Tasks 1–6.
- Produces: a reproducible setup/evaluation guide and the final result table.

- [ ] **Step 1: Document installation and dataset acquisition.**

  Include environment creation, dependencies, Flickr8k image/caption/split placement, checkpoint download behavior, storage requirements, and ignored artifact directories.

- [ ] **Step 2: Document the complete command sequence.**

  Cover manifest preparation, validation/test cache creation, alpha tuning, baseline evaluation, five-crop evaluation, method comparison, and Gradio startup.

- [ ] **Step 3: Document the comparison contract for collaborators.**

  Include the JSONL schema, metadata requirements, canonical manifest hashes, validation command, primary metric, confidence interval rule, and latency protocol.

- [ ] **Step 4: Run the complete pipeline from a clean artifact directory.**

  Confirm the test split is touched only after alpha is frozen. Preserve configuration, resolved checkpoint revision, manifest hashes, hardware description, metrics, and timing reports.

- [ ] **Step 5: Add the final baseline-versus-five-crop table to the report.**

  Report R@1/5/10, MRR, nDCG@10, mean/p50/p95 latency, paired R@1 difference, and its 95% confidence interval. Describe whether the result supports an improvement.

- [ ] **Step 6: Commit documentation and small result summaries only.**

  Do not commit images, model weights, cached vectors, indexes, or full prediction artifacts.

## Deliverables

- Frozen global SigLIP 2 baseline.
- Frozen five-crop fusion retriever with validation-selected alpha.
- Reproducible validation and test manifests.
- Cached global and five-crop embeddings with compatibility metadata.
- Standalone evaluation and collaborator comparison scripts.
- Paired image-clustered confidence intervals.
- Gradio demo showing ranked images and winning crops.
- Reproduction guide and final comparison report.

