# Frozen SigLIP 2 Retrieval with Five-Crop Fusion

## Project overview

This project builds a research demo for text-to-image retrieval on Flickr8k using a pretrained, frozen Google SigLIP 2 checkpoint. It compares a standard global-image embedding baseline with a five-crop image representation designed to recover small or localized visual details.

**Research question:** Does combining a full-image SigLIP 2 similarity score with the strongest score from five fixed image crops improve Flickr8k text-to-image retrieval, and what latency and storage costs does it introduce?

SigLIP 2 remains unchanged throughout the project. Image and text embeddings are created with the same pretrained checkpoint, L2-normalized, and compared using cosine similarity. No SigLIP 2 weights are trained or fine-tuned.

## Scientific basis

The selected method is inspired by **MINER: Multi-crop INference-time Enhancement for Rare-Object Retrieval with Frozen Dual Encoders**. MINER represents each image with one global embedding and five fixed regional crops, then combines global similarity with the maximum crop similarity. It reports experiments with frozen CLIP, SigLIP, and SigLIP 2 on COCO, Flickr30k, and rare-object retrieval subsets. [MINER paper](https://arxiv.org/abs/2609.27142)

This project adapts the method to Flickr8k. It is not a reproduction of the paper because the dataset, evaluation protocol, and implementation differ. The first implementation also omits MINER's test-query-dependent CSLS stage so offline evaluation and the one-query-at-a-time demo use the same deployable scoring method.

## Methods

### 1. Global SigLIP 2 baseline

For each Flickr8k image:

1. Apply the checkpoint's documented image preprocessing.
2. Encode the full image with the frozen SigLIP 2 image encoder.
3. L2-normalize and cache one global image embedding.

For each text query:

1. Apply the checkpoint's tokenizer with `padding="max_length"`.
2. Encode the complete query with the frozen SigLIP 2 text encoder.
3. L2-normalize the query embedding.
4. Rank all gallery images by cosine similarity.

For normalized vectors, cosine similarity is the same as their dot product.

### 2. Five-crop fusion

For an image with width `W` and height `H`, create five crops measuring `0.6W × 0.6H`:

1. Top left.
2. Top right.
3. Bottom left.
4. Bottom right.
5. Center.

Encode and cache the full image and all five crops with the same frozen SigLIP 2 image encoder. For query vector `q`, global image vector `g_i`, and crop vectors `c_ij`:

`global_score(i) = cosine(q, g_i)`

`crop_score(i) = max_j cosine(q, c_ij)`

`final_score(i) = (1 - alpha) * global_score(i) + alpha * crop_score(i)`

Select `alpha` from `{0.0, 0.1, ..., 1.0}` using validation Recall@1. Break ties using validation MRR and then choose the smaller `alpha`. Freeze the selected value before evaluating the test split. The setting `alpha = 0` is the global baseline.

### Validation-selected result threshold

After selecting `alpha`, choose a fused-score threshold using validation data only. The shared operating point requires every method to retain at least 95% of its validation relevant images. Select the highest score cutoff satisfying that target, including all score ties, then freeze it before test evaluation. Each method receives a different numeric cutoff because score scales differ, but every method is evaluated at the same predeclared validation-recall target.

Test prediction files still contain the complete 1,000-image ranking. The evaluator separately reports metrics after excluding images below the cutoff, including candidate precision, candidate recall, candidate F1, retained-candidate fraction, query coverage, average results per query, thresholded R@1/5/10, MRR, and nDCG@10. The original complete-ranking metrics remain the primary benchmark results.

This cutoff is a project extension, not part of the MINER retrieval method. MINER reports a confidence-gated fusion ablation using `global_score < tau` and `crop_score > global_score`; it fires on roughly 5% of pairs and improves R@1 by no more than 0.04, so the paper applies fusion to every pair. The released MINER evaluator exposes alpha, crop, and CSLS parameters but no result threshold. MINER's separate 5% mask-area threshold is used to construct its ROCS dataset and is unrelated to retrieval-score filtering.

## End-to-end flow

```text
Offline image preparation
Flickr8k image
  ├── Full image ───────────────→ frozen SigLIP 2 → global embedding
  └── Five fixed crops ─────────→ frozen SigLIP 2 → five crop embeddings
                                                        ↓
                                                  embedding cache

Online query
Text query → frozen SigLIP 2 text encoder → normalized query embedding
                                                        ↓
                               global and maximum-crop similarities
                                                        ↓
                                          weighted score fusion
                                                        ↓
                                             ranked Flickr8k images
```

Image embeddings are computed once and reused. Each new query requires one text encoding followed by exact vector comparisons against the cache.

## Dataset and leakage controls

The standalone guide to share with collaborators using other projects is [`docs/SHARED_BENCHMARK_GUIDE.md`](SHARED_BENCHMARK_GUIDE.md). It contains pinned download links, dataset preparation code, the prediction format, and the shared metric evaluator.

- Use a documented Flickr8k split containing 6,000 training, 1,000 validation, and 1,000 test images.
- Use the five captions for each validation or test image as text queries.
- Use the caption's source image as the single relevant image.
- Tune `alpha` and every other method parameter using validation data only.
- Do not use test captions during cache construction, method selection, or parameter tuning.
- Keep images, model weights, cached embeddings, predictions, and generated reports out of version control.
- Record the exact split files, checkpoint identifier and revision, processor settings, software versions, and hardware.

The benchmark uses the 1,000 test images as its gallery and 5,000 test captions as queries. A separate all-images gallery may be built for the demo, but its results must not be reported as benchmark metrics.

## Retrieval and caching

Use exact NumPy or PyTorch matrix multiplication for the primary search implementation. Flickr8k is small enough that approximate indexing is unnecessary.

Store:

- Global embeddings with shape `[number_of_images, embedding_dimension]`.
- Crop embeddings with shape `[number_of_images, 5, embedding_dimension]`.
- Image IDs in the same stable order as the arrays.
- Cache metadata containing the checkpoint revision, split hash, crop ratio and order, vector dtype, preprocessing configuration, and normalization setting.

Use float32 vectors for the reproducible benchmark. Reject a cache when its metadata does not match the active experiment configuration.

## Evaluation and comparison

Every method must receive the same gallery, queries, and relevance labels.

### Retrieval metrics

- **Recall@1, Recall@5, Recall@10:** fraction of queries whose relevant image appears in the first 1, 5, or 10 results.
- **Mean reciprocal rank:** average reciprocal rank of the relevant image.
- **nDCG@10:** discounted rank of the relevant image within the first 10 results. With one relevant image it is less informative than Recall and MRR, but it is reported consistently.
- **Query latency:** mean, p50, and p95 batch-size-one latency, with text encoding and search time also reported separately.

Recall@1 is the predeclared primary metric. Report all other metrics as supporting results rather than combining them into an arbitrary composite score.

### Statistical comparison

Compare the five-crop method with the global baseline and any collaborator's method using paired differences on identical queries. Calculate a 95% confidence interval from 10,000 bootstrap samples, sampling by source image so all five captions for an image remain together. Use random seed `20261009`.

An improvement is supported when the Recall@1 difference is positive and its 95% confidence interval excludes zero.

Compare latency directly only when methods run on the same machine, device, precision, query order, batch size, and measurement procedure. Otherwise treat latency as descriptive metadata.

### Collaborator prediction format

Each method must export one prediction record per query containing:

- Query ID.
- Relevant image ID.
- All 1,000 gallery image IDs in ranked order.
- Corresponding scores.
- Text-encoding, search, and end-to-end latency.

Method metadata must state its model checkpoint, parameters, split and manifest hashes, hardware, software versions, and whether it uses another pretrained model, generated text, or learned parameters. A shared evaluator will validate these files and calculate all metrics.

## Gradio demo

The demo will:

- Accept a free-form text query.
- Allow selection between `Global baseline` and `Five-crop fusion`.
- Return the top-ranked Flickr8k images.
- Show rank, fused score, global score, maximum crop score, and winning crop.
- Filter five-crop results using its validation-selected score threshold and report
  when no image passes it.
- Leave the global baseline unfiltered unless it receives its own
  validation-selected threshold.
- Load the model and cached image embeddings once at startup.
- Clearly report missing or incompatible model, dataset, or cache assets.

The demo is illustrative. Research conclusions come from the separate held-out evaluation script.

## Implementation components

Keep the code divided into small components:

1. Experiment configuration and Flickr8k manifest loading.
2. Frozen SigLIP 2 text and image encoding.
3. Deterministic five-crop extraction.
4. Embedding cache creation and compatibility validation.
5. Global and five-crop retrieval with a shared result format.
6. Validation-only `alpha` selection.
7. Evaluation, prediction export, and collaborator comparison.
8. Gradio interface.

The detailed implementation plan is in [`docs/superpowers/plans/2026-10-09-five-crop-siglip2-retrieval.md`](superpowers/plans/2026-10-09-five-crop-siglip2-retrieval.md).

## Implementation sequence and difficulty

| Phase | Work | Difficulty |
|---|---|---|
| 1 | Configure the project, load Flickr8k, and create canonical manifests. | Easy to moderate |
| 2 | Encode and cache global image vectors; implement the exact baseline. | Moderate |
| 3 | Extract five crops per image and cache their embeddings. | Moderate |
| 4 | Implement score fusion and select `alpha` on validation data. | Moderate |
| 5 | Evaluate the baseline and five-crop method with confidence intervals. | Moderate |
| 6 | Add collaborator prediction validation and comparison reports. | Moderate |
| 7 | Build the Gradio demo with winning-crop visualization. | Moderate |

Overall difficulty: **6/10**.

## Deliverables

- Reproducible frozen SigLIP 2 global baseline.
- MINER-inspired five-crop fusion retrieval method.
- Cached global and crop embeddings with compatibility metadata.
- Validation-only fusion-weight selection.
- Validation-only result-threshold selection and thresholded metrics.
- Evaluation script reporting R@1/5/10, MRR, nDCG@10, and latency.
- Shared prediction contract and collaborator comparison report.
- Paired image-clustered bootstrap confidence intervals.
- Gradio retrieval demo.
- Setup and reproduction documentation.

## Risks and limitations

- Flickr8k is small and has limited visual and linguistic diversity.
- A caption may omit visible details, producing semantically reasonable results that count as incorrect under single-image relevance.
- Fixed crops increase cached embedding storage and offline image-encoding cost by roughly six times compared with the global baseline.
- Maximum crop similarity can promote an image that matches one object but conflicts with the complete scene; retaining the global score reduces this risk.
- SigLIP 2 may have encountered some Flickr images during pretraining; disclose possible data provenance overlap.
- The cited MINER work is a recent preprint, and this project evaluates an adaptation on a different dataset.

## References

- Tschannen et al., [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786).
- Alquwayfili et al., [MINER: Multi-crop INference-time Enhancement for Rare-Object Retrieval with Frozen Dual Encoders](https://arxiv.org/abs/2609.27142), preprint, 2026.
- Pugnana and Ruggieri, [AUC-based Selective Classification](https://proceedings.mlr.press/v206/pugnana23a.html), AISTATS 2023.
- Dhillon, Deligiannidis, and Rainforth, [On the Expected Size of Conformal Prediction Sets](https://proceedings.mlr.press/v238/dhillon24a.html), AISTATS 2024.
- TensorFlow, [Image captioning tutorial and Flickr8k dataset description](https://www.tensorflow.org/text/tutorials/image_captioning?authuser=1).
