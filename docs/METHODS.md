# Retrieval methods: how they plug in, how to switch, how to add one

One repository, one embedding cache, several ways to search it. This page is for the
team member who owns a method and has to run, demo and explain it.

## The idea in one picture

```
                         offline, once                      online, per query
Flickr8k images ──► frozen SigLIP 2 ──► global.npy [N,768] ─┐
                                   └──► crops.npy  [N,5,768]│   text ──► frozen SigLIP 2 ──► q [768]
                                                            ▼                                   │
                                                 ┌──────────────────────────┐                   │
                                                 │  METHOD (pick one)       │◄──────────────────┘
                                                 │  global      NumPy exact │
                                                 │  five_crop   MINER       │──► ranked image IDs + scores
                                                 │  faiss_flat  exact scan  │     (always the full gallery for
                                                 │  faiss_ivf   cells       │      the benchmark contract)
                                                 └──────────────────────────┘
```

Everything above the method box is shared and frozen. A method is only the box.

## Switching methods

| Where | How |
|---|---|
| Benchmark | `python scripts/evaluate.py --method faiss_ivf` (any name from the table below) |
| One query | `python scripts/query.py "two dogs in the snow" --method faiss_ivf --top-k 5` |
| Demo | `python app.py`, then choose in the **Method** dropdown; the description and timing update live. The **threshold checkbox** hides results below the method's validation-selected cutoff (FAISS methods use `global`'s), so one query can be shown with and without it |
| List | `python scripts/evaluate.py --help` or `python -c "from vision_retrieval.methods import list_methods; print(list_methods())"` |

All methods write predictions in the shared format (`artifacts/predictions/<method>.jsonl.gz`)
and are scored by the same `scripts/score_predictions.py`, so their numbers are comparable.

## Registered methods

| Name | Family | What changes vs the baseline | Parameters | Owner's one-liner |
|---|---|---|---|---|
| `global` | exact | nothing: cosine against every gallery vector with one matmul | – | "The honest brute-force answer." |
| `five_crop` | five_crop | image representation: 1 global + 5 crop vectors, fused score | `alpha` (validation-selected), `crop_ratio=0.6` | "Look at five regions too, so small objects are not averaged away." |
| `faiss_flat` | faiss | library only: FAISS exact inner-product scan | – | "Same ranking as NumPy, proves the library does not change the answer." |
| `faiss_ivf` | faiss | search space: k-means cells, scan only the `nprobe` nearest cells | `nlist=32`, `nprobe=8` | "Sort the books onto 32 shelves, open only the 8 closest shelves." |

The FAISS entry is **one method, approximate nearest-neighbour search with FAISS**, shown
at two operating points: the exact scan (control) and the inverted-file index (the actual
idea). Because all gallery vectors have unit length, inner product = cosine, so `faiss_flat`
reproduces `global` exactly (checked by `tests/test_faiss_retrievers.py`).

### What Flat and IVF trade

| | Exact (`global`, `faiss_flat`) | IVF (`faiss_ivf`) |
|---|---|---|
| Offline step | none | k-means with `nlist` centroids on the gallery vectors |
| Per query | compare with all N vectors | compare with `nlist` centroids, then with the vectors in the `nprobe` closest cells |
| Vectors compared (N = 1,000) | 1,000 | 32 + ~250 |
| Can miss the true best image? | no | yes, if its cell is not among the `nprobe` visited |
| Memory | N·d·4 bytes | same + `nlist`·d·4 bytes of centroids |
| Knob | – | `nprobe` ↑ = more recall, more time; `nprobe = nlist` is exact again |

### IVF's one knob: `nprobe`

`scripts/sweep_ivf_nprobe.py` on the real 1,000-image test gallery, all 5,000 test captions,
`nlist=32` (numbers from the 2026-10-10 run; deterministic given the seed):

| nprobe | gallery scanned | R@1 | R@10 | top-10 overlap with exact |
|---:|---:|---:|---:|---:|
| 1 | 3% | 0.3712 | 0.4804 | 0.416 |
| 2 | 6% | 0.4914 | 0.6396 | 0.577 |
| 4 | 12% | 0.6088 | 0.7978 | 0.748 |
| **8** | **25%** | **0.6802** | **0.9002** | **0.895** |
| 16 | 50% | 0.7190 | 0.9570 | 0.986 |
| 32 | 100% | 0.7220 | 0.9606 | 1.000 |

Reading: scanning a quarter of the gallery costs about 4 R@1 points; scanning half costs
0.3 points; scanning everything is exact. This is the table to show on stage, because
it is the whole method in six rows.

### When does IVF pay off?

Not at 1,000 images: the search step is under 1 ms of a ~10 ms query and text encoding
dominates, so IVF can only lose recall here. `scripts/benchmark_index_scaling.py` grows
the gallery with synthetic vectors (mixes of pairs of real image embeddings plus a little
noise, renormalised; descriptive, not a benchmark) to 10k and 100k and reports build
time, memory, search latency and recall@10 against exact search, everything
single-threaded. Results: `docs/FAISS_RESULTS.md`.

IVF is built with the L2 metric on the unit vectors (what the index type is designed for)
and distances are mapped back to cosine with `cos = 1 - d/2`.

### Full ranking from an approximate index

The contract needs every gallery image in every ranking. An IVF index returns `-1` for
slots it cannot fill when the visited cells hold fewer than `k` vectors.
`faiss_retrievers.complete_ranking` appends the missing images after the found ones, all
with one filler score below the lowest returned score, ordered by image ID. The miss
therefore shows up as a lower Recall@K, never as an invalid prediction file.

### Score threshold per method

The shared guide's optional cutoff (keep 95% of validation relevant images) is linked per
method through `MethodSpec.threshold_source` (`src/vision_retrieval/evaluation/thresholds.py`). `global` and `five_crop` select their own
on validation; `faiss_flat` and `faiss_ivf` reuse `global`'s file because they score on the
same cosine scale, and IVF could not select one anyway: only ~92% of validation relevant
images sit in the 8 probed cells, so no cutoff reaches 95%. Selection:
`python scripts/evaluate.py --method global --split validation`; application: the test
run picks the file up automatically. Full pipeline: `docs/RUN_METRICS.md`.

## Questions the lecturer may ask (and the short answers)

1. **What exactly is your method?** The image and text embeddings are the shared frozen
   SigLIP 2 vectors. My method is the search block: instead of comparing the text vector
   with all 1,000 image vectors, FAISS IVF clusters the gallery with k-means and compares
   only with the images in the nearest clusters.
2. **Why is `faiss_flat` there if it equals the baseline?** It is the control. Same
   numbers to the fourth decimal show that FAISS does not change the answer; only IVF does.
3. **Why L2 distance when the baseline uses cosine?** All vectors are unit length, so
   `||a−b||² = 2 − 2·cos`. The ranking is the same, and I convert back with `cos = 1 − d/2`.
4. **How were `nlist` and `nprobe` chosen?** `nlist ≈ √N` is the FAISS rule of thumb
   (32 for 1,000). `nprobe=8` scans a quarter of the gallery, which makes the loss visible
   but not absurd. Nothing was tuned on the test split; the whole sweep is reported.
5. **Why is IVF worse than brute force?** Because it is approximate: when the relevant
   image sits in a cell that is not visited it cannot be returned. At 1,000 images brute
   force is already sub-millisecond, so IVF has nothing to gain. The scaling study shows
   where it starts to win.
6. **What does the −1 from FAISS mean and how do you handle it?** Fewer than `k` vectors in
   the visited cells. I append the unvisited images after the found ones with a filler
   score, so the file still ranks all 1,000 images and the miss counts as a miss.
7. **Which metric is primary and how is a difference judged?** Recall@1 on 5,000 test
   captions; differences come with a paired, image-clustered bootstrap 95% interval
   (`scripts/compare_methods.py`, guide Step 6).
9. **Why does IVF use the baseline's threshold?** The threshold says which cosine value
   counts as a plausible match; that depends on the embedding model, not on the index.
   IVF returns the same cosines for the images it reaches, so the same cutoff applies.
   Selecting on IVF's own validation scores is impossible at 95%: ~8% of relevant images
   are in unvisited cells and carry a filler score, so no cutoff reaches the target.
8. **Did you try other indexes?** Yes, HNSW (R@1 0.7132) and PQ (0.5806) early on. Both
   were removed to keep one clear idea: exact scan vs inverted file.

## Package layout (since 2026-10-11)

```
src/vision_retrieval/
├── config.py, encoder.py, timing.py        unchanged (config file: configs/experiment.yaml, was five_crop.yaml)
├── data/        flickr8k.py (was data.py) · cache.py · crops.py · setup_assets.py
├── methods/     registry.py (was methods.py, registry part) · builtin.py (global, five_crop registrations)
│                retrievers.py (was retrieval.py) · faiss_retrievers.py
├── evaluation/  metrics.py · predictions.py · thresholds.py
└── demo.py
```

Dependency direction: `data` → `methods` → `evaluation` → `demo`; nothing imports upward.
Public names are re-exported, so `from vision_retrieval import load_split, GlobalRetriever,
list_methods` and `from vision_retrieval.methods import build_method` are unchanged. Old
module paths map to: `vision_retrieval.data` → `.data.flickr8k`, `.cache` → `.data.cache`,
`.crops` → `.data.crops`, `.retrieval` → `.methods.retrievers`, `.metrics` → `.evaluation.metrics`,
`.predictions` → `.evaluation.predictions`. Scripts, CLI flags and every artifact path and
format are unchanged.

## Adding a method

1. Write a retriever class with

   ```python
   def search(self, query_vector, top_k=None) -> list[SearchResult]
   ```

   returning `SearchResult(image_id, score, global_score, crop_score, winning_crop)`
   sorted by descending score, ties by ascending `image_id`, and the **whole gallery**
   when `top_k is None`.

2. Write a builder `(BuildContext) -> (retriever, parameters)`. `context.cache` gives you
   `global_embeddings`, `crop_embeddings` and `metadata.image_ids`; `context.config`
   gives paths; `context.checkpoint_revision` lets you validate tuned parameter files.
   Tune parameters on the **validation** split only and persist them under
   `artifacts/parameters/`, as `five_crop` does with alpha.

3. Register it once:

   ```python
   from vision_retrieval.methods import MethodSpec, register
   register(MethodSpec("my_method", "My method (readable label)", "my_family",
                       "One paragraph you can read aloud.", build_my_method,
                       parameters={"k": 3}))
   ```

   Put the file in `src/vision_retrieval/methods/` next to `faiss_retrievers.py` and import it
   in `methods/__init__.py` (that import is what registers it). `evaluate.py`, `query.py`
   and the demo pick it up automatically.

4. Add a test in `tests/` (synthetic normalised vectors are enough) and run
   `PYTHONPATH=src python -m unittest discover -s tests`.

A FAISS variant with other settings is a one-liner:

```python
from vision_retrieval.methods.faiss_retrievers import register_faiss_method
register_faiss_method("faiss_ivf_half", "FAISS IVF (nlist=32, nprobe=16)", "...", kind="ivf", nlist=32, nprobe=16)
```

## macOS note: torch + faiss in one process

On Apple Silicon both `torch` and `faiss-cpu` ship their own OpenMP runtime. Loading both
aborts with `OMP: Error #15 ... libomp.dylib already initialized`. `faiss_retrievers`
sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing FAISS, which is the documented
workaround; FAISS is also pinned to one thread for reproducible single-query timing.
If you import `faiss` yourself in a new script, call `vision_retrieval.methods.faiss_retrievers._faiss()`
or set the variable first.

## Known open point for the whole team

`FrozenSiglip2Encoder.encode_texts` does not lowercase text and does not pass
`max_length=64` explicitly. The SigLIP 2 paper (Sec. 2.1) lowercases before
tokenising and the Gemma tokenizer in this checkpoint does not do it by itself. In an
earlier experiment on the same split, adding `.lower()` moved the global baseline from
R@1 0.72 to 0.76. Changing it changes every method's numbers equally, so it should be
decided once by the team and then re-run for all methods.
