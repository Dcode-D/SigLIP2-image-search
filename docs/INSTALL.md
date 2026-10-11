# Installation and first run

Everything below was run end to end on the reference machine (macOS 26.6, Apple M3 Pro,
36 GB RAM, Python 3.13) on 2026-10-10. Linux with CUDA and Windows are supported by the same
commands; differences are called out where they exist. Run every command from the
repository root.

## 1. Requirements

| | Minimum | Reference run |
|---|---|---|
| Python | 3.9+ | 3.13.13 |
| Disk | ~4 GB free | data 528 MB · checkpoint 1.4 GB · venv 1.0 GB · artifacts 0.6 GB |
| Network | first run only | downloads the two Flickr8k Parquet shards and the pinned checkpoint |
| Accelerator | none (CPU works) | Apple MPS; CUDA is picked automatically when present |

Pinned package versions of the reference run: torch 2.14.1, transformers 5.19.0,
numpy 2.5.3, faiss-cpu 1.15.1, gradio 6.30.0, pillow 12.3.0, pyarrow 23.0.1,
huggingface-hub 1.33.0. `pyproject.toml` only sets ranges; these are the versions the
numbers in `docs/FAISS_RESULTS.md` were produced with.

## 2. Create the environment and install

With the standard library venv (has `pip`):

```bash
git clone <repository-url> && cd <repository-directory>
python3 -m venv .venv && source .venv/bin/activate
make install            # = pip install --upgrade pip setuptools wheel && pip install -e .
```

With `uv` (faster; a `uv venv` has no `pip`, so use the uv target):

```bash
uv venv .venv && source .venv/bin/activate
make install-uv         # = uv pip install --python .venv/bin/python -e .
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`, then run the `python -m pip ...`
commands from the Makefile by hand (`make` is usually not installed).

Check the device the code will pick:

```bash
python -c "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))"
```

Every model command accepts `--device cpu|mps|cuda` to override.

## 3. Unit tests (no data or model needed, ~1 s)

```bash
make test               # = PYTHONPATH=src python -m unittest discover -s tests
```

Expected: `Ran 74 tests ... OK`.

## 4. Download the pinned assets

```bash
make assets             # = scripts/setup_assets.py && scripts/prepare_manifest.py
```

Reference run: 129 s in total (validation shard 49 s, test shard 9 s, checkpoint 67 s).
Result:

```
data/flickr8k/            2,000 images, Flickr8k.token.txt, dev/test split lists, SOURCE.json
models/huggingface/       google/siglip2-base-patch16-224 @ 0ad8c6e0ff16615356a08a1ad8c8bbc8930c434e
artifacts/manifests/      validation_*.jsonl, test_*.jsonl, metadata.json
artifacts/timing/         asset_setup.json, manifest_preparation.json
```

Stop if `artifacts/manifests/metadata.json` does not show these hashes:

```
validation  1,000 images / 5,000 queries  d3259e317aba8ae64a3e9ef152253f0fe4031289371bb214ace409a1ea0685b2
test        1,000 images / 5,000 queries  2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce
```

## 5. Build the image embedding cache (once)

```bash
make cache              # = scripts/build_image_cache.py --split all --batch-size 16
```

Reference run on MPS: 88 s (validation) + 82 s (test). Out of memory: `make cache BATCH=8`.
Output: `artifacts/embeddings/{validation,test}/{global.npy,crops.npy,metadata.json}`
(35 MB). The script refuses to overwrite an existing cache; delete the split folder to
rebuild.

## 6. Select the five-crop parameters (validation only)

```bash
make tune               # = scripts/tune_alpha.py --batch-size 64
```

Expected: `alpha = 0.5` in `artifacts/parameters/five_crop_alpha.json` and a cutoff of
`0.1228` in `five_crop_threshold.json`.

## 7. Smoke test

```bash
python scripts/query.py "two dogs playing in the snow" --method faiss_ivf --top-k 5
```

Prints the ranked image ids, scores and local paths, plus build / encode / search times.

## 8. Produce the metrics

```bash
make metrics            # = bash scripts/run_metrics.sh   (~14 min on the M3 Pro)
cat artifacts/reports/summary.md
```

What it does, step by step, and how to run the steps by hand (needed on Windows, where
the runner is a bash script): `docs/RUN_METRICS.md`. `make diagnostics` adds the IVF
`nprobe` sweep and the index scaling study (another ~2 min).

## 9. Demo

```bash
make demo               # = python app.py  ->  http://127.0.0.1:7860
python app.py --host 0.0.0.0 --port 7861 --device cpu     # other interface / port / device
```

Start-up takes 20–30 s (model load + building every method); the log prints one `[ok]`
and one `[thr]` line per method. Needs the test embedding cache, `five_crop_alpha.json`,
and, for the threshold checkbox to have an effect, the threshold files from step 8
(a method without one still works, unfiltered).

## Disk hygiene

| Path | Size | Safe to delete? |
|---|---|---|
| `artifacts/predictions/` | ~580 MB | yes, any `evaluate.py` run recreates its file (3 min each) |
| `artifacts/reports/`, `artifacts/run_metrics.log` | few MB | yes, `make metrics` recreates them |
| `artifacts/embeddings/` | 35 MB | keep; rebuilding costs 3 min of GPU/MPS time |
| `artifacts/parameters/`, `artifacts/manifests/`, `artifacts/timing/` | < 1 MB | keep; they are the provenance of every number |
| `models/huggingface/`, `data/flickr8k/` | 1.9 GB | keep unless you accept re-downloading |

`make clean` only removes `__pycache__` and packaging metadata.

### Restoring after a clean-up

If `models/`, `data/`, `.venv/` and `artifacts/predictions/` were deleted to free disk
(everything else under `artifacts/` kept), this brings the demo back in about 5 minutes:

```bash
python3 -m venv .venv && source .venv/bin/activate && make install   # or: uv venv .venv && make install-uv
make assets            # re-downloads the pinned shards + checkpoint; manifests are rebuilt identically
make demo              # embeddings, alpha and thresholds were kept, so nothing else is needed
make metrics           # only if you need the prediction files / summary again (~14 min)
```

The kept embedding cache is validated against the manifest hash and the checkpoint
revision when loaded, so a re-download that matches is accepted and `make cache` can be
skipped.

## Troubleshooting

- **`OMP: Error #15 ... libomp.dylib already initialized`** (macOS, torch + faiss in one
  process): handled by the code, which sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing
  FAISS. If you import `faiss` in a script of your own, import
  `vision_retrieval.methods.faiss_retrievers` first or set the variable yourself.
- **Hangs or crashes inside IVF training**: FAISS is pinned to one thread for that reason
  (`FaissRetriever(threads=1)`); do not raise it while torch is loaded in the same process.
- **`No module named pip`**: the venv was created by `uv`; use `make install-uv`.
- **Hub access on a machine without network**: after step 4 everything is read from
  `models/huggingface/` and `data/flickr8k/`; if transformers still tries to reach the Hub,
  set `HF_HUB_OFFLINE=1`.
- **MPS out of memory or a MPS-specific error**: add `--device cpu` (slower, same numbers)
  or lower the batch size.
- **`selected threshold not found`**: run step 8 (or `python scripts/evaluate.py --method global --split validation --timing-runs 1 --warmup 0`) before a test evaluation with `--require-threshold`.
- **Demo says "needed: 3, got: 2"**: the browser tab was loaded before the server was
  restarted; reload the page.
- **Port 7860 in use**: `python app.py --port 7861`, or find the old process with
  `lsof -nP -iTCP:7860 -sTCP:LISTEN`.
- **Numbers differ from the docs**: quality metrics must match to the fourth decimal if the
  manifest hashes match; latency and build times are machine-specific and only comparable
  on the same hardware.
