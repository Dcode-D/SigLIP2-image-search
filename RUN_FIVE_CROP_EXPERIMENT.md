# Reproduce the Flickr8k Five-Crop Experiment

Use this guide after pulling a fresh copy of the repository onto another machine. The commands recreate the local dataset, pinned SigLIP 2 model, embeddings, validation-selected fusion weight, predictions, metrics, and preparation timing reports.

Run every command from the repository root.

## 0. Clone and install

```bash
git clone <repository-url>
cd <repository-directory>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Python 3.9 or newer is required. A first-time setup requires internet access and several gigabytes of free disk space.

## 1. Download and prepare the assets

This downloads only the pinned Flickr8k validation/test splits and the pinned frozen SigLIP 2 checkpoint:

```bash
python scripts/setup_assets.py
```

Expected local data:

```text
data/flickr8k/
├── images/                       # 2,000 images
├── Flickr8k.token.txt            # 10,000 captions
├── Flickr_8k.devImages.txt       # 1,000 validation images
├── Flickr_8k.testImages.txt      # 1,000 test images
└── SOURCE.json

models/huggingface/               # Pinned SigLIP 2 checkpoint cache
```

The setup timing report is saved to:

```text
artifacts/timing/asset_setup.json
```

It records dataset download, Parquet reading, image extraction, model download, total time, and whether the files were already cached. Compare clean preparation times only when both cache flags are `false`.

## 2. Verify the installation and device

```bash
python -c "import torch, transformers, numpy, pyarrow, PIL; mps=getattr(torch.backends, 'mps', None); device='cuda' if torch.cuda.is_available() else ('mps' if mps and mps.is_available() else 'cpu'); print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Device:', device)"
```

The project automatically selects CUDA, then Apple MPS, then CPU. You can explicitly select a device by adding `--device cuda`, `--device mps`, or `--device cpu` to model commands.

## 3. Run the code checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## 4. Generate and validate manifests

```bash
python scripts/prepare_manifest.py
cat artifacts/manifests/metadata.json
```

Windows PowerShell:

```powershell
Get-Content artifacts/manifests/metadata.json
```

Expected counts and hashes:

```text
Validation: 1,000 images, 5,000 queries
SHA-256:   d3259e317aba8ae64a3e9ef152253f0fe4031289371bb214ace409a1ea0685b2

Test:       1,000 images, 5,000 queries
SHA-256:   2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce
```

Manifest preparation time is saved to:

```text
artifacts/timing/manifest_preparation.json
```

Stop if the counts or hashes differ.

## 5. Build image embeddings

This encodes one full image and five fixed crops for every validation and test image:

```bash
python scripts/build_image_cache.py --split all --batch-size 16
```

If memory runs out:

```bash
python scripts/build_image_cache.py --split all --batch-size 8
```

Embeddings are saved under:

```text
artifacts/embeddings/
├── validation/
│   ├── global.npy
│   ├── crops.npy
│   └── metadata.json
└── test/
    ├── global.npy
    ├── crops.npy
    └── metadata.json
```

Index-construction timing is saved to:

```text
artifacts/timing/image_cache_validation.json
artifacts/timing/image_cache_test.json
```

These reports separate model loading, dataset loading, image loading/cropping/encoding, and cache writing. Record the batch size and device when comparing preparation time.

The command refuses to overwrite an existing embedding cache. Reuse the existing cache and its original timing report, or use a fresh checkout/output directory for a new timing run.

## 6. Select the five-crop fusion weight

```bash
python scripts/tune_alpha.py --batch-size 64
cat artifacts/parameters/five_crop_alpha.json
```

Windows PowerShell:

```powershell
Get-Content artifacts/parameters/five_crop_alpha.json
```

This evaluates every configured alpha on the validation set only. It selects the highest validation Recall@1, then highest MRR, then the smaller alpha when tied.

Expected selected value for the pinned experiment:

```text
alpha = 0.5
```

Do not select alpha using test results.

## 7. Evaluate the five-crop method

```bash
python scripts/evaluate.py --method five_crop
```

Outputs:

```text
artifacts/predictions/five_crop.jsonl.gz
artifacts/predictions/five_crop.metadata.json
artifacts/reports/five_crop_per_query.csv
artifacts/reports/five_crop_metrics.json
artifacts/reports/five_crop_metrics.md
```

View the report:

```bash
cat artifacts/reports/five_crop_metrics.md
```

Windows PowerShell:

```powershell
Get-Content artifacts/reports/five_crop_metrics.md
```

The report contains:

- Recall@1
- Recall@5
- Recall@10
- MRR
- nDCG@10
- Mean, p50, and p95 query latency
- Separate text-encoding and search latency in the JSON report

Reference quality results from the completed pinned run:

```text
R@1:     0.736800
R@5:     0.920600
R@10:    0.965800
MRR:     0.818594
nDCG@10: 0.853057
```

Quality metrics should match apart from very small numerical or tie differences. Latency will vary by machine.

## 8. Optional: reproduce the common global baseline

The five-crop method is this project's candidate. Run the global baseline when you want to reproduce the complete experiment or verify a collaborator's baseline:

```bash
python scripts/evaluate.py --method global
cat artifacts/reports/global_metrics.md
```

Reference global quality results:

```text
R@1:     0.722000
R@5:     0.917400
R@10:    0.960600
MRR:     0.807700
nDCG@10: 0.843356
```

If another collaborator owns the global baseline, the five-crop command in Step 7 is the only required test evaluation. Their predictions must use the same test manifest, checkpoint, gallery, queries, and relevance rule described in [`docs/SHARED_BENCHMARK_GUIDE.md`](docs/SHARED_BENCHMARK_GUIDE.md).

## 9. Try a raw query

```bash
python scripts/query.py "a dog running through grass" --method five_crop --top-k 5
```

The command prints ranked scores and local image paths.

## 10. Files to keep for comparison

Keep these files with the experiment result:

```text
configs/five_crop.yaml
data/flickr8k/SOURCE.json
artifacts/manifests/metadata.json
artifacts/parameters/five_crop_alpha.json
artifacts/predictions/five_crop.jsonl.gz
artifacts/predictions/five_crop.metadata.json
artifacts/reports/five_crop_metrics.json
artifacts/reports/five_crop_metrics.md
artifacts/timing/asset_setup.json
artifacts/timing/manifest_preparation.json
artifacts/timing/image_cache_validation.json
artifacts/timing/image_cache_test.json
```

Dataset files, model weights, embeddings, predictions, and reports are ignored by Git. Archive the result files separately when they need to be shared.

## 11. Fair comparison rules

- Compare quality only when the test manifest hash is `2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce`.
- Tune every method on validation only.
- Use the caption's source image as the single relevant result.
- Quality metrics can be compared across machines.
- Compare latency and index-construction time only on the same hardware with the same device, precision, batch size, warmup, and timing procedure.
- Do not add offline dataset/index preparation time to online query latency.
- Use the standalone collaborator protocol in [`docs/SHARED_BENCHMARK_GUIDE.md`](docs/SHARED_BENCHMARK_GUIDE.md).
