"""Gradio demo: one text box, a method selector fed by the method registry, one
checkbox for the validation-selected score threshold.

Every registered method is built once at start-up from the same embedding cache, so
switching the dropdown re-runs only the search step and the latency shown is the
honest difference between methods. A method whose assets are missing (for example
five-crop before ``tune_alpha.py`` has run) is listed as unavailable instead of
crashing the whole demo.

Thresholds follow ``vision_retrieval.thresholds``: each method applies the cutoff frozen
by its ``threshold_source`` (``global`` and ``five_crop`` own theirs, FAISS reuses
``global``'s). The checkbox lets the presenter show the same query with and without the
cutoff; a method without a threshold file simply shows the top results and says so.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .data.cache import CacheMetadata, load_embedding_cache, read_cache_metadata
from .config import ExperimentConfig, load_config
from .data.crops import five_crops
from .data.flickr8k import load_split, manifest_sha256
from .encoder import FrozenSiglip2Encoder
from .methods import BuildContext, BuiltMethod, build_method, get_method, list_methods
from .methods.retrievers import SearchResult
from .evaluation.thresholds import load_threshold, threshold_path

TOP_K = 10


def apply_score_threshold(
    results: Sequence[SearchResult], threshold: Optional[float]
) -> Tuple[List[Tuple[int, SearchResult]], int]:
    """Keep (rank, result) pairs whose score reaches the threshold; None keeps everything.

    Ranks are the original 1-based positions, so a filtered list still reads "#1, #2, #5".
    Returns the kept pairs and how many were dropped.
    """
    ranked = list(enumerate(results, start=1))
    if threshold is None:
        return ranked, 0
    kept = [(rank, result) for rank, result in ranked if result.score >= threshold]
    return kept, len(ranked) - len(kept)


def threshold_status(
    label: str,
    threshold: Optional[float],
    source: str,
    target_recall: float,
    shown: int,
    total: int,
    applied: bool,
    reason: str = "",
) -> str:
    if not applied:
        return f"Threshold **off**: showing the top {total} results of {label}."
    if threshold is None:
        return (
            f"No validation-selected threshold is available for {label}"
            + (f" ({reason})" if reason else "")
            + f"; showing the top {total} results."
        )
    origin = "its own validation scores" if source == "" else f"the validation scores of `{source}`"
    head = (
        f"Threshold **on**: score ≥ `{threshold:.4f}`, selected on {origin} to keep "
        f"{target_recall:.0%} of relevant images. "
    )
    if shown == 0:
        return head + f"**No image** of the top {total} passed the cutoff."
    return head + f"Showing **{shown} of the top {total}**; {total - shown} below the cutoff hidden."


@dataclass
class DemoRuntime:
    config: ExperimentConfig
    encoder: FrozenSiglip2Encoder
    methods: Dict[str, BuiltMethod]  # label -> built method
    unavailable: Dict[str, str]  # label -> reason
    image_paths: Dict[str, Path]
    gallery_size: int
    thresholds: Dict[str, Optional[float]] = field(default_factory=dict)  # label -> cutoff
    threshold_notes: Dict[str, str] = field(default_factory=dict)  # label -> why missing
    load_log: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers

    def labels(self) -> List[str]:
        return list(self.methods)

    def describe(self, label: str) -> str:
        built = self.methods.get(label)
        if built is None:
            return self.unavailable.get(label, "Unknown method.")
        params = {k: v for k, v in built.parameters.items() if k not in ("index_parameters",)}
        lines = [f"**{built.spec.label}** · family `{built.spec.family}` · id `{built.spec.name}`", "", built.spec.description, ""]
        if params:
            pretty = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items()))
            lines.append(f"Parameters: {pretty}")
        lines.append(f"Index/build time: {built.build_seconds * 1000:.1f} ms")
        threshold = self.thresholds.get(label)
        if threshold is None:
            lines.append(f"Score threshold: none available ({self.threshold_notes.get(label, 'no file')})")
        else:
            source = built.spec.threshold_method
            via = "own" if source == built.spec.name else f"from `{source}`"
            lines.append(f"Score threshold: `{threshold:.4f}` ({via}, keeps {self.config.threshold_target_recall:.0%} of validation relevant images)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ search

    def search(self, query: str, label: str, apply_threshold: bool = True):
        text = query.strip()
        if not text:
            raise ValueError("Enter a text description before searching.")
        built = self.methods.get(label)
        if built is None:
            raise ValueError(self.unavailable.get(label, f"Method {label!r} is not available."))

        started = time.perf_counter()
        query_vector = self.encoder.encode_texts([text])[0]
        encoded = time.perf_counter()
        results = built.retriever.search(query_vector, top_k=TOP_K)
        finished = time.perf_counter()

        threshold = self.thresholds.get(label) if apply_threshold else None
        kept, _dropped = apply_score_threshold(results, threshold)

        gallery = []
        crop_gallery = []
        rows = []
        for rank, result in kept:
            image_path = self.image_paths[result.image_id]
            gallery.append((str(image_path), f"#{rank}  score={result.score:.4f}"))
            rows.append(
                [
                    rank,
                    result.image_id,
                    round(result.score, 6),
                    round(result.global_score, 6),
                    None if result.crop_score is None else round(result.crop_score, 6),
                    result.winning_crop or "",
                ]
            )
            if result.winning_crop:
                with Image.open(image_path) as image:
                    crop = dict(five_crops(image.convert("RGB"), self.config.crop_ratio))[
                        result.winning_crop
                    ]
                crop_gallery.append((crop, f"#{rank}  {result.winning_crop.replace('_', ' ')}"))

        timing = (
            f"**{built.spec.label}** — text encoding {1000 * (encoded - started):.1f} ms · "
            f"search {1000 * (finished - encoded):.2f} ms · "
            f"total {1000 * (finished - started):.1f} ms · gallery {self.gallery_size} images"
        )
        source = built.spec.threshold_method
        status = threshold_status(
            built.spec.label,
            self.thresholds.get(label),
            "" if source == built.spec.name else source,
            self.config.threshold_target_recall,
            len(kept),
            len(results),
            apply_threshold,
            self.threshold_notes.get(label, ""),
        )
        return gallery, rows, crop_gallery, timing, status


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int) and value > 10_000:
        return f"{value:,}"
    return str(value)


def load_demo_runtime(config_path: Path, device: str = None, methods=None) -> DemoRuntime:
    """Load model, dataset and cache once, then build every registered method."""
    config = load_config(config_path)
    images, queries = load_split(config, "test")
    validation_images, validation_queries = load_split(config, "validation")
    validation_hash = manifest_sha256(validation_images, validation_queries)
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=device,
        cache_dir=config.checkpoint_cache_dir,
    )

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
    context = BuildContext(config, cache, encoder.checkpoint_revision)

    built_methods: Dict[str, BuiltMethod] = {}
    unavailable: Dict[str, str] = {}
    thresholds: Dict[str, Optional[float]] = {}
    threshold_notes: Dict[str, str] = {}
    log: List[str] = []
    for name in methods or list_methods():
        spec = get_method(name)
        try:
            built = build_method(name, context)
        except Exception as exc:  # keep the demo alive; report the reason in the UI
            unavailable[spec.label] = f"Unavailable: {exc}"
            log.append(f"[skip] {name}: {exc}")
            continue
        built_methods[built.spec.label] = built
        log.append(f"[ok]   {name}: built in {built.build_seconds * 1000:.1f} ms")
        try:
            path = threshold_path(config, spec)
            thresholds[spec.label] = load_threshold(
                path,
                validation_manifest_sha256=validation_hash,
                checkpoint_revision=encoder.checkpoint_revision,
                target_recall=config.threshold_target_recall,
                alpha=float(built.parameters.get("alpha", 0.0)),
            )
            log.append(f"[thr]  {name}: {thresholds[spec.label]:.6f} from {path.name}")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            thresholds[spec.label] = None
            threshold_notes[spec.label] = str(exc).split(";")[0]
            log.append(f"[thr]  {name}: none ({exc})")
    if not built_methods:
        raise RuntimeError("no retrieval method could be built:\n" + "\n".join(log))

    return DemoRuntime(
        config=config,
        encoder=encoder,
        methods=built_methods,
        unavailable=unavailable,
        image_paths={image.image_id: image.path for image in images},
        gallery_size=len(images),
        thresholds=thresholds,
        threshold_notes=threshold_notes,
        load_log=log,
    )


def create_demo(config_path: Path, device: str = None):
    """Create the Gradio interface after loading the shared runtime."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required; install the project dependencies") from exc

    runtime = load_demo_runtime(config_path, device=device)
    for line in runtime.load_log:
        print(line, flush=True)

    labels = runtime.labels()
    default = labels[0]

    def search(query, label, apply_threshold):
        try:
            return runtime.search(query, label, bool(apply_threshold))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def describe(label):
        return runtime.describe(label)

    unavailable_note = ""
    if runtime.unavailable:
        unavailable_note = "\n\nNot available in this run: " + "; ".join(
            f"{label} ({reason})" for label, reason in runtime.unavailable.items()
        )

    with gr.Blocks(title="SigLIP 2 Flickr8k Image Search") as demo:
        gr.Markdown(
            "# SigLIP 2 Flickr8k Image Search\n"
            f"Same frozen `{runtime.config.checkpoint_id}` embeddings for every method; "
            f"gallery = {runtime.gallery_size} Flickr8k test images. Pick a method, type a "
            "description, compare results and timing. The threshold checkbox hides results "
            f"below the method's validation-selected cutoff (keeps "
            f"{runtime.config.threshold_target_recall:.0%} of validation relevant images)."
            + unavailable_note
        )
        with gr.Row():
            query = gr.Textbox(
                label="Text query",
                placeholder="Example: a dog running through grass",
                scale=4,
            )
            method = gr.Dropdown(choices=labels, value=default, label="Method", scale=2)
            apply_threshold = gr.Checkbox(
                value=True, label="Apply validation-selected score threshold", scale=2
            )
        description = gr.Markdown(runtime.describe(default))
        search_button = gr.Button("Search", variant="primary")
        timing = gr.Markdown()
        status = gr.Markdown()
        results = gr.Gallery(label="Ranked results", columns=5, height="auto")
        scores = gr.Dataframe(
            headers=["Rank", "Image ID", "Score", "Global score", "Best crop score", "Winning crop"],
            datatype=["number", "str", "number", "number", "number", "str"],
            interactive=False,
            label="Scores",
        )
        crops = gr.Gallery(label="Winning regions (five-crop only)", columns=5, height="auto")

        outputs = [results, scores, crops, timing, status]
        inputs = [query, method, apply_threshold]
        method.change(describe, inputs=[method], outputs=[description])
        search_button.click(search, inputs=inputs, outputs=outputs)
        query.submit(search, inputs=inputs, outputs=outputs)
    return demo
