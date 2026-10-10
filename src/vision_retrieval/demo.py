"""Gradio demo for cached Flickr8k text-to-image retrieval."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

from PIL import Image

from .cache import CacheMetadata, load_embedding_cache, read_cache_metadata
from .config import ExperimentConfig, load_config
from .crops import five_crops
from .data import load_split, manifest_sha256
from .encoder import FrozenSiglip2Encoder
from .retrieval import FiveCropRetriever, GlobalRetriever


GLOBAL_METHOD = "Global baseline"
FIVE_CROP_METHOD = "Five-crop fusion"


@dataclass
class DemoRuntime:
    config: ExperimentConfig
    encoder: FrozenSiglip2Encoder
    global_retriever: GlobalRetriever
    five_crop_retriever: FiveCropRetriever
    image_paths: Dict[str, Path]
    alpha: float

    def search(self, query: str, method: str):
        text = query.strip()
        if not text:
            raise ValueError("Enter a text description before searching.")
        retriever = (
            self.global_retriever if method == GLOBAL_METHOD else self.five_crop_retriever
        )
        results = retriever.search(self.encoder.encode_texts([text])[0], top_k=10)

        gallery = []
        crop_gallery = []
        rows = []
        for rank, result in enumerate(results, start=1):
            image_path = self.image_paths[result.image_id]
            caption = f"#{rank}  {result.image_id}\nscore={result.score:.4f}"
            gallery.append((str(image_path), caption))
            rows.append(
                [
                    rank,
                    result.image_id,
                    result.score,
                    result.global_score,
                    result.crop_score,
                    result.winning_crop or "",
                ]
            )
            if result.winning_crop:
                with Image.open(image_path) as image:
                    crop = dict(five_crops(image.convert("RGB"), self.config.crop_ratio))[
                        result.winning_crop
                    ]
                crop_gallery.append(
                    (crop, f"#{rank}  {result.winning_crop.replace('_', ' ')}")
                )
        return gallery, rows, crop_gallery


def load_demo_runtime(config_path: Path, device: str = None) -> DemoRuntime:
    """Load and validate all model, dataset, and cache assets once."""
    config = load_config(config_path)
    images, queries = load_split(config, "test")
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

    alpha_path = config.output_root / "parameters" / "five_crop_alpha.json"
    if not alpha_path.is_file():
        raise FileNotFoundError(
            f"selected alpha not found: {alpha_path}; run scripts/tune_alpha.py first"
        )
    alpha_data = json.loads(alpha_path.read_text(encoding="utf-8"))
    validation_images, validation_queries = load_split(config, "validation")
    if alpha_data.get("validation_manifest_sha256") != manifest_sha256(
        validation_images, validation_queries
    ):
        raise ValueError("selected alpha does not match the validation manifest")
    if alpha_data.get("checkpoint_revision") != cache.metadata.checkpoint_revision:
        raise ValueError("selected alpha and embedding cache use different checkpoints")
    alpha = float(alpha_data["alpha"])

    return DemoRuntime(
        config=config,
        encoder=encoder,
        global_retriever=GlobalRetriever(
            cache.metadata.image_ids, cache.global_embeddings
        ),
        five_crop_retriever=FiveCropRetriever(
            cache.metadata.image_ids,
            cache.global_embeddings,
            cache.crop_embeddings,
            cache.metadata.crop_names,
            alpha,
        ),
        image_paths={image.image_id: image.path for image in images},
        alpha=alpha,
    )


def create_demo(config_path: Path, device: str = None):
    """Create the Gradio interface after loading the shared runtime."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required; install the project dependencies") from exc

    runtime = load_demo_runtime(config_path, device=device)

    def search(query, method):
        try:
            return runtime.search(query, method)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="SigLIP 2 Flickr8k Image Search") as demo:
        gr.Markdown(
            "# SigLIP 2 Flickr8k Image Search\n"
            "Search the 1,000-image Flickr8k test gallery using the frozen "
            f"`{runtime.config.checkpoint_id}` checkpoint. Five-crop fusion uses "
            f"validation-selected alpha `{runtime.alpha:g}`."
        )
        with gr.Row():
            query = gr.Textbox(
                label="Text query",
                placeholder="Example: a dog running through grass",
                scale=4,
            )
            method = gr.Radio(
                choices=[FIVE_CROP_METHOD, GLOBAL_METHOD],
                value=FIVE_CROP_METHOD,
                label="Method",
                scale=1,
            )
        search_button = gr.Button("Search", variant="primary")
        results = gr.Gallery(label="Top 10 results", columns=5, height="auto")
        scores = gr.Dataframe(
            headers=[
                "Rank",
                "Image ID",
                "Final score",
                "Global score",
                "Best crop score",
                "Winning crop",
            ],
            datatype=["number", "str", "number", "number", "number", "str"],
            interactive=False,
            label="Scores",
        )
        crops = gr.Gallery(
            label="Winning regions (five-crop fusion only)", columns=5, height="auto"
        )
        outputs = [results, scores, crops]
        search_button.click(search, inputs=[query, method], outputs=outputs)
        query.submit(search, inputs=[query, method], outputs=outputs)
    return demo
