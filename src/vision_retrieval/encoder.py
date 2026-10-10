"""Frozen Hugging Face SigLIP 2 encoder."""

from typing import Any, Optional, Sequence
from pathlib import Path


class FrozenSiglip2Encoder:
    """Encode text and images with a frozen projected SigLIP 2 embedding model."""

    def __init__(
        self,
        checkpoint_id: str,
        checkpoint_revision: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and Transformers are required for SigLIP 2 encoding; "
                "install the project dependencies"
            ) from exc

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        model_source = checkpoint_id
        load_kwargs = {"local_files_only": True}
        if cache_dir and checkpoint_revision:
            snapshot = (
                Path(cache_dir)
                / f"models--{checkpoint_id.replace('/', '--')}"
                / "snapshots"
                / checkpoint_revision
            )
            if snapshot.is_dir():
                model_source = str(snapshot)
            else:
                load_kwargs["cache_dir"] = str(cache_dir)
                load_kwargs["revision"] = checkpoint_revision
        elif checkpoint_revision:
            load_kwargs["revision"] = checkpoint_revision
        self.processor = AutoProcessor.from_pretrained(model_source, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_source, **load_kwargs)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.checkpoint_id = checkpoint_id
        self.checkpoint_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or checkpoint_revision
            or "unresolved"
        )

    def _to_device(self, inputs: Any) -> dict:
        return {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }

    def _normalized_numpy(self, features: Any):
        torch = self._torch
        if not torch.is_tensor(features):
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            else:
                raise TypeError("SigLIP 2 feature method returned an unsupported value")
        if features.ndim != 2:
            raise ValueError(f"expected a 2D embedding tensor; got shape {tuple(features.shape)}")
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        return features.cpu().numpy().astype("float32", copy=False)

    def encode_texts(self, texts: Sequence[str]):
        if not texts:
            raise ValueError("texts must not be empty")
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            features = self.model.get_text_features(**self._to_device(inputs))
        return self._normalized_numpy(features)

    def encode_images(self, images: Sequence[Any]):
        if not images:
            raise ValueError("images must not be empty")
        inputs = self.processor(images=list(images), return_tensors="pt")
        with self._torch.inference_mode():
            features = self.model.get_image_features(**self._to_device(inputs))
        return self._normalized_numpy(features)

