"""Method registry: the single place that maps a method name to a retriever builder.

Every entry point (``scripts/evaluate.py``, ``scripts/query.py``, the Gradio demo)
discovers retrieval methods through this registry, so adding a method means:

1. write a retriever with ``search(query_vector, top_k=None) -> list[SearchResult]``,
2. write a builder ``(BuildContext) -> (retriever, parameters)``,
3. call :func:`register` with a :class:`MethodSpec` (built-ins live in ``builtin.py`` and
   ``faiss_retrievers.py``; ``methods/__init__.py`` imports them so they register on import).

Nothing else has to change. All methods share the same frozen SigLIP 2 embedding
cache, the same gallery, the same queries and the same evaluator, which is what makes
their numbers comparable.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ..config import ExperimentConfig
from ..data.cache import EmbeddingCache


@dataclass(frozen=True)
class BuildContext:
    """Everything a builder may need; the cache is already validated."""

    config: ExperimentConfig
    cache: EmbeddingCache
    checkpoint_revision: str


Builder = Callable[[BuildContext], Tuple[Any, Dict[str, Any]]]


@dataclass(frozen=True)
class MethodSpec:
    name: str  # stable CLI / file identifier, e.g. "faiss_ivf"
    label: str  # human label shown in the demo, e.g. "FAISS IVF (nlist=16, nprobe=2)"
    family: str  # "exact" | "five_crop" | "faiss"
    description: str  # one paragraph a presenter can read aloud
    builder: Builder
    parameters: Dict[str, Any] = field(default_factory=dict)
    uses_crops: bool = False  # True when the method needs the [N, 5, D] crop cache
    # Name of the method whose validation-selected score threshold applies to this one.
    # None means "select my own on validation". Methods that score on the same scale as
    # another method (FAISS vs the exact cosine baseline) point at that method instead,
    # so that one cutoff is selected once and shared. See vision_retrieval.thresholds.
    threshold_source: Optional[str] = None

    @property
    def threshold_method(self) -> str:
        return self.threshold_source or self.name


@dataclass(frozen=True)
class BuiltMethod:
    spec: MethodSpec
    retriever: Any
    parameters: Dict[str, Any]
    build_seconds: float


_REGISTRY: Dict[str, MethodSpec] = {}


def register(spec: MethodSpec, replace: bool = False) -> MethodSpec:
    """Add a method to the registry; names and labels must be unique."""
    if not spec.name or not spec.name.replace("_", "").isalnum():
        raise ValueError(f"method name must be alphanumeric/underscore: {spec.name!r}")
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"method {spec.name!r} is already registered")
    for other in _REGISTRY.values():
        if other.name != spec.name and other.label == spec.label:
            raise ValueError(f"method label {spec.label!r} is already used by {other.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get_method(name: str) -> MethodSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown method {name!r}; registered methods: {', '.join(list_methods())}"
        ) from exc


def list_methods() -> Tuple[str, ...]:
    """Method names in registration order."""
    return tuple(_REGISTRY)


def method_labels() -> Dict[str, str]:
    """Map demo label -> method name."""
    return {spec.label: spec.name for spec in _REGISTRY.values()}


def build_method(name: str, context: BuildContext) -> BuiltMethod:
    """Instantiate a registered method and time its index construction."""
    spec = get_method(name)
    started = time.perf_counter()
    retriever, parameters = spec.builder(context)
    build_seconds = time.perf_counter() - started
    merged = dict(spec.parameters)
    merged.update(parameters)
    return BuiltMethod(spec=spec, retriever=retriever, parameters=merged, build_seconds=build_seconds)
