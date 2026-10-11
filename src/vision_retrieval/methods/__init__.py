"""Retrieval methods and the registry every entry point uses.

``from vision_retrieval.methods import list_methods, build_method, ...`` is the only API
scripts and the demo need. Importing this package registers the built-in presets in
dropdown order: ``global``, ``five_crop`` (``builtin.py``), ``faiss_flat``, ``faiss_ivf``
(``faiss_retrievers.py``). Add a method by writing a module next to those and importing
it here.
"""

from .registry import (
    BuildContext,
    Builder,
    BuiltMethod,
    MethodSpec,
    build_method,
    get_method,
    list_methods,
    method_labels,
    register,
)
from .retrievers import FiveCropRetriever, GlobalRetriever, SearchResult, select_alpha
from . import builtin as _builtin  # noqa: F401  registers global, five_crop
from . import faiss_retrievers as _faiss_retrievers  # noqa: F401  registers faiss_flat, faiss_ivf
from .builtin import load_selected_alpha
from .faiss_retrievers import FaissRetriever, register_faiss_method

__all__ = [
    "BuildContext",
    "Builder",
    "BuiltMethod",
    "FaissRetriever",
    "FiveCropRetriever",
    "GlobalRetriever",
    "MethodSpec",
    "SearchResult",
    "build_method",
    "get_method",
    "list_methods",
    "load_selected_alpha",
    "method_labels",
    "register",
    "register_faiss_method",
    "select_alpha",
]
