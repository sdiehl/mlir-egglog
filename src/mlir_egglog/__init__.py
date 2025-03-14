"""
MLIR-Egglog: A Python library for optimizing and compiling numerical expressions using MLIR and egglog.
"""

import llvmlite

llvmlite.opaque_pointers_enabled = True

# Version of the mlir-egglog package
__version__ = "0.1.0"

from mlir_egglog.expr_model import (  # noqa: E402
    BinaryOp,
    FloatLiteral,
    IntLiteral,
    Symbol,
)
from mlir_egglog.term_ir import Term, as_egraph  # noqa: E402
from mlir_egglog.dispatcher import kernel  # noqa: E402

__all__ = [
    "BinaryOp",
    "FloatLiteral",
    "IntLiteral",
    "Symbol",
    "Term",
    "as_egraph",
    "kernel",
]
