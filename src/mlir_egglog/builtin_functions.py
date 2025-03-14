# A mock NumPy namespace that we convert into our own expression model

import math
from mlir_egglog.expr_model import (
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    tanh,
    sinh,
    cosh,
    sqrt,
    exp,
    log,
    log10,
    log2,
    float32,
    int64,
    maximum,
)  # noq

# Constants
e = math.e
pi = math.pi


# Define abs function
def abs(x):
    return maximum(x, -x)


def relu(x):
    return maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


__all__ = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "tanh",
    "sinh",
    "cosh",
    "sqrt",
    "exp",
    "log",
    "log10",
    "log2",
    "float32",
    "int64",
    "e",
    "pi",
    "maximum",
    "abs",
]
