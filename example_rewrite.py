import llvmlite
import numpy as np

llvmlite.opaque_pointers_enabled = True

from egglog import rewrite, ruleset

from mlir_egglog import kernel
from mlir_egglog.optimization_rules import basic_math
from mlir_egglog.term_ir import Term, cos, sin


# A rewrite rule
@ruleset
def trig_double_angle(a: Term):
    sin_a = sin(a)
    cos_a = cos(a)
    mul1 = sin_a * cos_a
    mul2 = cos_a * sin_a

    # sin(a)*cos(a) + cos(a)*sin(a) -> 2 * sin(a)*cos(a)
    yield rewrite(mul1 + mul2).to(Term.lit_f32(2.0) * mul1)


# Apply the rewrites
@kernel("float32(float32)", rewrites=(basic_math, trig_double_angle))
def fn(a):
    return np.sin(a) * np.cos(a) + np.cos(a) * np.sin(a)


def ref_fn(a):
    return np.sin(a) * np.cos(a) + np.cos(a) * np.sin(a)


# Observe that the output LLVM IR is optimized.
out = fn(np.array([1.0], dtype=np.float32))
ref = ref_fn(np.array([1.0], dtype=np.float32))
print(out)
print(ref)

assert np.allclose(out, ref)
