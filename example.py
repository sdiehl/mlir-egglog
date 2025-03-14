import llvmlite
import numpy as np  # noqa: E402

llvmlite.opaque_pointers_enabled = True

from mlir_egglog import kernel  # noqa: E402


@kernel("float32(float32)")
def fn(a):
    return np.sin(a) * np.cos(a) + np.cos(a) * np.sin(a)


def ref_fn(a):
    return np.sin(a) * np.cos(a) + np.cos(a) * np.sin(a)


out = fn(np.array([1.0], dtype=np.float32))
ref = ref_fn(np.array([1.0], dtype=np.float32))
print(out)
print(ref)

assert np.allclose(out, ref)
