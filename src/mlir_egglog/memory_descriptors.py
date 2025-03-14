import ctypes
from typing import Type, Any
import numpy.typing as npt


def build_struct(ty_ptr, intptr_t, N) -> Type[ctypes.Structure]:
    """
    Build a ctypes structure for a NumPy array of a given element type.
    """

    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ty_ptr),
            ("aligned", ty_ptr),
            ("offset", intptr_t),
            ("sizes", intptr_t * N),
            ("strides", intptr_t * N),
        ]

    return MemRefDescriptor


def as_memref_descriptor(arr: npt.NDArray[Any], ty: Type[Any]) -> ctypes.Structure:
    """
    Convert a numpy array to a memref descriptor
    """
    N = arr.ndim
    ty_ptr = ctypes.POINTER(ty)

    intptr_t = getattr(ctypes, f"c_int{8 * ctypes.sizeof(ctypes.c_void_p)}")
    struct_constructor = build_struct(ty_ptr, intptr_t, N)

    allocated = ctypes.cast(arr.ctypes.data, ty_ptr)
    aligned = allocated
    offset = intptr_t(0)
    sizes = (intptr_t * N)(*arr.shape)
    strides = (intptr_t * N)(*arr.strides)

    # Return the memref descriptor
    return struct_constructor(allocated, aligned, offset, sizes, strides)
