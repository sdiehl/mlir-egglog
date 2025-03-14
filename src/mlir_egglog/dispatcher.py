from __future__ import annotations

import ctypes
import types
import numpy as np
from egglog import RewriteOrRule

from mlir_egglog.jit_engine import JITEngine
from mlir_egglog.memory_descriptors import as_memref_descriptor


class Dispatcher:
    """
    Dispatch a python function to a compiled vectorized kernel.
    """

    _compiled_func: bytes | None
    _compiler: JITEngine | None
    py_func: types.FunctionType | types.MethodType
    rewrites: tuple[RewriteOrRule, ...] | None

    def __init__(
        self,
        py_func: types.FunctionType,
        rewrites: tuple[RewriteOrRule, ...] | None = None,
    ):
        self.py_func = py_func
        self._compiled_func = None
        self._compiler = None
        self.rewrites = rewrites

    def compile(self):
        self._compiler = JITEngine()
        binary = self._compiler.jit_compile(self.py_func, self.rewrites)
        self._compiled_func = binary
        return binary

    def __call__(self, *args, **kwargs):
        assert not kwargs
        if self._compiled_func is None:
            raise RuntimeError("Function must be compiled before calling")

        # Get the input array and its shape
        input_array = args[0]
        original_shape = input_array.shape

        # Flatten the input array
        flattened_input = input_array.flatten()

        # Create a flattened result array
        flattened_output = np.empty_like(flattened_input)

        # Convert to memrefs
        memrefs = [
            as_memref_descriptor(flattened_input, ctypes.c_float),
            as_memref_descriptor(flattened_output, ctypes.c_float),
        ]

        # Create a prototype for the C function
        prototype = ctypes.CFUNCTYPE(None, *[ctypes.POINTER(type(x)) for x in memrefs])

        # Call the compiled function
        cfunc = prototype(self._compiled_func)
        cfunc(*[ctypes.byref(x) for x in memrefs])

        # Reshape the output to match the input shape
        output = flattened_output.reshape(original_shape)
        return output


def kernel(func_or_sig, rewrites: tuple[RewriteOrRule, ...] | None = None):
    """
    Decorator to compile a Python function into a vectorized kernel.
    """
    if isinstance(func_or_sig, types.FunctionType):
        wrap = Dispatcher(func_or_sig, rewrites)
    elif isinstance(func_or_sig, str):

        def wrap(py_func: types.FunctionType):
            disp = Dispatcher(py_func, rewrites)
            disp.compile()
            return disp

    else:
        raise TypeError("Expected a python function or a string signature")
    return wrap
