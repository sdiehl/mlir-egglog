from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from types import FunctionType

import llvmlite.binding as llvm

from mlir_egglog.llvm_runtime import (
    create_execution_engine,
    init_llvm,
    compile_mod,
)
from mlir_egglog.mlir_gen import KERNEL_NAME
from mlir_egglog.mlir_backend import MLIRCompiler, Target
from mlir_egglog.egglog_optimizer import compile


def find_omp_path():
    if sys.platform.startswith("linux"):
        omppath = ctypes.util.find_library("libgomp.so")
    elif sys.platform.startswith("darwin"):
        omppath = ctypes.util.find_library("iomp5")
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
    return omppath


class JITEngine:
    def __init__(self):
        init_llvm()
        omppath = find_omp_path()
        ctypes.CDLL(omppath, mode=os.RTLD_NOW)

        self.ee = create_execution_engine()

    def run_frontend(self, fn: FunctionType) -> str:
        return compile(fn, debug=False)

    def run_backend(self, mlir_src: str) -> bytes:
        mlir_compiler = MLIRCompiler(debug=False)

        mlir_omp = mlir_compiler.to_llvm_dialect(mlir_src, target=Target.BASIC_LOOPS)
        llvm_ir = mlir_compiler.mlir_translate_to_llvm_ir(mlir_omp)

        print(llvm_ir)
        print("Parsing LLVM assembly.")

        try:
            # Clean up the LLVM IR by ensuring proper line endings and formatting
            llvm_ir = llvm_ir.strip()

            # Clean up problematic attribute strings (hack for divergence in modern LLVM IR syntax with old llvmlite)
            llvm_ir = llvm_ir.replace("captures(none)", " ")
            llvm_ir = llvm_ir.replace("memory(argmem: readwrite)", "")
            llvm_ir = llvm_ir.replace("memory(none)", "")
            llvm_ir += "\n"

            mod = llvm.parse_assembly(llvm_ir)
            mod = compile_mod(self.ee, mod)

            # Resolve the function address
            func_name = f"_mlir_ciface_{KERNEL_NAME}"
            address = self.ee.get_function_address(func_name)

            assert address, "Function must be compiled successfully."
            return address
        except Exception as e:
            print(f"Error during LLVM IR parsing/compilation: {str(e)}")
            print("LLVM IR that failed to parse:")
            print(llvm_ir)
            raise

    def jit_compile(self, fn: FunctionType) -> bytes:
        mlir = self.run_frontend(fn)
        address = self.run_backend(mlir)
        return address
