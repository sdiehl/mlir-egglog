from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FunctionType

from egglog import RewriteOrRule, Ruleset

from mlir_egglog.egglog_optimizer import compile, OPTS
from mlir_egglog.mlir_backend import MLIRCompiler, Target
from mlir_egglog.mlir_gen import KERNEL_NAME

LLC = "llc"
CC = "cc"


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
        omppath = find_omp_path()
        ctypes.CDLL(omppath, mode=os.RTLD_NOW)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._libs: list[ctypes.CDLL] = []
        self._kernel_count = 0

    def run_frontend(
        self,
        fn: FunctionType,
        rewrites: tuple[RewriteOrRule | Ruleset, ...] | None = None,
    ) -> str:
        actual_rewrites = rewrites if rewrites is not None else OPTS
        return compile(fn, rewrites=actual_rewrites, debug=False)

    def run_backend(self, mlir_src: str) -> int:
        mlir_compiler = MLIRCompiler(debug=False)
        mlir_llvm = mlir_compiler.to_llvm_dialect(mlir_src, target=Target.BASIC_LOOPS)
        llvm_ir = mlir_compiler.mlir_translate_to_llvm_ir(mlir_llvm)

        self._kernel_count += 1
        base = Path(self._tmpdir.name) / f"kernel_{self._kernel_count}"
        ir_path = base.with_suffix(".ll")
        obj_path = base.with_suffix(".o")
        so_path = base.with_suffix(".so")

        ir_path.write_text(llvm_ir)

        subprocess.run(
            [
                LLC,
                "-filetype=obj",
                "-relocation-model=pic",
                str(ir_path),
                "-o",
                str(obj_path),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [CC, "-shared", "-fPIC", str(obj_path), "-o", str(so_path), "-lm"],
            check=True,
            capture_output=True,
        )

        lib = ctypes.CDLL(str(so_path))
        self._libs.append(lib)

        func = getattr(lib, f"_mlir_ciface_{KERNEL_NAME}")
        address = ctypes.cast(func, ctypes.c_void_p).value
        assert address, f"Failed to resolve address for _mlir_ciface_{KERNEL_NAME}"
        return address

    def jit_compile(
        self,
        fn: FunctionType,
        rewrites: tuple[RewriteOrRule | Ruleset, ...] | None = None,
    ) -> int:
        mlir = self.run_frontend(fn, rewrites)
        return self.run_backend(mlir)
