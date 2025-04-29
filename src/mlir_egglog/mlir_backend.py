from tempfile import NamedTemporaryFile
import subprocess
import enum


class Target(enum.Enum):
    OPENMP = "openmp"
    BASIC_LOOPS = "loops"


# MLIR_BIN = "/Users/sdiehl/Downloads/bin/mlir-opt"
# MLIR_TRANSLATE_BIN = "/Users/sdiehl/Downloads/bin/mlir-translate"

MLIR_BIN = "mlir-opt"
MLIR_TRANSLATE_BIN = "mlir-translate"

# Debug options for MLIR compilation
DEBUG_OPTIONS = (
    "--mlir-print-debuginfo",
    "--mlir-print-ir-after-all",
    "--debug-pass=Details",
)

# MLIR to LLVM dialect conversion options - common options used by both paths
COMMON_MLIR_TO_LLVM_OPTIONS = (
    "--debugify-level=locations",
    "--snapshot-op-locations",
    "--inline",
    "-affine-loop-normalize",
    "-affine-parallelize",
    "-affine-super-vectorize",
    "--affine-scalrep",
    "-lower-affine",
    "-convert-vector-to-scf",
    "-convert-linalg-to-loops",
    "-lower-affine",
)

# Common initial transformations for both paths
COMMON_INITIAL_OPTIONS = (
    "--debugify-level=locations",
    "--snapshot-op-locations",
    "--inline",
    "-affine-loop-normalize",
    "-affine-parallelize",
    "-affine-super-vectorize",
    "--affine-scalrep",
    "-lower-affine",
    "-convert-vector-to-scf",
    "-convert-linalg-to-loops",
    "-lower-affine",
)

# OpenMP lowering sequence
OPENMP_OPTIONS = (
    "-convert-scf-to-openmp",
    "-convert-scf-to-cf",
    "-cse",
    "-convert-openmp-to-llvm",
    "-convert-vector-to-llvm",
    "-convert-math-to-llvm",
    "-expand-strided-metadata",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-convert-index-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
    "--llvm-request-c-wrappers",
)

# Basic loops lowering sequence
BASIC_LOOPS_OPTIONS = (
    "-convert-scf-to-cf",
    "-cse",
    "-convert-vector-to-llvm",
    "-convert-math-to-llvm",
    "-expand-strided-metadata",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-convert-index-to-llvm",
    "-convert-arith-to-llvm",
    "-convert-cf-to-llvm",
    "-reconcile-unrealized-casts",
    "--llvm-request-c-wrappers",
)

# MLIR to LLVM IR translation options
MLIR_TRANSLATE_OPTIONS = (
    "--mlir-print-local-scope",
    "--mlir-print-debuginfo=false",
    "--print-after-all",
    "--mlir-to-llvmir",
    "--verify-diagnostics",
)


class MLIRCompiler:

    def __init__(self, debug=False):
        self._debug = debug

    def to_llvm_dialect(self, mlir_src: str, target: Target = Target.OPENMP) -> str:
        """
        Convert MLIR to LLVM dialect.

        Args:
            mlir_src: The MLIR source code
            target: Target compilation mode (openmp or basic_loops)
        """
        if self._debug:
            print(mlir_src)
        binary = (MLIR_BIN,)
        dbg_cmd = DEBUG_OPTIONS if self._debug else ()

        # Choose compilation path based on target
        target_options = (
            OPENMP_OPTIONS if target == Target.OPENMP else BASIC_LOOPS_OPTIONS
        )
        shell_cmd = binary + dbg_cmd + COMMON_INITIAL_OPTIONS + target_options
        return self._run_shell(shell_cmd, "t", "t", mlir_src)

    def mlir_translate_to_llvm_ir(self, mlir_src):
        if self._debug:
            print(mlir_src)
        binary = (MLIR_TRANSLATE_BIN,)
        shell_cmd = binary + MLIR_TRANSLATE_OPTIONS
        return self._run_shell(shell_cmd, "t", "t", mlir_src)

    def llvm_ir_to_bitcode(self, llvmir_src):
        if self._debug:
            print(llvmir_src)
        binary = ("llvm-as",)
        shell_cmd = binary
        return self._run_shell(shell_cmd, "t", "b", llvmir_src)

    def _run_shell(self, cmd, in_mode, out_mode, src):
        assert in_mode in "tb"
        assert out_mode in "tb"

        with (
            NamedTemporaryFile(mode=f"w{in_mode}") as src_file,
            NamedTemporaryFile(mode=f"r{out_mode}") as out_file,
        ):
            src_file.write(src)
            src_file.flush()

            shell_cmd = *cmd, src_file.name, "-o", out_file.name
            if self._debug:
                print(shell_cmd)
            subprocess.run(shell_cmd)
            out_file.flush()
            return out_file.read()
