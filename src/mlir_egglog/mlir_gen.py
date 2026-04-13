from textwrap import indent
from typing import Callable
import platform
import llvmlite.binding as llvm
from mlir_egglog import term_ir as ir
from mlir_egglog.llvm_runtime import init_llvm
from egglog import f64, get_callable_args, String, i64

KERNEL_NAME = "kernel_worker"
F32_TYPE = "f32"
I64_TYPE = "i64"
KERNEL_INDENT = "    " * 2
MODULE_INDENT = "  "


def get_target_info():
    init_llvm()

    # Get the default triple for the current system
    triple = llvm.get_default_triple()

    # Create target and target machine to get the data layout
    target = llvm.Target.from_triple(triple)
    target_machine = target.create_target_machine()
    layout = str(target_machine.target_data)

    return triple, layout


# Module wrapper with LLVM target information
def get_module_prologue():
    """Generate module prologue with target triple and data layout from runtime system."""
    triple, layout = get_target_info()
    return f"""module attributes {{llvm.data_layout = "{layout}",
                   llvm.target_triple = "{triple}"}} {{
"""


module_epilogue = """
}
"""

# Numpy vectorized kernel that supports N-dimensional arrays
kernel_prologue = f"""
func.func @{KERNEL_NAME}(
    %arg0: memref<?xf32>,
    %arg1: memref<?xf32>
) attributes {{llvm.emit_c_interface}} {{
    %c0 = index.constant 0

    // Get dimension of input array
    %dim = memref.dim %arg0, %c0 : memref<?xf32>

    // Process each element in a flattened manner
    affine.for %idx = %c0 to %dim {{
"""

kernel_epilogue = """
    }
    return
}
"""


def generate_maximum_mlir(lhs_val: str, rhs_val: str, result_var: str) -> list[str]:
    """
    Generate MLIR code for maximum operation with architecture-specific handling.
    ARM: Use arith.maximumf (supported)
    x86: Use arith.cmpf + arith.select (fallback for compatibility)
    """
    cpu_arch = platform.machine().lower()

    # ARM architectures (including Apple Silicon) support arith.maximumf
    if cpu_arch in ("arm64", "aarch64"):
        return [f"{result_var} = arith.maximumf {lhs_val}, {rhs_val} : {F32_TYPE}"]
    else:
        # x86 and other architectures use cmpf + select fallback
        cmp_var = result_var.replace("max_", "cmp_")
        return [
            f"{cmp_var} = arith.cmpf ogt, {lhs_val}, {rhs_val} : {F32_TYPE}",
            f"{result_var} = arith.select {cmp_var}, {lhs_val}, {rhs_val} : {F32_TYPE}",
        ]


class MLIRGen:
    """
    Generate textual MLIR from a symbolic expression.
    """

    root: ir.Term
    cache: dict[ir.Term, str]
    vars: list[str]  # local variables
    temp_counter: int  # Counter for generating unique variable names

    def __init__(self, root: ir.Term, argmap: dict[str, str]):
        # Use the keys from argmap as the variable names
        self.root = root
        self.cache = {}
        self.vars = list(argmap.keys())
        self.temp_counter = 0

    def generate(self) -> str:
        """
        Generate MLIR code for the root expression.
        """
        subexprs = list(self.unfold(self.root))
        subexprs.sort(key=lambda x: len(str(x)))

        buf = []
        # First load input arguments from memref
        for var in self.vars:
            buf.append(f"%arg_{var} = affine.load %arg0[%idx] : memref<?xf32>")

        for i, subex in enumerate(subexprs):
            # Skip if this is just a variable reference
            match get_callable_args(subex, ir.Term.var):
                case (String(var_name),) if var_name in self.vars:
                    continue

            # Handle Maximum operations specially for multi-operation Linux case
            match get_callable_args(subex, ir.maximum):
                case (lhs, rhs):
                    # Process operands
                    self.walk(lhs)
                    self.walk(rhs)

                    # Generate platform-specific MLIR
                    self.temp_counter += 1
                    var_name = f"%max_{self.temp_counter}"
                    lhs_val = self.cache[lhs]
                    rhs_val = self.cache[rhs]

                    max_ops = generate_maximum_mlir(lhs_val, rhs_val, var_name)
                    buf.extend(max_ops)

                    self.cache[subex] = var_name
                    continue

            # Recurse and cache the subexpression
            self.walk(subex)
            orig_expr = self.cache[subex]

            # Generate a unique name for the subexpression
            var_name = f"%v{i}"
            self.cache[subex] = var_name

            # Append the subexpression to the buffer
            buf.append(f"{var_name} = {orig_expr}")

        self.walk(self.root)
        res = self.cache[self.root]

        # Handle the output
        buf.append(f"affine.store {res}, %arg1[%idx] : memref<?xf32>")

        # Format the kernel body
        kernel_body = indent("\n".join(buf), KERNEL_INDENT)
        kernel_code = kernel_prologue + kernel_body + kernel_epilogue

        # Wrap kernel in module with target information
        return (
            get_module_prologue() + indent(kernel_code, MODULE_INDENT) + module_epilogue
        )

    def unfold(self, expr: ir.Term) -> set[ir.Term]:
        """
        Unfold an expression into a set of subexpressions.
        """
        visited = set()
        all_subexprs = set()
        to_visit = [expr]
        while to_visit:
            current = to_visit.pop()
            all_subexprs.add(current)
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(get_children(current))

        return all_subexprs

    def walk(self, expr: ir.Term):
        """
        Walk an expression recursively and generate MLIR code for subexpressions,
        caching the intermediate expressions in a lookup table.
        """
        if expr in self.cache:
            return

        def lookup(e):
            return self.cache.get(e) or as_source(e, self.vars, lookup)

        self.cache[expr] = as_source(expr, self.vars, lookup)


def get_children(expr: ir.Term) -> set[ir.Term]:
    """Get child expressions for an AST node."""
    return {child for child in get_callable_args(expr) if isinstance(child, ir.Term)}


def as_source(
    expr: ir.Term, vars: list[str], lookup_fn: Callable[[ir.Term], str]
) -> str:
    """
    Convert expressions to MLIR source code using arith and math dialects.
    """
    # Literals and Symbols
    match get_callable_args(expr, ir.Term.lit_f32):
        case (f64(f),):
            return f"arith.constant {f:e} : {F32_TYPE}"
    match get_callable_args(expr, ir.Term.lit_i64):
        case (i64(i),):
            return f"arith.constant {i} : {I64_TYPE}"
    match get_callable_args(expr, ir.Term.var):
        case (String(var_name),):
            return f"%arg_{var_name}" if var_name in vars else f"%{var_name}"
    match get_callable_args(expr, ir.Term.__add__):
        case (lhs, rhs):
            return f"arith.addf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
    match get_callable_args(expr, ir.Term.__mul__):
        case (lhs, rhs):
            return f"arith.mulf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
    match get_callable_args(expr, ir.Term.__truediv__):
        case (lhs, rhs):
            return f"arith.divf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
    match get_callable_args(expr, ir.maximum):
        case (lhs, rhs):
            # Maximum is handled in the generate() method for multi-operation support
            return "ERROR_MAXIMUM_HANDLED_IN_GENERATE"
    match get_callable_args(expr, ir.sin):
        case (arg,):
            return f"math.sin {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.cos):
        case (arg,):
            return f"math.cos {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.log):
        case (arg,):
            return f"math.log {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.sqrt):
        case (arg,):
            return f"math.sqrt {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.exp):
        case (arg,):
            return f"math.exp {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.sinh):
        case (arg,):
            return f"math.sinh {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.cosh):
        case (arg,):
            return f"math.cosh {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.tanh):
        case (arg,):
            return f"math.tanh {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.Term.__neg__):
        case (arg,):
            return f"arith.negf {lookup_fn(arg)} : {F32_TYPE}"
    match get_callable_args(expr, ir.astype):
        case (arg, ir.DType("float32")):
            return f"arith.sitofp {lookup_fn(arg)} : {I64_TYPE} to {F32_TYPE}"
        case (arg, ir.DType("int64")):
            return f"arith.fptosi {lookup_fn(arg)} : {F32_TYPE} to {I64_TYPE}"

    raise NotImplementedError(
        f"Unsupported expression type: {type(expr)}\n"
        f"Expression: {expr}\n"
        f"Supported types: literals, variables, +, *, /, max, sin, cos, log, sqrt, exp, sinh, cosh, tanh, ~, astype\n"
        f"Hint: If you're using a NumPy function, make sure it's supported by the compiler. "
        f"Check term_ir.py for available operations."
    )
