from textwrap import indent
from typing import Callable
import platform
import llvmlite.binding as llvm
from mlir_egglog import expr_model as ir
from mlir_egglog.llvm_runtime import init_llvm

KERNEL_NAME = "kernel_worker"
F32_TYPE = "f32"
I32_TYPE = "i32"
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

    root: ir.Expr
    cache: dict[ir.Expr, str]
    vars: list[str]  # local variables
    temp_counter: int  # Counter for generating unique variable names

    def __init__(self, root: ir.Expr, argmap: dict[str, str]):
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
            if isinstance(subex, ir.Symbol) and subex.name in self.vars:
                continue

            # Handle Maximum operations specially for multi-operation Linux case
            if isinstance(subex, ir.Maximum):
                # Process operands
                self.walk(subex.lhs)
                self.walk(subex.rhs)

                # Generate platform-specific MLIR
                self.temp_counter += 1
                var_name = f"%max_{self.temp_counter}"
                lhs_val = self.cache[subex.lhs]
                rhs_val = self.cache[subex.rhs]

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

    def unfold(self, expr: ir.Expr) -> set[ir.Expr]:
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

    def walk(self, expr: ir.Expr):
        """
        Walk an expression recursively and generate MLIR code for subexpressions,
        caching the intermediate expressions in a lookup table.
        """
        if expr in self.cache:
            return

        def lookup(e):
            return self.cache.get(e) or as_source(e, self.vars, lookup)

        self.cache[expr] = as_source(expr, self.vars, lookup)


def get_children(expr: ir.Expr) -> set[ir.Expr]:
    """Get child expressions for an AST node."""
    match expr:
        case ir.BinaryOp():
            return {expr.lhs, expr.rhs}
        case ir.UnaryOp():
            return {expr.operand}
        case ir.FloatLiteral() | ir.IntLiteral() | ir.Symbol():
            return set()
        case _:
            raise NotImplementedError(
                f"Cannot get children for expression type: {type(expr)}\n"
                f"Expression: {expr}\n"
                f"Supported types: BinaryOp (Add, Mul, Div, Maximum), UnaryOp (Sin, Cos, etc.), "
                f"FloatLiteral, IntLiteral, Symbol\n"
                f"This error typically occurs when trying to process an unsupported operation."
            )


def as_source(
    expr: ir.Expr, vars: list[str], lookup_fn: Callable[[ir.Expr], str]
) -> str:
    """
    Convert expressions to MLIR source code using arith and math dialects.
    """
    match expr:
        # Literals and Symbols
        case ir.FloatLiteral(fval=val):
            return f"arith.constant {val:e} : {F32_TYPE}"
        case ir.IntLiteral(ival=val):
            return f"arith.constant {val} : {I32_TYPE}"
        case ir.Symbol(name=name) if name in vars:
            return f"%arg_{name}"
        case ir.Symbol(name=name):
            return f"%{name}"

        # Binary Operations
        case ir.Add(lhs=lhs, rhs=rhs):
            return f"arith.addf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
        case ir.Mul(lhs=lhs, rhs=rhs):
            return f"arith.mulf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
        case ir.Div(lhs=lhs, rhs=rhs):
            return f"arith.divf {lookup_fn(lhs)}, {lookup_fn(rhs)} : {F32_TYPE}"
        case ir.Maximum(lhs=lhs, rhs=rhs):
            # Maximum is handled in the generate() method for multi-operation support
            return "ERROR_MAXIMUM_HANDLED_IN_GENERATE"

        # Unary Math Operations
        case (
            ir.Sin()
            | ir.Cos()
            | ir.Log()
            | ir.Sqrt()
            | ir.Exp()
            | ir.Sinh()
            | ir.Cosh()
            | ir.Tanh()
        ) as op:
            op_name = type(op).__name__.lower()
            return f"math.{op_name} {lookup_fn(op.operand)} : {F32_TYPE}"
        case ir.Neg(operand=op):
            return f"arith.negf {lookup_fn(op)} : {F32_TYPE}"

        # Type Casting
        case ir.CastF32(operand=op):
            return f"arith.sitofp {lookup_fn(op)} : {I64_TYPE} to {F32_TYPE}"
        case ir.CastI64(operand=op):
            return f"arith.fptosi {lookup_fn(op)} : {F32_TYPE} to {I64_TYPE}"

        case _:
            raise NotImplementedError(
                f"Unsupported expression type: {type(expr)}\n"
                f"Expression: {expr}\n"
                f"Supported types: FloatLiteral, IntLiteral, Symbol, Add, Mul, Div, Maximum, "
                f"Sin, Cos, Log, Sqrt, Exp, Sinh, Cosh, Tanh, Neg, CastF32, CastI64\n"
                f"Hint: If you're using a NumPy function, make sure it's supported by the compiler. "
                f"Check builtin_functions.py for available operations."
            )
