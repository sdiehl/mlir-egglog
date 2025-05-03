from textwrap import indent
from typing import Callable
import llvmlite.binding as llvm
from mlir_egglog import expr_model as ir
from mlir_egglog.llvm_runtime import init_llvm

KERNEL_NAME = "kernel_worker"


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


class MLIRGen:
    """
    Generate textual MLIR from a symbolic expression.
    """

    root: ir.Expr
    cache: dict[ir.Expr, str]
    subexprs: dict[str, str]
    vars: list[str]  # local variables

    def __init__(self, root: ir.Expr, argmap: dict[str, str]):
        # Use the keys from argmap as the variable names
        self.root = root
        self.cache = {}
        self.vars = list(argmap.keys())
        self.subexprs = {}

    def generate(self):
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

            # Recurse and cache the subexpression
            self.walk(subex)
            orig = self.cache[subex]

            # Generate a unique name for the subexpression
            k = f"%v{i}"
            self.cache[subex] = k
            self.subexprs[k] = orig

            # Append the subexpression to the buffer
            buf.append(f"{k} = {orig}")

        self.walk(self.root)
        res = self.cache[self.root]

        # Handle the output
        buf.append(f"affine.store {res}, %arg1[%idx] : memref<?xf32>")

        # Format the kernel body
        kernel_body = indent("\n".join(buf), "    " * 2)
        kernel_code = kernel_prologue + kernel_body + kernel_epilogue

        # Wrap kernel in module with target information
        return get_module_prologue() + indent(kernel_code, "  ") + module_epilogue

    def unfold(self, expr: ir.Expr):
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


def get_children(expr: ir.Expr):
    """Get child expressions for an AST node."""
    match expr:
        case ir.BinaryOp():
            return {expr.lhs, expr.rhs}
        case ir.UnaryOp():
            return {expr.operand}
        case ir.FloatLiteral() | ir.IntLiteral() | ir.Symbol():
            return set()
        case _:
            raise NotImplementedError(f"Unsupported expression type: {type(expr)}")


def as_source(
    expr: ir.Expr, vars: list[str], lookup_fn: Callable[[ir.Expr], str]
) -> str:
    """
    Convert expressions to MLIR source code using arith and math dialects.
    """
    match expr:
        # Literals and Symbols
        case ir.FloatLiteral(fval=val):
            return f"arith.constant {val:e} : f32"
        case ir.IntLiteral(ival=val):
            return f"arith.constant {val} : i32"
        case ir.Symbol(name=name) if name in vars:
            return f"%arg_{name}"
        case ir.Symbol(name=name):
            return f"%{name}"

        # Binary Operations
        case ir.Add(lhs=lhs, rhs=rhs):
            return f"arith.addf {lookup_fn(lhs)}, {lookup_fn(rhs)} : f32"
        case ir.Mul(lhs=lhs, rhs=rhs):
            return f"arith.mulf {lookup_fn(lhs)}, {lookup_fn(rhs)} : f32"
        case ir.Div(lhs=lhs, rhs=rhs):
            return f"arith.divf {lookup_fn(lhs)}, {lookup_fn(rhs)} : f32"
        case ir.Maximum(lhs=lhs, rhs=rhs):
            return f"arith.maximumf {lookup_fn(lhs)}, {lookup_fn(rhs)} : f32"

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
            return f"math.{op_name} {lookup_fn(op.operand)} : f32"
        case ir.Neg(operand=op):
            return f"arith.negf {lookup_fn(op)} : f32"

        # Type Casting
        case ir.CastF32(operand=op):
            return f"arith.sitofp {lookup_fn(op)} : i64 to f32"
        case ir.CastI64(operand=op):
            return f"arith.fptosi {lookup_fn(op)} : f32 to i64"

        case _:
            raise NotImplementedError(f"Unsupported expression type: {type(expr)}")
