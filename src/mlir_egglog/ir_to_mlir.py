import ast
from typing import Any

from mlir_egglog import expr_model as ir
from mlir_egglog.term_ir import Term as IRTerm, PowConst as IRPowConst
from mlir_egglog import builtin_functions as builtins  # noqa: F401
from mlir_egglog.mlir_gen import MLIRGen


class ExprFactory:
    """
    A factory class for creating IR expressions from primitive values.
    This helps convert between different representations during compilation.
    """

    @classmethod
    def lit_f32(cls, v: float) -> ir.Expr:
        return ir.FloatLiteral(v)

    @classmethod
    def lit_f64(cls, v: float) -> ir.Expr:
        return ir.FloatLiteral(v)

    @classmethod
    def lit_i64(cls, v: int) -> ir.Expr:
        return ir.IntLiteral(v)

    @classmethod
    def var(cls, k: str) -> ir.Expr:
        return ir.Symbol(k)


function_map = {
    "Term": ExprFactory,
    "Add": ir.Expr.__add__,
    "Sub": ir.Expr.__sub__,
    "Mul": ir.Expr.__mul__,
    "Div": ir.Expr.__truediv__,
    "Pow": ir.Expr.__pow__,
    "PowConst": IRPowConst,
    "Neg": ir.Expr.__neg__,
    "Exp": builtins.exp,
    "Log": builtins.log,
    "Sin": builtins.sin,
    "Cos": builtins.cos,
    "Tan": builtins.tan,
    "Tanh": builtins.tanh,
    "Sinh": builtins.sinh,
    "Cosh": builtins.cosh,
    "Sqrt": builtins.sqrt,
    "CastF32": builtins.float32,
    "CastI64": builtins.int64,
    "Maximum": builtins.maximum,
    "Symbol": ir.Symbol,
    "Const": ir.IntLiteral,
}


def mangle_assignment(tree: ast.AST) -> ast.AST:
    """
    Mangle an AST tree to attach the result variable to the end of the tree assigned to a
    variable named `_out`.

    >>> mangle_assignment(ast.parse("1+2"))
    _out = 1 + 2
    """

    match tree:
        case ast.Expr() as expr:
            return ast.Assign(
                targets=[ast.Name("_out", ast.Store())],
                value=expr.value,
            )
    return tree


def convert_term_to_expr(tree: IRTerm) -> ir.Expr:
    """
    Convert a term to an expression.
    """

    # Parse the term into an AST
    astree = ast.parse(str(tree))

    # Mangle the assignment
    astree.body[-1] = ast.fix_missing_locations(mangle_assignment(astree.body[-1]))  # type: ignore

    # Execute the AST
    globals: dict[str, Any] = {}
    exec(compile(astree, "<string>", "exec"), function_map, globals)

    # Get the result
    result = globals["_out"]
    return result


def convert_term_to_mlir(tree: IRTerm, argspec: str) -> str:
    """
    Convert a term to MLIR.
    """

    expr = convert_term_to_expr(tree)
    argnames = map(lambda x: x.strip(), argspec.split(","))
    argmap = {k: f"%arg_{k}" for k in argnames}
    source = MLIRGen(expr, argmap).generate()
    return source
