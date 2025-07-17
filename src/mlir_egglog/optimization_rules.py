"""
Unified optimization rules for mathematical expressions.

This module combines basic algebraic simplifications and trigonometric identities
into a single, well-organized collection of rewrite rules for the e-graph optimizer.
"""

from mlir_egglog.term_ir import (
    Term,
    Add,
    Mul,
    Div,
    Pow,
    PowConst,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    birewrite_subsume,
)
from egglog import RewriteOrRule, ruleset, rewrite, i64, f64
from typing import Generator


@ruleset
def basic_math(
    x: Term, y: Term, z: Term, i: i64, f: f64
) -> Generator[RewriteOrRule, None, None]:
    """
    Basic algebraic simplification rules.
    """
    # Allow us to translate Term into their specializations
    yield from birewrite_subsume(x + y, Add(x, y))
    yield from birewrite_subsume(x * y, Mul(x, y))
    yield from birewrite_subsume(x / y, Div(x, y))
    yield from birewrite_subsume(x**y, Pow(x, y))

    # Identity elements
    # x + 0 = x (integer case)
    yield rewrite(Add(x, Term.lit_i64(0))).to(x)
    # x + 0.0 = x (float case)
    yield rewrite(Add(x, Term.lit_f32(0.0))).to(x)
    # 0.0 + x = x (float case)
    yield rewrite(Add(Term.lit_f32(0.0), x)).to(x)

    # x * 1 = x
    yield rewrite(Mul(x, Term.lit_i64(1))).to(x)

    # x * 0 = 0
    yield rewrite(Mul(x, Term.lit_i64(0))).to(Term.lit_i64(0))

    # Associativity
    # (x + y) + z = x + (y + z)
    yield rewrite(Add(x, Add(y, z))).to(Add(Add(x, y), z))

    # (x * y) * z = x * (y * z)
    yield rewrite(Mul(x, Mul(y, z))).to(Mul(Mul(x, y), z))

    # Common patterns
    # x + x = 2 * x
    yield rewrite(Add(x, x)).to(Mul(Term.lit_i64(2), x))

    # x * x = x^2
    yield rewrite(Mul(x, x)).to(Pow(x, Term.lit_i64(2)))

    # Power rules
    # (x^y) * (x^z) = x^(y + z)
    yield rewrite(Pow(x, y) * Pow(x, z)).to(Pow(x, Add(y, z)))

    # x^i = x * x^(i - 1)
    yield rewrite(Pow(x, Term.lit_i64(i))).to(PowConst(x, i))

    # x^0 = 1
    yield rewrite(PowConst(x, 0)).to(Term.lit_f32(1.0))

    # x^1 = x
    yield rewrite(PowConst(x, 1)).to(x)

    # x^i = x * x^(i - 1)
    yield rewrite(PowConst(x, i)).to(Mul(x, PowConst(x, i - 1)), i > 1)


@ruleset
def trig_simplify(
    x: Term, y: Term, z: Term, i: i64, fval: f64
) -> Generator[RewriteOrRule, None, None]:
    """
    Trigonometric and hyperbolic function simplification rules.
    """
    # Fundamental trig identities
    # sin²(x) + cos²(x) = 1
    two = Term.lit_i64(2)
    yield rewrite(Add(Pow(Sin(x), two), Pow(Cos(x), two))).to(Term.lit_f32(1.0))

    # Double angle formulas
    yield rewrite(Sin(x + y)).to(Sin(x) * Cos(y) + Cos(x) * Sin(y))
    yield rewrite(Sin(x - y)).to(Sin(x) * Cos(y) - Cos(x) * Sin(y))
    yield rewrite(Cos(x + y)).to(Cos(x) * Cos(y) - Sin(x) * Sin(y))
    yield rewrite(Cos(x - y)).to(Cos(x) * Cos(y) + Sin(x) * Sin(y))

    # Hyperbolic identities
    yield rewrite(Sinh(x) * Cosh(y) + Cosh(y) * Sinh(x)).to(Sinh(x + y))
    yield rewrite(Cosh(x) * Cosh(y) + Sinh(x) * Sinh(y)).to(Cosh(x + y))
    yield rewrite((Tanh(x) + Tanh(y)) / (Term.lit_i64(1) + Tanh(x) * Tanh(y))).to(
        Tanh(x + y)
    )


# For backward compatibility, keep the original exports
__all__ = ["basic_math", "trig_simplify"]
