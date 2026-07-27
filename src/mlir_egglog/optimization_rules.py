"""
Unified optimization rules for mathematical expressions.

This module combines basic algebraic simplifications and trigonometric identities
into a single, well-organized collection of rewrite rules for the e-graph optimizer.
"""

from collections.abc import Generator

from egglog import RewriteOrRule, f64, i64, rewrite, ruleset

from mlir_egglog.term_ir import PowConst, Term, basic_math, cos, cosh, sin, sinh, tanh


@basic_math.register
def _basic_math(
    x: Term, y: Term, z: Term, i: i64, f: f64
) -> Generator[RewriteOrRule, None, None]:
    """
    Basic algebraic simplification rules.
    """

    # Identity elements
    # x + 0 = x (integer case)
    yield rewrite(x + Term.lit_i64(0)).to(x)
    # x + 0.0 = x (float case)
    yield rewrite(x + Term.lit_f32(0.0)).to(x)
    # 0.0 + x = x (float case)
    yield rewrite(Term.lit_f32(0.0) + x).to(x)

    # x * 1 = x
    yield rewrite(x * Term.lit_i64(1)).to(x)

    # x * 0 = 0
    yield rewrite(x * Term.lit_i64(0)).to(Term.lit_i64(0))

    # Associativity
    yield rewrite((x + y) + z).to(x + (y + z))

    yield rewrite((x * y) * z).to(x * (y * z))

    # Common patterns
    yield rewrite(x + x).to(Term.lit_i64(2) * x)

    # x * x = x^2
    yield rewrite(x * x).to(x ** Term.lit_i64(2))

    # Power rules
    # (x^y) * (x^z) = x^(y + z)
    yield rewrite((x**y) * (x**z)).to(x ** (y + z))

    # x^i = x * x^(i - 1)
    yield rewrite(x ** Term.lit_i64(i)).to(PowConst(x, i))

    # x^0 = 1
    yield rewrite(PowConst(x, 0)).to(Term.lit_f32(1.0))

    # x^1 = x
    yield rewrite(PowConst(x, 1)).to(x)

    # x^i = x * x^(i - 1)
    yield rewrite(PowConst(x, i)).to(x * PowConst(x, i - 1), i > 1)


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
    yield rewrite(sin(x) ** two + cos(x) ** two).to(Term.lit_f32(1.0))

    # Double angle formulas
    yield rewrite(sin(x + y)).to(sin(x) * cos(y) + cos(x) * sin(y))
    yield rewrite(sin(x - y)).to(sin(x) * cos(y) - cos(x) * sin(y))
    yield rewrite(cos(x + y)).to(cos(x) * cos(y) - sin(x) * sin(y))
    yield rewrite(cos(x - y)).to(cos(x) * cos(y) + sin(x) * sin(y))

    # Hyperbolic identities
    yield rewrite(sinh(x) * cosh(y) + cosh(y) * sinh(x)).to(sinh(x + y))
    yield rewrite(cosh(x) * cosh(y) + sinh(x) * sinh(y)).to(cosh(x + y))
    yield rewrite((tanh(x) + tanh(y)) / (Term.lit_i64(1) + tanh(x) * tanh(y))).to(
        tanh(x + y)
    )


# For backward compatibility, keep the original exports
__all__ = ["basic_math", "trig_simplify"]
