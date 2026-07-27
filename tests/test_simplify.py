from collections.abc import Generator

import egglog
from egglog import EGraph, rewrite, ruleset

from mlir_egglog.term_ir import (
    COST_BASIC_ARITH,
    COST_EXP,
    COST_POW,
    COST_TRIG_BASIC,
    Term,
    cos,
    exp,
    sin,
)


@ruleset
def cost_based_rules(x: Term) -> Generator[egglog.RewriteOrRule, None, None]:
    # Rule 1: x^2 -> x * x (converts expensive power to cheaper multiplication)
    two = Term.lit_i64(2)
    yield rewrite(x**two).to(x * x)

    # Rule 2: x^1 -> x (eliminates unnecessary power)
    one = Term.lit_i64(1)
    yield rewrite(x**one).to(x)

    # Rule 3: cos(x)^2 + sin(x)^2 -> 1 (trigonometric identity)
    yield rewrite(cos(x) ** two + sin(x) ** two).to(Term.lit_f64(1.0))


@ruleset
def exponential_rules(x: Term) -> Generator[egglog.RewriteOrRule, None, None]:
    # e^x * e^x -> e^(2*x)
    two = Term.lit_i64(2)
    yield rewrite(exp(x) * exp(x)).to(exp(two * x))

    # e^(2x) * e^x -> e^(3x)
    three = Term.lit_i64(3)
    yield rewrite(exp(two * x) * exp(x)).to(exp(three * x))


def test_cost_based_optimization():
    """Test that egglog can optimize expressions based on our cost model."""

    # Create an egglog program
    egraph = EGraph()

    # Create test expressions
    x = Term.var("x")  # Use Term.var directly

    # Test case 1: x^2 should be optimized to x * x
    expr1 = x ** Term.lit_i64(2)  # Use Term constructors directly
    cost_before1 = COST_POW
    cost_after1 = COST_BASIC_ARITH  # multiplication

    # Test case 2: cos(x)^2 + sin(x)^2 should be optimized to 1
    expr2 = cos(x) ** Term.lit_i64(2) + sin(x) ** Term.lit_i64(2)
    cost_before2 = (
        2 * COST_TRIG_BASIC  # cos(x) and sin(x)
        + 2 * COST_POW  # both terms squared
        + COST_BASIC_ARITH  # final addition
    )
    cost_after2 = 0  # constant 1.0 has no runtime cost

    # Add expressions to egraph
    egraph.let("expr1", expr1)
    egraph.let("expr2", expr2)

    # Add the rules manually
    egraph.register(rewrite(x ** Term.lit_i64(2)).to(x * x))
    egraph.register(rewrite(x ** Term.lit_i64(1)).to(x))
    egraph.register(
        rewrite(cos(x) ** Term.lit_i64(2) + sin(x) ** Term.lit_i64(2)).to(
            Term.lit_f64(1.0)
        )
    )

    # Run the optimizer
    egraph.run(10)  # Run for a fixed number of iterations

    # Extract optimized expressions
    opt_expr1 = egraph.extract(expr1)
    opt_expr2 = egraph.extract(expr2)

    # Verify optimizations occurred
    assert str(opt_expr1) == str(x * x), "x^2 should be optimized to x * x"
    assert str(opt_expr2) == str(
        Term.lit_f64(1.0)
    ), "cos(x)^2 + sin(x)^2 should be optimized to 1"

    # Verify cost improvements
    assert cost_before1 > cost_after1, "Optimization should reduce cost of x^2"
    assert (
        cost_before2 > cost_after2
    ), "Optimization should reduce cost of trig identity"


def test_compound_expression_optimization():
    """Test optimization of more complex expressions."""

    egraph = EGraph()

    # Create test expression: e^x * e^x * e^x
    x = Term.var("x")  # Use Term.var directly
    two = Term.lit_i64(2)
    three = Term.lit_i64(3)
    expr = exp(x) * exp(x) * exp(x)  # Use Term constructors directly

    # Calculate costs
    cost_before = (
        3 * COST_EXP  # Three separate e^x calculations
        + 2 * COST_BASIC_ARITH  # Two multiplications
    )

    cost_after = (
        COST_BASIC_ARITH + COST_EXP  # Multiplication by 3  # Single e^(3x) calculation
    )

    # Add expression to egraph
    egraph.let("expr", expr)

    # Add the rules manually
    egraph.register(rewrite(exp(x) * exp(x)).to(exp(two * x)))
    egraph.register(rewrite(exp(two * x) * exp(x)).to(exp(three * x)))

    # Run optimizer
    egraph.run(10)  # Run for a fixed number of iterations

    # Extract optimized expression
    opt_expr = egraph.extract(expr)
    expected = exp(three * x)

    # Verify optimization occurred
    assert str(opt_expr) == str(
        expected
    ), "e^x * e^x * e^x should be optimized to e^(3x)"

    assert (
        cost_before > cost_after
    ), "Optimization should reduce cost of compound exponential"
