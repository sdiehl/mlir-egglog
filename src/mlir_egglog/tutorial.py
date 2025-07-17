"""
This module provides tools for visualizing and understanding each stage
of the compilation process.
"""

import ast
import inspect
from typing import Callable, Any

from mlir_egglog.python_to_ir import interpret
from mlir_egglog.egglog_optimizer import extract
from mlir_egglog.ir_to_mlir import convert_term_to_mlir
from mlir_egglog.jit_engine import JITEngine
from mlir_egglog.optimization_rules import basic_math


def show_compilation_pipeline(func: Callable[..., Any], debug: bool = True) -> None:
    """
    Educational function showing each stage of the compilation pipeline.

    This helps students understand how a Python function gets transformed
    through each stage: Python AST → IR → Optimized IR → MLIR → Binary

    Args:
        func: Python function to compile
        debug: Whether to show detailed output
    """
    print("=" * 60)
    print(f"COMPILATION PIPELINE FOR: {func.__name__}")
    print("=" * 60)

    # Stage 1: Python Source Code
    print("\nSTAGE 1: Python Source Code")
    print("-" * 30)
    source = inspect.getsource(func)
    print(source)

    # Stage 2: Python AST
    print("\nSTAGE 2: Python AST")
    print("-" * 30)
    try:
        tree = ast.parse(source)
        print(ast.dump(tree, indent=2))
    except Exception as e:
        print(f"Error parsing AST: {e}")

    # Stage 3: IR Expression
    print("\nSTAGE 3: IR Expression")
    print("-" * 30)
    try:
        ir_expr = interpret(func, {"np": __import__("numpy")})  # type: ignore
        print(f"Type: {type(ir_expr).__name__}")
        print(f"Expression: {ir_expr}")
    except Exception as e:
        print(f"Error creating IR: {e}")
        return

    # Stage 4: E-graph Optimization
    print("\nSTAGE 4: E-graph Optimization")
    print("-" * 30)
    try:
        optimized = extract(ir_expr, (basic_math,))
        print(f"Original:  {ir_expr}")
        print(f"Optimized: {optimized}")

        if str(ir_expr) != str(optimized):
            print("Optimization applied!")
        else:
            print("No optimization found")
    except Exception as e:
        print(f"Error in optimization: {e}")
        optimized = ir_expr

    # Stage 5: MLIR Generation
    print("\nSTAGE 5: MLIR Generation")
    print("-" * 30)
    try:
        mlir_code = convert_term_to_mlir(optimized, "x")
        print(mlir_code)
    except Exception as e:
        print(f"Error generating MLIR: {e}")
        return

    # Stage 6: Binary Compilation
    print("\nSTAGE 6: Binary Compilation")
    print("-" * 30)
    try:
        jit = JITEngine()
        func_addr = jit.jit_compile(func)  # type: ignore
        print(f"Successfully compiled to binary (size: {len(func_addr)} bytes)")
    except Exception as e:
        print(f"Binary compilation failed: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def explain_optimization(original_expr: Any, optimized_expr: Any) -> None:
    """
    Explain why one expression is better than another.

    Args:
        original_expr: The original expression
        optimized_expr: The optimized expression
    """
    print(f"Original:  {original_expr}")
    print(f"Optimized: {optimized_expr}")

    if str(original_expr) == str(optimized_expr):
        print("No optimization was applied")
        return

    print("Analysis:")

    # Simple heuristics for common optimizations
    orig_str = str(original_expr)
    opt_str = str(optimized_expr)

    if "+ 0" in orig_str and "+ 0" not in opt_str:
        print("  • Removed addition of zero (x + 0 = x)")

    if "* 1" in orig_str and "* 1" not in opt_str:
        print("  • Removed multiplication by one (x * 1 = x)")

    if "sin(" in orig_str and "cos(" in orig_str and "2 *" in opt_str:
        print(
            "  • Applied trigonometric identity (sin(x)*cos(x) + cos(x)*sin(x) = 2*sin(x)*cos(x))"
        )

    # Count operations
    orig_ops = (
        orig_str.count("+")
        + orig_str.count("*")
        + orig_str.count("-")
        + orig_str.count("/")
    )
    opt_ops = (
        opt_str.count("+")
        + opt_str.count("*")
        + opt_str.count("-")
        + opt_str.count("/")
    )

    if orig_ops > opt_ops:
        print(f"  • Reduced operation count: {orig_ops} → {opt_ops}")

    print("Optimization reduces computational complexity")


def compare_with_without_optimization(func: Callable[..., Any]) -> None:
    """
    Compare compilation with and without e-graph optimization.

    Args:
        func: Function to analyze
    """
    print(f"OPTIMIZATION COMPARISON FOR: {func.__name__}")
    print("=" * 50)

    # Without optimization
    print("\nWITHOUT OPTIMIZATION:")
    ir_expr = interpret(func, {"np": __import__("numpy")})  # type: ignore
    mlir_without = convert_term_to_mlir(ir_expr, "x")
    print(mlir_without)

    # With optimization
    print("\nWITH OPTIMIZATION:")
    optimized = extract(ir_expr, (basic_math,))
    mlir_with = convert_term_to_mlir(optimized, "x")
    print(mlir_with)

    # Analysis
    print("\nANALYSIS:")
    if mlir_without != mlir_with:
        print("  • Optimization produced different MLIR")

        # Simple metrics
        without_lines = mlir_without.count("\n")
        with_lines = mlir_with.count("\n")

        if with_lines < without_lines:
            print(f"  • Reduced MLIR lines: {without_lines} → {with_lines}")

        without_ops = mlir_without.count("arith.") + mlir_without.count("math.")
        with_ops = mlir_with.count("arith.") + mlir_with.count("math.")

        if with_ops < without_ops:
            print(f"  • Reduced operations: {without_ops} → {with_ops}")
    else:
        print("  • No optimization was applied")
