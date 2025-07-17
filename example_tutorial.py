#!/usr/bin/env python3
"""
Educational example demonstrating the mlir-egglog compilation pipeline.

This example shows how to use the tutorial utilities to understand
how the compiler transforms Python code through each stage.
"""

import numpy as np
from mlir_egglog.tutorial import (
    show_compilation_pipeline,
    compare_with_without_optimization,
)


def simple_arithmetic(x):
    """Simple arithmetic expression: x * 2 + 1"""
    return x * 2.0 + 1.0


def redundant_arithmetic(x):
    """Arithmetic with redundant operations: x + 0 * 1"""
    return x + 0.0 * 1.0


def trig_identity(x):
    """Trigonometric identity that can be optimized"""
    return np.sin(x) * np.cos(x) + np.cos(x) * np.sin(x)


def relu_function(x):
    """ReLU activation function: max(0, x)"""
    return np.maximum(x, 0.0)


if __name__ == "__main__":
    print("MLIR-EGGLOG TUTORIAL")
    print("=" * 50)

    print("\n1. SIMPLE ARITHMETIC EXAMPLE")
    print("Understanding the basic compilation pipeline")
    show_compilation_pipeline(simple_arithmetic)

    print("\n\n2. OPTIMIZATION EXAMPLE")
    print("See how e-graph optimization eliminates redundant operations")
    show_compilation_pipeline(redundant_arithmetic)

    print("\n\n3. TRIGONOMETRIC IDENTITY EXAMPLE")
    print("Advanced optimization using mathematical identities")
    show_compilation_pipeline(trig_identity)

    print("\n\n4. COMPARISON WITH/WITHOUT OPTIMIZATION")
    print("Direct comparison of optimized vs unoptimized code")
    compare_with_without_optimization(trig_identity)

    print("\n\n5. PLATFORM-SPECIFIC COMPILATION")
    print("See how Maximum operations are handled differently on different platforms")
    show_compilation_pipeline(relu_function)

    print(
        "\nTutorial complete! You now understand the mlir-egglog compilation pipeline."
    )
    print("Try modifying the functions above to see how different optimizations work.")
