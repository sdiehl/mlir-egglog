<p align="center">
    <img src=".github/logo.png" width="500px" alt="mlir-egglog">
</p>

# MLIR Egglog

A toy specializing compiler for NumPy expressions that uses MLIR as a target and can use equality saturation (e-graphs) to do term rewriting on the intermediate representation, enabling extremely precise and composable optimizations of mathematical expressions before lowering to MLIR.

We use the embedded Datalog DSL [`egglog`](https://github.com/egraphs-good/egglog) to express and compose rewrite rules in pure Python and the [`egg`](https://docs.rs/egg/latest/egg/) library to extract optimized syntax trees from the e-graph.

The whole project is just under 1500 lines of code, and is designed to be a simple and easy to understand example of how to integrate e-graphs into a compiler pipeline.

## What is Egg and Equality Saturation?

Think of an e-graph as this magical data structure that's like a super-powered hash table of program expressions. Instead of just storing one way to write a program, it efficiently stores ALL equivalent ways to write it.

Equality saturation is the process of filling this e-graph with all possible equivalent programs by applying rewrite rules until we can't find any more rewrites (that's the "saturation" part). The cool part? We can explore tons of different optimizations simultaneously, rather than having to pick a specific sequence of transformations. The you can apply a cost function over the entire e-graph to find the best solution. 

Traditionally you'd have to muddle through with a fixed-point iteration system and tons of top-down/bottom-up rewrite rule contingent on application orders, but e-graphs make it much more efficient and declarative.

## Installation

On MacOS, install LLVM 20 which includes MLIR:

```shell
brew install llvm@20
```

On Linux, install the dependencies (setup instructions [here](https://apt.llvm.org/)):

```shell
sudo apt-get install -y llvm-20 llvm-20-dev llvm-20-tools mlir-20-tools
```

Then to use the library:

```shell
git clone https://github.com/sdiehl/mlir-egglog.git
cd mlir-egglog
uv sync
uv run python example.py
```

## Usage

```python
from mlir_egglog import kernel

@kernel("float32(float32)")
def fn(x : float) -> float:
    # sinh(x + y) = sinh(x) * cosh(y) + cosh(x) * sinh(y)
    return np.sinh(x) * np.cosh(x) + np.cosh(x) * np.sinh(x)

out = fn(np.array([1, 2, 3]))
print(out)
```

## Custom Rewrite Rules

You can create your own optimization rules using the `ruleset` decorator. Here's a complete example that optimizes away addition with zero:

```python
from mlir_egglog import kernel
from mlir_egglog.term_ir import Term, Add
from egglog import rewrite, ruleset, RewriteOrRule, i64, f64
from typing import Generator

@ruleset
def float_rules(x: Term, y: Term, z: Term, i: i64, f: f64):
    yield rewrite(Add(x, Term.lit_f32(0.0))).to(x)
    yield rewrite(Add(Term.lit_f32(0.0), x)).to(x)

@kernel("float32(float32)", rewrites=(basic_math, float_rules))
def custom_fn(x):
    return x + 0.0  # This addition will be optimized away!

test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = custom_fn(test_input)
print(result)
```

The rewrite rules are applied during compilation, so there's no runtime overhead. The generated MLIR code will be as if you just wrote `return x`. You can combine multiple rulesets to build up more complex program optimizations.

## Codebase

Here's the recommended order to understand the codebase:

**Foundation Layer** - Expression representation and manipulation

1. [`memory_descriptors.py`](src/mlir_egglog/memory_descriptors.py) - Basic memory management utilities for handling NumPy arrays and MLIR memory references
2. [`expr_model.py`](src/mlir_egglog/expr_model.py) - Core expression model defining the base classes for mathematical expressions
3. [`builtin_functions.py`](src/mlir_egglog/builtin_functions.py) - Implementation of basic mathematical functions and operations
4. [`term_ir.py`](src/mlir_egglog/term_ir.py) - Intermediate representation for the egraph system with cost models for operations

**Transformation Layer** - Code transformation and lowering

5. [`python_to_ir.py`](src/mlir_egglog/python_to_ir.py) - Converts Python functions to the internal IR representation
6. [`ir_to_mlir.py`](src/mlir_egglog/ir_to_mlir.py) - Transforms internal IR to MLIR representation
7. [`basic_simplify.py`](src/mlir_egglog/basic_simplify.py) - Basic mathematical simplification rules
8. [`trig_simplify.py`](src/mlir_egglog/trig_simplify.py) - Trigonometric function simplification rules

**Optimization Layer** - Optimization and compilation

9. [`egglog_optimizer.py`](src/mlir_egglog/egglog_optimizer.py) - Core optimization engine using egg-rewrite rules
10. [`mlir_backend.py`](src/mlir_egglog/mlir_backend.py) - MLIR compilation pipeline and optimization passes
11. [`llvm_runtime.py`](src/mlir_egglog/llvm_runtime.py) - LLVM runtime initialization and management

**Execution Layer** - Runtime execution

12. [`jit_engine.py`](src/mlir_egglog/jit_engine.py) - JIT compilation engine for executing optimized code
13. [`dispatcher.py`](src/mlir_egglog/dispatcher.py) - High-level interface for function compilation and execution

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
