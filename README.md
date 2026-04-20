<p align="center">
    <img src=".github/logo.png" width="500px" alt="mlir-egglog">
</p>

# MLIR Egglog

A toy specializing compiler for NumPy expressions that uses MLIR as a target and can use equality saturation (e-graphs) to do term rewriting on the intermediate representation, enabling extremely precise and composable optimizations of mathematical expressions before lowering to MLIR.

We use the embedded Datalog DSL [`egglog`](https://github.com/egraphs-good/egglog) to express and compose rewrite rules in pure Python and the [`egg`](https://docs.rs/egg/latest/egg/) library to extract optimized syntax trees from the e-graph.

The whole project is just under 1300 lines of code, and is designed to be a simple and easy to understand example of how to integrate e-graphs into a compiler pipeline.

## What is Egg and Equality Saturation?

Think of an e-graph as this magical data structure that's like a super-powered hash table of program expressions. Instead of just storing one way to write a program, it efficiently stores ALL equivalent ways to write it.

Equality saturation is the process of filling this e-graph with all possible equivalent programs by applying rewrite rules until we can't find any more rewrites (that's the "saturation" part). We can explore tons of different optimizations simultaneously, rather than having to pick a specific sequence of transformations. The you can apply a cost function over the entire e-graph to *extract* the best solution that minimizes some user-defined objective function.

Traditionally you'd have to muddle through with a fixed-point iteration system and tons of top-down/bottom-up rewrite rule contingent on application orders, but e-graphs make term rewriting much more efficient, declarative and compositional.

## Installation

On MacOS, install LLVM 20 which includes MLIR:

```shell
brew install llvm@20
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"
```

On Linux, install the dependencies (setup instructions [here](https://apt.llvm.org/)):

```shell
sudo apt-get install -y llvm-20 llvm-20-dev llvm-20-tools mlir-20-tools
```

Then to use the library built it with [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```shell
git clone https://github.com/sdiehl/mlir-egglog.git
cd mlir-egglog
uv sync
uv run python example_basic.py
```

## Usage

```python
import numpy as np
from mlir_egglog import kernel

@kernel("float32(float32)")
def fn(x: float) -> float:
    # sinh(2x) = 2 * sinh(x) * cosh(x)
    return np.sinh(x) * np.cosh(x) + np.cosh(x) * np.sinh(x)

out = fn(np.array([1, 2, 3], dtype=np.float32))
print(out)
```

## Custom Rewrite Rules

You can create your own optimization rules using the `ruleset` decorator. Here's a complete example that optimizes away addition with zero:

```python
import numpy as np
from egglog import rewrite, ruleset
from mlir_egglog import kernel
from mlir_egglog.term_ir import Term
from mlir_egglog.optimization_rules import basic_math

@ruleset
def float_rules(x: Term):
    yield rewrite(x + Term.lit_f32(0.0)).to(x)
    yield rewrite(Term.lit_f32(0.0) + x).to(x)

@kernel("float32(float32)", rewrites=(basic_math, float_rules))
def custom_fn(x):
    return x + 0.0  # This addition will be optimized away!

test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = custom_fn(test_input)
print(result)
```

The rewrite rules are applied during compilation, so there's no runtime overhead. The generated MLIR code will be as if you just wrote `return x`. You can combine multiple rulesets to build up more complex program optimizations.

For a full example see [`example_rewrite.py`](./example_rewrite.py).

## Codebase

Here's the recommended order to understand the codebase:

**Foundation Layer** - Expression representation and manipulation

1. [`memory_descriptors.py`](src/mlir_egglog/memory_descriptors.py) - Memory management utilities for NumPy arrays and MLIR memref descriptors
2. [`term_ir.py`](src/mlir_egglog/term_ir.py) - Intermediate representation for the e-graph system with cost models for each operation

**Transformation Layer** - Code transformation and lowering

3. [`python_to_ir.py`](src/mlir_egglog/python_to_ir.py) - Converts Python functions to the internal IR representation
4. [`optimization_rules.py`](src/mlir_egglog/optimization_rules.py) - Built-in algebraic and trigonometric rewrite rules

**Optimization and Code Generation Layer**

5. [`egglog_optimizer.py`](src/mlir_egglog/egglog_optimizer.py) - Runs the e-graph saturation and extracts the lowest-cost term
6. [`mlir_gen.py`](src/mlir_egglog/mlir_gen.py) - Lowers the extracted term tree to MLIR source text
7. [`mlir_backend.py`](src/mlir_egglog/mlir_backend.py) - Shells out to `mlir-opt` and `mlir-translate` to produce LLVM IR

**Execution Layer** - Runtime execution

8. [`llvm_runtime.py`](src/mlir_egglog/llvm_runtime.py) - Initialises llvmlite to query the host target triple and data layout
9. [`jit_engine.py`](src/mlir_egglog/jit_engine.py) - Compiles LLVM IR to a shared library via `llc` and `cc`, then loads it with `ctypes`
10. [`dispatcher.py`](src/mlir_egglog/dispatcher.py) - `@kernel` decorator: drives compilation and dispatches NumPy array calls

**Educational**

11. [`tutorial.py`](src/mlir_egglog/tutorial.py) - Walks through each stage of the compilation pipeline (AST -> IR -> e-graph -> MLIR -> LLVM IR -> machine code), used by [`example_tutorial.py`](./example_tutorial.py)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
