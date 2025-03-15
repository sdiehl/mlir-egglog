<p align="center">
    <img src=".github/logo.jpeg" width="500px" alt="mlir-egglog">
</p>

# MLIR Egglog

A toy specializing compiler for NumPy expressions that uses MLIR as a target and can use equality saturation (e-graphs) to do term rewriting on the intermediate representation, enabling extremely precise and composable optimizations of mathematical expressions before lowering to MLIR.

We use the embedded Datalog DSL `egglog` to express and compose rewrite rules in pure Python and the `egg` library to extract optimized syntax trees from the e-graph.

## What is Egg and Equality Saturation?

Think of an e-graph as this magical data structure that's like a super-powered hash table of program expressions. Instead of just storing one way to write a program, it efficiently stores ALL equivalent ways to write it.

Equality saturation is the process of filling this e-graph with all possible equivalent programs by applying rewrite rules until we can't find any more rewrites (that's the "saturation" part). The cool part? We can explore tons of different optimizations simultaneously, rather than having to pick a specific sequence of transformations. The you can apply a cost function over the entire e-graph to find the best solution. 

Traditionally you'd have to muddle through with tons of top-down/bottom-up and rewrite rule application orders, but e-graphs make it much more efficient and declarative.

## Installation

On MacOS, build LLVM and MLIR from source:

```shell
brew install cmake ccache ninja
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;ARM;X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_CCACHE_BUILD=ON
cmake --build . --target check-mlir
cmake --build . --target install
```

Or if you're using the Anaconda distribution, install the dependencies:

```shell
conda install conda-forge::mlir
```

Then to use the library:

```shell
git clone https://github.com/sdiehl/mlir-egglog.git
cd mlir-egglog
poetry install
poetry run python example.py
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

# Define custom rewrite rules
@ruleset
def float_rules(x: Term, y: Term, z: Term, i: i64, f: f64):
    # x + 0.0 = x (float case)
    yield rewrite(Add(x, Term.lit_f32(0.0))).to(x)
    # 0.0 + x = x (float case)
    yield rewrite(Add(Term.lit_f32(0.0), x)).to(x)

# Use the custom rules in a kernel
@kernel("float32(float32)", rewrites=(basic_math, float_rules))
def custom_fn(x):
    return x + 0.0  # This addition will be optimized away!

# Test with some input values
test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = custom_fn(test_input)
print(result)  # Will print [1.0, 2.0, 3.0]
```

The rewrite rules are applied during compilation, so there's no runtime overhead. The generated MLIR code will be as if you just wrote `return x`!

You can combine multiple rulesets to build up more complex program optimizations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
