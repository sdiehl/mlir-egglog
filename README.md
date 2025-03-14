<p align="center">
    <img src=".github/logo.jpeg" width="500px" alt="mlir-egglog">
</p>

# MLIR Egglog

A toy specializing compiler for NumPy expressions that uses MLIR as a target and can use equality saturation (e-graphs) to do term rewriting on the intermediate representation, enabling extremely precise and composable optimizations of mathematical expressions before lowering to MLIR.

We use the embedded Datalog DSL `egglog` to express and compose rewrite rules in pure Python and the `egg` library to extract optimized syntax trees from the e-graph.

## What is Egg and Equality Saturation?

Egg is a library for equality saturation, a technique used in program optimization. Equality saturation is a method where a program is represented as an e-graph (a data structure that compactly represents many equivalent programs). The optimizer then applies rewrite rules to the e-graph until no more rules can be applied (saturation). This allows the optimizer to explore many possible optimizations simultaneously and choose the best one. This is very interesting research because it enables more powerful and flexible optimizations compared to traditional methods, which often rely on a fixed sequence of optimization passes.

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

You can pass custom rewrite rules to the `kernel` decorator using the `rewrites` parameter. This allows you to specify your own optimizations.

```python
from mlir_egglog import kernel
from mlir_egglog.basic_simplify import basic_math

@kernel("float32(float32)", rewrites=(basic_math,))
def fn(x : float) -> float:
    return np.sin(x) * np.cos(x) + np.cos(x) * np.sin(x)

out = fn(np.array([1, 2, 3]))
print(out)
```

## Basic Rewrite Rule Stack

Here is an example of a very basic rewrite rule stack other than the basic rewrite:

```python
from egglog import rewrite
from mlir_egglog.term_ir import Term, Add

@rewrite
def custom_rewrite(x: Term):
    return Add(x, Term.lit_f32(0.0)).to(x)

@kernel("float32(float32)", rewrites=(custom_rewrite,))
def custom_fn(x):
    return x + 0.0

out = custom_fn(np.array([1, 2, 3]))
print(out)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
