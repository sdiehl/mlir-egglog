<p align="center">
    <img src=".github/logo.jpeg" width="500px" alt="mlir-egglog">
</p>

# MLIR Egglog

A toy specializing compiler for NumPy expressions that uses MLIR as a target and can use equality saturation (e-graphs) to do term rewriting on the intermediate representation, enabling extremely precise and composable optimizations of mathematical expressions before lowering to MLIR.

We use the embedded Datalog DSL `egglog` to express and compose rewrite rules in pure Python and the `egg` library to extract optimized syntax trees from the e-graph.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.