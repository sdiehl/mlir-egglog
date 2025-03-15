import numpy as np
from mlir_egglog import kernel
from mlir_egglog.egglog_optimizer import compile
from mlir_egglog.jit_engine import JITEngine
from mlir_egglog.basic_simplify import basic_math
from egglog import rewrite, ruleset, RewriteOrRule, i64, f64
from mlir_egglog.term_ir import Term, Add
from typing import Generator


def test_sin2_plus_cos2():
    """Test that sin²(x) + cos²(x) simplifies to 1 and executes correctly."""

    # First define the raw function for compilation
    def raw_fn(a):
        return np.sin(a) ** 2 + np.cos(a) ** 2

    # Test that it compiles to MLIR and simplifies to 1
    mlir_code = compile(raw_fn, debug=True)
    # The simplified version should just be a constant 1
    assert "arith.constant 1.0" in mlir_code
    assert "math.sin" not in mlir_code  # Should not contain sin after optimization
    assert "math.cos" not in mlir_code  # Should not contain cos after optimization

    # Now create the kernelized version for runtime testing
    @kernel("float32(float32)")
    def fn(a):
        return np.sin(a) ** 2 + np.cos(a) ** 2

    # Test JIT compilation and runtime execution
    jit = JITEngine()

    # Compile the function
    func_addr = jit.jit_compile(fn.py_func)  # Use the underlying Python function
    assert func_addr is not None
    assert func_addr > 0

    # Test with some input values
    test_input = np.array([0.0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float32)
    result = fn(test_input)

    # All results should be very close to 1.0
    np.testing.assert_allclose(result, np.ones_like(test_input), rtol=1e-6)


def test_sigmoid():
    """Test that our kernelized sigmoid matches NumPy's implementation."""

    def numpy_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Define our kernelized sigmoid
    @kernel("float32(float32)")
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Create test input
    x = np.linspace(-5, 5, 100, dtype=np.float32)

    # Get results from both implementations
    expected = numpy_sigmoid(x)
    result = sigmoid(x)

    # Compare results
    np.testing.assert_allclose(result, expected, rtol=1e-6)

    # Test with a 1D array
    x_1d = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    expected_1d = numpy_sigmoid(x_1d)
    result_1d = sigmoid(x_1d)

    # Compare 1D results
    np.testing.assert_allclose(result_1d, expected_1d, rtol=1e-6)

    # Test with a 2D array
    x_2d = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    expected_2d = numpy_sigmoid(x_2d)
    result_2d = sigmoid(x_2d)

    # Compare 2D results
    np.testing.assert_allclose(result_2d, expected_2d, rtol=1e-6)

    # Test some known values
    special_cases = np.array([0.0], dtype=np.float32)  # sigmoid(0) = 0.5
    assert np.abs(sigmoid(special_cases)[0] - 0.5) < 1e-6


def test_multidimensional_arrays():
    """Test that our kernelized functions work with multidimensional arrays."""

    # Define a simple function that adds 1 to each element
    @kernel("float32(float32)")
    def add_one(x):
        return x + 1.0

    # Test with a 2D array
    x_2d = np.ones((3, 4), dtype=np.float32)
    expected_2d = x_2d + 1.0
    result_2d = add_one(x_2d)
    np.testing.assert_allclose(result_2d, expected_2d, rtol=1e-6)

    # Test with a 3D array
    x_3d = np.ones((2, 3, 2), dtype=np.float32)
    expected_3d = x_3d + 1.0
    result_3d = add_one(x_3d)
    np.testing.assert_allclose(result_3d, expected_3d, rtol=1e-6)

    # Test with a more complex function
    @kernel("float32(float32)")
    def complex_fn(x):
        return np.sin(x) * np.cos(x) + np.sqrt(np.abs(x))

    # Test with a 2D array
    test_2d = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    expected_complex = np.sin(test_2d) * np.cos(test_2d) + np.sqrt(np.abs(test_2d))
    result_complex = complex_fn(test_2d)
    np.testing.assert_allclose(result_complex, expected_complex, rtol=1e-5)


def test_custom_rewrites():
    """Test that custom rewrite rules can be passed to the kernel decorator."""

    # Define a custom rewrite rule
    @ruleset
    def float_rules(
        x: Term, y: Term, z: Term, i: i64, f: f64
    ) -> Generator[RewriteOrRule, None, None]:
        # x + 0.0 = x (float case)
        yield rewrite(Add(x, Term.lit_f32(0.0))).to(x)
        # 0.0 + x = x (float case)
        yield rewrite(Add(Term.lit_f32(0.0), x)).to(x)

    # Define a function that uses the custom rewrite rule
    @kernel("float32(float32)", rewrites=(basic_math, float_rules))
    def custom_fn(x):
        return x + 0.0

    # Test that the custom rewrite rule is applied
    mlir_code = compile(
        custom_fn.py_func, rewrites=(basic_math, float_rules), debug=True
    )

    # Test that the custom rewrite rule can be composed with other rules
    mlir_code = compile(
        custom_fn.py_func, rewrites=(basic_math | float_rules,), debug=True
    )
    assert "arith.addf" not in mlir_code  # The addition should be optimized away

    # Test JIT compilation and runtime execution
    jit = JITEngine()
    func_addr = jit.jit_compile(custom_fn.py_func)
    assert func_addr is not None
    assert func_addr > 0

    # Test with some input values
    test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = custom_fn(test_input)
    expected = test_input  # The addition should be optimized away
    np.testing.assert_allclose(result, expected, rtol=1e-6)
