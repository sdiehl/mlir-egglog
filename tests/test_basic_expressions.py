import unittest
import platform
import numpy as np
from mlir_egglog.egglog_optimizer import compile
from mlir_egglog.jit_engine import JITEngine
from egglog import rewrite, ruleset, RewriteOrRule, i64, f64
from mlir_egglog.term_ir import Term, Add
from mlir_egglog.basic_simplify import basic_math
from typing import Generator


class TestBasicExpressions(unittest.TestCase):
    def test_arithmetic_expression(self):
        def arithmetic_fn(x):
            return x * 2.0 + 1.0

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(arithmetic_fn, debug=True)
        self.assertIn("arith.mulf", mlir_code)
        self.assertIn("arith.addf", mlir_code)

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(arithmetic_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_trigonometric_expression(self):
        def trig_fn(x):
            return np.sin(x) * np.cos(x)

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(trig_fn, debug=True)
        self.assertIn("math.sin", mlir_code)
        self.assertIn("math.cos", mlir_code)

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(trig_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_exponential_expression(self):
        def exp_fn(x):
            return np.exp(x) + np.log(x)

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(exp_fn, debug=True)
        self.assertIn("math.exp", mlir_code)
        self.assertIn("math.log", mlir_code)

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(exp_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_type_casting(self):
        def cast_fn(x):
            # Cast to int64 and back to float32
            return np.float32(np.int64(x))

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(cast_fn, debug=True)
        self.assertIn("arith.fptosi", mlir_code)  # float to int
        self.assertIn("arith.sitofp", mlir_code)  # int to float

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(cast_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_constants_and_sqrt(self):
        def const_fn(x):
            return np.sqrt(x) + np.pi

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(const_fn, debug=True)
        print(mlir_code)
        self.assertIn("math.sqrt", mlir_code)
        self.assertIn("arith.constant", mlir_code)

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(const_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_full_compilation_pipeline(self):
        def simple_fn(x):
            return x * 2.0 + 1.0

        # Create JIT engine instance
        jit = JITEngine()

        # Test frontend compilation (MLIR generation)
        mlir_code = jit.run_frontend(simple_fn)
        self.assertIn("arith.mulf", mlir_code)
        self.assertIn("arith.addf", mlir_code)

        # Test backend compilation (MLIR to machine code)
        try:
            func_addr = jit.run_backend(mlir_code)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Backend compilation failed: {str(e)}")

    def test_relu_function(self):
        def relu_fn(x):
            # ReLU(x) = max(0, x)
            return np.maximum(x, 0.0)

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(relu_fn, debug=True)
        print("Generated MLIR:")
        print(mlir_code)

        # Check for the maximum operations used to implement maximum
        # Platform-specific expectations
        if platform.system() == "Darwin":
            self.assertIn("arith.maximumf", mlir_code)
        else:
            # On Linux, expect cmpf + select
            self.assertIn("arith.cmpf", mlir_code)
            self.assertIn("arith.select", mlir_code)

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(relu_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_sigmoid_function(self):
        def sigmoid_fn(x):
            # Sigmoid(x) = 1/(1 + e^(-x))
            return 1.0 / (1.0 + np.exp(-x))

        # Test frontend compilation (MLIR generation)
        mlir_code = compile(sigmoid_fn, debug=True)
        print("Generated MLIR:")
        print(mlir_code)
        self.assertIn("math.exp", mlir_code)  # check for exponential
        self.assertIn("arith.divf", mlir_code)  # check for division
        self.assertIn("arith.negf", mlir_code)  # check for negation

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(sigmoid_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")

    def test_custom_rewrites_in_compile(self):
        @ruleset
        def float_rules(
            x: Term, y: Term, z: Term, i: i64, f: f64
        ) -> Generator[RewriteOrRule, None, None]:
            # x + 0.0 = x (float case)
            yield rewrite(Add(x, Term.lit_f32(0.0))).to(x)
            # 0.0 + x = x (float case)
            yield rewrite(Add(Term.lit_f32(0.0), x)).to(x)

        def custom_fn(x):
            return x + 0.0

        # Test frontend compilation (MLIR generation) with custom rewrites
        mlir_code = compile(custom_fn, rewrites=(basic_math, float_rules), debug=True)
        self.assertNotIn(
            "arith.addf", mlir_code
        )  # The addition should be optimized away

        # Test full pipeline compilation
        jit = JITEngine()
        try:
            func_addr = jit.jit_compile(custom_fn)
            self.assertIsNotNone(func_addr)
            self.assertGreater(func_addr, 0)
        except Exception as e:
            self.fail(f"Full pipeline compilation failed: {str(e)}")
