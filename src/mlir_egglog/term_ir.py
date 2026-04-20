# mypy: disable-error-code=empty-body

from __future__ import annotations

from abc import abstractmethod
from types import ModuleType
import sys

import numpy as np
import egglog

basic_math = egglog.ruleset(name="basic_math")

# Operation costs based on LLVM instruction complexity
# Basic arithmetic (single CPU instruction)
COST_BASIC_ARITH = 1

# Type conversion operations
COST_CAST = 2

# More expensive arithmetic operations
COST_DIV = 5
COST_POW_INTEGER = 10

# Hardware-assisted transcendental operations
COST_SQRT = 20
COST_LOG2 = 25
COST_LOG = 30
COST_LOG10 = 35
COST_EXP = 40

# General power operation (requires exp and log)
COST_POW = 50

# Trigonometric functions (Taylor series implementations)
COST_TRIG_BASIC = 75  # sin, cos
COST_TRIG_TAN = 80
COST_TRIG_ATAN = 85
COST_TRIG_ASINCOS = 90

# Hyperbolic functions (complex Taylor series)
COST_HYPERBOLIC = 180
COST_TANH = 200

_array_ns = sys.modules[__name__]

# Must be globally defined as a method because NumPy checks the type object itself instead of relying on `__getattr__`
egglog.define_expr_method("__array_ufunc__")
egglog.define_expr_method("__array_function__")


class Term(egglog.Expr, ruleset=basic_math):
    """
    Intermediate representation for the egraph.
    """

    @egglog.method(preserve=True)
    def __array_namespace__(self, api_version: object = None) -> ModuleType:
        """
        Returns this module which should be compatible as an array API namespace

        https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.array.__array_namespace__.html
        """
        return _array_ns

    @egglog.method(preserve=True)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Redirect calling ufuncs on terms to top level functions in this module of the same name.

        https://numpy.org/devdocs/user/basics.dispatch.html
        """
        if method == "__call__":
            return getattr(_array_ns, ufunc.__name__)(*inputs, **kwargs)
        raise NotImplementedError(
            f"Unsupported ufunc method: {method}. Only '__call__' is supported."
        )

    @egglog.method(preserve=True)
    def __array_function__(self, func, types, args, kwargs):
        return getattr(_array_ns, func.__name__)(*args, **kwargs)

    @classmethod
    @abstractmethod
    def var(self, k: egglog.StringLike) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_f64(self, v: egglog.f64Like) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_f32(self, v: egglog.f64Like) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_i64(self, v: egglog.i64Like) -> Term: ...

    @egglog.method(cost=COST_BASIC_ARITH)
    @abstractmethod
    def __add__(self, other: Term) -> Term: ...

    @egglog.method(cost=COST_BASIC_ARITH)
    @abstractmethod
    def __mul__(self, other: Term) -> Term: ...

    @egglog.method(cost=COST_BASIC_ARITH)
    @abstractmethod
    def __neg__(self) -> Term: ...

    @egglog.method(cost=COST_BASIC_ARITH)
    @abstractmethod
    def __sub__(self, other: Term) -> Term: ...

    @egglog.method(cost=COST_DIV)
    @abstractmethod
    def __truediv__(self, other: Term) -> Term: ...

    @egglog.method(cost=COST_POW)
    @abstractmethod
    def __pow__(self, other: Term) -> Term: ...

    @egglog.method(preserve=True)
    def exp(self) -> Term:
        return exp(self)

    @egglog.method(preserve=True)
    def sqrt(self) -> Term:
        return sqrt(self)

    @egglog.method(preserve=True)
    def log(self) -> Term:
        return log(self)

    @egglog.method(preserve=True)
    def sin(self) -> Term:
        return sin(self)

    @egglog.method(preserve=True)
    def cos(self) -> Term:
        return cos(self)

    @egglog.method(preserve=True)
    def tan(self) -> Term:
        return tan(self)

    @egglog.method(preserve=True)
    def asin(self) -> Term:
        return asin(self)

    @egglog.method(preserve=True)
    def acos(self) -> Term:
        return acos(self)

    @egglog.method(preserve=True)
    def atan(self) -> Term:
        return atan(self)


class DType(egglog.Expr):
    def __init__(self, name: egglog.StringLike) -> None: ...

    __match_args__ = ("name",)

    @egglog.method(preserve=True)  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        match egglog.get_callable_args(self, DType):
            case (egglog.String(name),):
                return name
        raise egglog.ExprValueError(self, "DType(String(name))")


egglog.converter(type, DType, lambda t: DType(np.dtype(t).name))


@egglog.function(cost=COST_POW_INTEGER)
def PowConst(x: Term, i: egglog.i64Like) -> Term: ...


# Unary Operations
@egglog.function(cost=COST_TRIG_BASIC)
def sin(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_BASIC)
def cos(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_TAN)
def tan(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ASINCOS)
def asin(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ASINCOS)
def acos(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ATAN)
def atan(x: Term) -> Term: ...


@egglog.function(egg_fn="term_sqrt", cost=COST_SQRT)
def sqrt(x: Term) -> Term: ...


@egglog.function(cost=COST_TANH)
def tanh(x: Term) -> Term: ...


@egglog.function(cost=COST_HYPERBOLIC)
def sinh(x: Term) -> Term: ...


@egglog.function(cost=COST_HYPERBOLIC)
def cosh(x: Term) -> Term: ...


@egglog.function(cost=COST_EXP)
def exp(x: Term) -> Term: ...


@egglog.function(egg_fn="term_log", cost=COST_LOG)
def log(x: Term) -> Term: ...


@egglog.function(cost=COST_LOG10)
def log10(x: Term) -> Term: ...


@egglog.function(cost=COST_LOG2)
def log2(x: Term) -> Term: ...


@egglog.function(cost=COST_BASIC_ARITH)
def maximum(x: Term, y: Term) -> Term: ...


@egglog.function(subsume=True, ruleset=basic_math)
def absolute(x: Term) -> Term:
    return maximum(x, -x)


@egglog.function(subsume=True, ruleset=basic_math)
def relu(x):
    return maximum(x, 0.0)


@egglog.function(subsume=True, ruleset=basic_math)
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


@egglog.function(cost=COST_CAST)
def astype(x: Term, dtype: DType) -> Term: ...


egglog.converter(egglog.f64, Term, Term.lit_f32)  # type: ignore[type-abstract]
egglog.converter(egglog.i64, Term, Term.lit_i64)  # type: ignore[type-abstract]
