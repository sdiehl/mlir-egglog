# mypy: disable-error-code=empty-body

from __future__ import annotations

import egglog
from egglog import StringLike, i64, f64, i64Like, f64Like  # noqa: F401
from egglog import RewriteOrRule, rewrite
from typing import Generator
from mlir_egglog.expr_model import Expr, FloatLiteral, Symbol, IntLiteral
from abc import abstractmethod

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


class Term(egglog.Expr):
    """
    Intermediate representation for the egraph.
    """

    @classmethod
    @abstractmethod
    def var(self, k: StringLike) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_f64(self, v: f64Like) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_f32(self, v: f64Like | i64) -> Term: ...

    @classmethod
    @abstractmethod
    def lit_i64(self, v: i64Like) -> Term: ...

    @abstractmethod
    def __add__(self, other: Term) -> Term: ...

    @abstractmethod
    def __mul__(self, other: Term) -> Term: ...

    @abstractmethod
    def __neg__(self) -> Term: ...

    @abstractmethod
    def __sub__(self, other: Term) -> Term: ...

    @abstractmethod
    def __truediv__(self, other: Term) -> Term: ...

    @abstractmethod
    def __pow__(self, other: Term) -> Term: ...


# Binary Operations
@egglog.function(cost=COST_BASIC_ARITH)
def Add(x: Term, y: Term) -> Term: ...


@egglog.function(cost=COST_BASIC_ARITH)
def Mul(x: Term, y: Term) -> Term: ...


@egglog.function(cost=COST_DIV)
def Div(x: Term, y: Term) -> Term: ...


@egglog.function(cost=COST_POW)
def Pow(x: Term, y: Term) -> Term: ...


@egglog.function(cost=COST_POW_INTEGER)
def PowConst(x: Term, i: i64Like) -> Term: ...


# Unary Operations
@egglog.function(cost=COST_TRIG_BASIC)
def Sin(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_BASIC)
def Cos(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_TAN)
def Tan(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ASINCOS)
def ASin(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ASINCOS)
def ACos(x: Term) -> Term: ...


@egglog.function(cost=COST_TRIG_ATAN)
def ATan(x: Term) -> Term: ...


@egglog.function(cost=COST_SQRT)
def Sqrt(x: Term) -> Term: ...


@egglog.function(cost=COST_TANH)
def Tanh(x: Term) -> Term: ...


@egglog.function(cost=COST_HYPERBOLIC)
def Sinh(x: Term) -> Term: ...


@egglog.function(cost=COST_HYPERBOLIC)
def Cosh(x: Term) -> Term: ...


@egglog.function(cost=COST_EXP)
def Exp(x: Term) -> Term: ...


@egglog.function(cost=COST_LOG)
def Log(x: Term) -> Term: ...


@egglog.function(cost=COST_LOG10)
def Log10(x: Term) -> Term: ...


@egglog.function(cost=COST_LOG2)
def Log2(x: Term) -> Term: ...


@egglog.function(cost=COST_CAST)
def CastF32(x: Term) -> Term: ...


@egglog.function(cost=COST_CAST)
def CastI64(x: Term) -> Term: ...


@egglog.function(cost=COST_BASIC_ARITH)
def Maximum(x: Term, y: Term) -> Term: ...


@egglog.function(cost=COST_BASIC_ARITH)
def Neg(x: Term) -> Term: ...


def as_egraph(expr: Expr) -> Term:
    """
    Convert a syntax tree expression to an egraph term.
    """
    from mlir_egglog import expr_model

    match expr:
        # Literals and Symbols
        case FloatLiteral(fval=val):
            return Term.lit_f64(val)
        case IntLiteral(ival=val):
            return Term.lit_i64(int(val))
        case Symbol(name=name):
            return Term.var(name)

        # Binary Operations
        case expr_model.Add(lhs=lhs, rhs=rhs):
            return Add(as_egraph(lhs), as_egraph(rhs))
        case expr_model.Mul(lhs=lhs, rhs=rhs):
            return Mul(as_egraph(lhs), as_egraph(rhs))
        case expr_model.Div(lhs=lhs, rhs=rhs):
            return Div(as_egraph(lhs), as_egraph(rhs))
        case expr_model.Pow(lhs=lhs, rhs=rhs):
            return Pow(as_egraph(lhs), as_egraph(rhs))
        case expr_model.Maximum(lhs=lhs, rhs=rhs):
            return Maximum(as_egraph(lhs), as_egraph(rhs))

        # Trigonometric Functions
        case expr_model.Sin(operand=op):
            return Sin(as_egraph(op))
        case expr_model.Cos(operand=op):
            return Cos(as_egraph(op))
        case expr_model.Tan(operand=op):
            return Tan(as_egraph(op))
        case expr_model.ASin(operand=op):
            return ASin(as_egraph(op))
        case expr_model.ACos(operand=op):
            return ACos(as_egraph(op))
        case expr_model.ATan(operand=op):
            return ATan(as_egraph(op))

        # Hyperbolic Functions
        case expr_model.Tanh(operand=op):
            return Tanh(as_egraph(op))
        case expr_model.Sinh(operand=op):
            return Sinh(as_egraph(op))
        case expr_model.Cosh(operand=op):
            return Cosh(as_egraph(op))

        # Exponential and Logarithmic Functions
        case expr_model.Exp(operand=op):
            return Exp(as_egraph(op))
        case expr_model.Log(operand=op):
            return Log(as_egraph(op))
        case expr_model.Log10(operand=op):
            return Log10(as_egraph(op))
        case expr_model.Log2(operand=op):
            return Log2(as_egraph(op))

        # Type Casting and Other Operations
        case expr_model.CastF32(operand=op):
            return CastF32(as_egraph(op))
        case expr_model.CastI64(operand=op):
            return CastI64(as_egraph(op))
        case expr_model.Neg(operand=op):
            return Neg(as_egraph(op))
        case expr_model.Sqrt(operand=op):
            return Sqrt(as_egraph(op))

        case _:
            raise NotImplementedError(f"Unsupported expression type: {type(expr)}")


def birewrite_subsume(a: Term, b: Term) -> Generator[RewriteOrRule, None, None]:
    yield rewrite(a, subsume=True).to(b)
    yield rewrite(b).to(a)
