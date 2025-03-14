from __future__ import annotations
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    def __add__(self, other: Expr) -> Expr:
        return Add(self, as_expr(other))

    def __radd__(self, other: Expr) -> Expr:
        return Add(as_expr(other), self)

    def __mul__(self, other: Expr) -> Expr:
        return Mul(self, as_expr(other))

    def __rmul__(self, other: Expr) -> Expr:
        return Mul(as_expr(other), self)

    def __truediv__(self, other: Expr) -> Expr:
        return Div(self, as_expr(other))

    def __rtruediv__(self, other: Expr) -> Expr:
        return Div(as_expr(other), self)

    def __sub__(self, other: Expr) -> Expr:
        return Sub(self, as_expr(other))

    def __rsub__(self, other: Expr) -> Expr:
        return Sub(as_expr(other), self)

    def __pow__(self, other: Expr) -> Expr:
        return Pow(self, as_expr(other))

    def __neg__(self) -> Expr:
        return Neg(self)


@dataclass(frozen=True)
class FloatLiteral(Expr):
    """
    A floating-point constant in the expression tree.
    """

    fval: float


@dataclass(frozen=True)
class IntLiteral(Expr):
    """
    An integer constant in the expression tree.
    """

    ival: float


@dataclass(frozen=True)
class Symbol(Expr):
    """
    A variable or function name in the expression tree.
    """

    name: str


@dataclass(frozen=True)
class UnaryOp(Expr):
    """
    Base class for unary operations in the expression tree.
    """

    operand: Expr


@dataclass(frozen=True)
class BinaryOp(Expr):
    """
    Base class for binary operations in the expression tree.
    """

    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Add(BinaryOp):
    """Addition operation"""

    pass


@dataclass(frozen=True)
class Mul(BinaryOp):
    """Multiplication operation"""

    pass


@dataclass(frozen=True)
class Div(BinaryOp):
    """Division operation"""

    pass


@dataclass(frozen=True)
class Pow(BinaryOp):
    """Power operation"""

    pass


# Transcendental Functions
@dataclass(frozen=True)
class Sin(UnaryOp):
    """Sine operation"""

    pass


@dataclass(frozen=True)
class Cos(UnaryOp):
    """Cosine operation"""

    pass


@dataclass(frozen=True)
class Tan(UnaryOp):
    """Tangent operation"""

    pass


@dataclass(frozen=True)
class ASin(UnaryOp):
    """Arc sine operation"""

    pass


@dataclass(frozen=True)
class ACos(UnaryOp):
    """Arc cosine operation"""

    pass


@dataclass(frozen=True)
class ATan(UnaryOp):
    """Arc tangent operation"""

    pass


@dataclass(frozen=True)
class Sqrt(UnaryOp):
    """Square root operation"""

    pass


@dataclass(frozen=True)
class Tanh(UnaryOp):
    """Hyperbolic tangent operation"""

    pass


@dataclass(frozen=True)
class Sinh(UnaryOp):
    """Hyperbolic sine operation"""

    pass


@dataclass(frozen=True)
class Cosh(UnaryOp):
    """Hyperbolic cosine operation"""

    pass


@dataclass(frozen=True)
class Exp(UnaryOp):
    """Exponential operation"""

    pass


@dataclass(frozen=True)
class Log(UnaryOp):
    """Natural logarithm operation"""

    pass


@dataclass(frozen=True)
class Log10(UnaryOp):
    """Base-10 logarithm operation"""

    pass


@dataclass(frozen=True)
class Log2(UnaryOp):
    """Base-2 logarithm operation"""

    pass


# Type Casting Operations
@dataclass(frozen=True)
class CastF32(UnaryOp):
    """Cast to float32 operation"""

    pass


@dataclass(frozen=True)
class CastI64(UnaryOp):
    """Cast to int64 operation"""

    pass


@dataclass(frozen=True)
class Maximum(BinaryOp):
    """Maximum operation"""

    pass


@dataclass(frozen=True)
class Neg(UnaryOp):
    """Negation operation"""

    pass


@dataclass(frozen=True)
class Sub(BinaryOp):
    """Subtraction operation"""

    pass


# Helper functions for creating operations
def sin(x: Expr) -> Expr:
    return Sin(as_expr(x))


def cos(x: Expr) -> Expr:
    return Cos(as_expr(x))


def tan(x: Expr) -> Expr:
    return Tan(as_expr(x))


def asin(x: Expr) -> Expr:
    return ASin(as_expr(x))


def acos(x: Expr) -> Expr:
    return ACos(as_expr(x))


def atan(x: Expr) -> Expr:
    return ATan(as_expr(x))


def sqrt(x: Expr) -> Expr:
    return Sqrt(as_expr(x))


def tanh(x: Expr) -> Expr:
    return Tanh(as_expr(x))


def sinh(x: Expr) -> Expr:
    return Sinh(as_expr(x))


def cosh(x: Expr) -> Expr:
    return Cosh(as_expr(x))


def exp(x: Expr) -> Expr:
    return Exp(as_expr(x))


def log(x: Expr) -> Expr:
    return Log(as_expr(x))


def log10(x: Expr) -> Expr:
    return Log10(as_expr(x))


def log2(x: Expr) -> Expr:
    return Log2(as_expr(x))


def float32(x: Expr) -> Expr:
    return CastF32(as_expr(x))


def int64(x: Expr) -> Expr:
    return CastI64(as_expr(x))


def maximum(x: Expr, y: Expr) -> Expr:
    return Maximum(as_expr(x), as_expr(y))


def as_expr(val: Any) -> Expr:
    """
    Convert dynamic Python values to Expr nodes.
    """
    match val:
        case float(x):
            return FloatLiteral(x)
        case int(x):
            return IntLiteral(x)
        case Expr() as x:
            return x
        case _:
            raise TypeError(type(val))
