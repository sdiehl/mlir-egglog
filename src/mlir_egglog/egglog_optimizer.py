import inspect
from types import FunctionType

from egglog import EGraph, RewriteOrRule, Ruleset

from mlir_egglog.term_ir import Term, as_egraph
from mlir_egglog.python_to_ir import interpret
from mlir_egglog import builtin_functions as ns
from mlir_egglog.expr_model import Expr
from mlir_egglog.ir_to_mlir import convert_term_to_mlir

# Rewrite rules
from mlir_egglog.basic_simplify import basic_math
from mlir_egglog.trig_simplify import trig_simplify

OPTS: tuple[Ruleset | RewriteOrRule, ...] = (basic_math, trig_simplify)


def extract(ast: Expr, rules: tuple[RewriteOrRule | Ruleset, ...], debug=False) -> Term:
    root = as_egraph(ast)

    egraph = EGraph()
    egraph.let("root", root)

    schedule: Ruleset | None = None
    for opt in rules:
        if schedule is None:
            if isinstance(opt, Ruleset):
                schedule = opt
            else:
                schedule = Ruleset("temp")
                schedule.append(opt)
        else:
            if isinstance(opt, Ruleset):
                schedule = schedule | opt  # type: ignore
            else:
                next_ruleset = Ruleset("temp")
                next_ruleset.append(opt)
                schedule = schedule | next_ruleset  # type: ignore

    if schedule:
        egraph.run(schedule.saturate())
    else:
        raise ValueError("No schedule provided")

    extracted = egraph.extract(root)

    if debug:
        egraph.display()

    return extracted


def compile(
    fn: FunctionType, rewrites: tuple[RewriteOrRule | Ruleset, ...] = OPTS, debug=True
) -> str:
    # Convert np functions accordinging to the namespace map
    exprtree = interpret(fn, {"np": ns})
    extracted = extract(exprtree, rewrites, debug)

    # Get the argument spec
    argspec = inspect.signature(fn)
    params = ",".join(map(str, argspec.parameters))

    return convert_term_to_mlir(extracted, params)
