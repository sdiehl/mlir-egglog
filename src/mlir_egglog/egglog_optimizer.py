import inspect
from types import FunctionType

from egglog import EGraph, RewriteOrRule, Ruleset
from egglog.egraph import UnstableCombinedRuleset

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

    # The user can compose rules as (rule1 | rule2) to apply them in parallel
    # or (rule1, rule2) to apply them sequentially
    for opt in rules:
        if isinstance(opt, Ruleset):
            egraph.run(opt.saturate())
        elif isinstance(opt, UnstableCombinedRuleset):
            egraph.run(opt.saturate())
        else:
            # For individual rules, create a temporary ruleset
            temp_ruleset = Ruleset("temp")
            temp_ruleset.append(opt)
            egraph.run(temp_ruleset.saturate())

    extracted = egraph.extract(root)

    # if debug:
    #     egraph.display()

    return extracted


def compile(
    fn: FunctionType, rewrites: tuple[RewriteOrRule | Ruleset, ...] = OPTS, debug=True
) -> str:
    # Convert np functions according to the namespace map
    exprtree = interpret(fn, {"np": ns})
    extracted = extract(exprtree, rewrites, debug)

    # Get the argument spec
    argspec = inspect.signature(fn)
    params = ",".join(map(str, argspec.parameters))
    return convert_term_to_mlir(extracted, params)
