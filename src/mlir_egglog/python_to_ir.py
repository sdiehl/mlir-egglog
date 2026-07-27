import inspect
import types

from mlir_egglog import term_ir as ir


def interpret(fn: types.FunctionType):
    """
    Symbolically interpret a python function.
    """
    # Get the function's signature
    sig = inspect.signature(fn)

    # Create symbolic parameters for each of the function's arguments
    params = [n for n in sig.parameters]
    symbolic_params = [ir.Term.var(n) for n in params]

    # Bind the symbolic parameters to the function's arguments
    ba = sig.bind(*symbolic_params)
    return fn(*ba.args, **ba.kwargs)
