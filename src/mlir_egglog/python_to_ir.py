import types
import inspect

from mlir_egglog import expr_model as ir


def interpret(fn: types.FunctionType, globals: dict[str, object]):
    """
    Symbolically interpret a python function.
    """
    # Get the function's signature
    sig = inspect.signature(fn)

    # Create symbolic parameters for each of the function's arguments
    params = [n for n in sig.parameters]
    symbolic_params = [ir.Symbol(name=n) for n in params]

    # Bind the symbolic parameters to the function's arguments
    ba = sig.bind(*symbolic_params)

    # Inject our globals (i.e. np) into the function's globals
    custom_globals = fn.__globals__.copy()
    custom_globals.update(globals)

    # Create a temporary function with our custom globals
    tfn = types.FunctionType(
        fn.__code__,
        custom_globals,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    return tfn(*ba.args, **ba.kwargs)
