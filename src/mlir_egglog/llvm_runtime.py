import llvmlite.binding as llvm
import llvmlite
from functools import cache


@cache
def init_llvm():
    print(llvmlite.__version__)
    llvm.initialize()
    llvm.initialize_all_targets()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    return None


def compile_mod(engine, mod):
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    return compile_mod(engine, mod)
