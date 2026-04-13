import llvmlite.binding as llvm
from functools import cache


@cache
def init_llvm():
    llvm.initialize_all_targets()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    return None
