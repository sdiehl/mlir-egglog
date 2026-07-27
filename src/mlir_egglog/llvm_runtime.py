from functools import cache

import llvmlite.binding as llvm


@cache
def init_llvm():
    llvm.initialize_all_targets()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
