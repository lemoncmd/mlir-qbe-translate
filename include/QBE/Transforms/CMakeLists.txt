set(LLVM_TARGET_DEFINITIONS QBEPasses.td)
mlir_tablegen(QBEPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRQBEPassesIncGen)
