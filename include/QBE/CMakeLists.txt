add_mlir_dialect(QBEOps qbe)
add_mlir_doc(QBEDialect QBEDialect QBE/ -gen-dialect-doc)
add_mlir_doc(QBEOps QBEOps QBE/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS QBEPasses.td)
mlir_tablegen(QBEPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRQBEPassesIncGen)
