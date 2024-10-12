get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        MLIRIndexDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIROptLib
        MLIRQBEDialect
        )
add_llvm_executable(qbe-opt qbe-opt.cpp)

llvm_update_compile_flags(qbe-opt)
target_link_libraries(qbe-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(qbe-opt)
