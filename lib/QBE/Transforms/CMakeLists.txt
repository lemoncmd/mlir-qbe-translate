add_mlir_dialect_library(MLIRQBETransforms
  ConvertArithToQBE.cpp
  ConvertControlFlowToQBE.cpp
  ConvertFuncToQBE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QBE

  DEPENDS
  MLIRQBEPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRMathDialect
  MLIRMemRefDialect
  )
