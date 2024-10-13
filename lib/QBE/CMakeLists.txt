add_mlir_dialect_library(MLIRQBEDialect
  IR/QBETypes.cpp
  IR/QBEDialect.cpp
  IR/QBEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QBE

  DEPENDS
  MLIRQBEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )

add_mlir_dialect_library(MLIRQBETransforms
  Transforms/ConvertArithToQBE.cpp
  Transforms/ConvertFuncToQBE.cpp

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
