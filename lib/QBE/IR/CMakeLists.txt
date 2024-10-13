add_mlir_dialect_library(MLIRQBEDialect
  QBETypes.cpp
  QBEDialect.cpp
  QBEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QBE

  DEPENDS
  MLIRQBEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  )
