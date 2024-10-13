add_mlir_translation_library(MLIRTargetQBE
  TranslateRegistration.cpp
  TranslateToQBE.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QBE

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQBEDialect
  MLIRSupport
  MLIRTranslateLib
  )
