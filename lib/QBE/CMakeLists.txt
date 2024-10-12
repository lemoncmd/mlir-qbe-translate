add_mlir_dialect_library(MLIRQBE
        QBETypes.cpp
        QBEDialect.cpp
        QBEOps.cpp
        QBEPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/QBE

        DEPENDS
        MLIRQBEOpsIncGen
        MLIRQBEPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
