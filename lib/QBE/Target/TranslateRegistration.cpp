//===- TranslateRegistration.cpp - Register translation ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/Target/QBEEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir::qbe {
void registerToQBETranslation() {
  mlir::TranslateFromMLIRRegistration withdescription(
      "mlir-to-qbeir", "Translate MLIR to QBE IR",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        return mlir::qbe::translateToQBE(op, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::qbe::QBEDialect>();
      });
}
} // namespace mlir::qbe
