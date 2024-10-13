//===- TranslateToQBE.cpp - Translate QBE dialect to QBE IR -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace mlir::qbe {
LogicalResult translateToQBE(Operation *op, raw_ostream &os) {
  return success();
}
} // namespace mlir::qbe
