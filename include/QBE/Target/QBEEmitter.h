//===- QBEEmitter.h - Emits QBE IR from QBE dialect  ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_TARGET_QBEEMITTER_H
#define QBE_TARGET_QBEEMITTER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::qbe {
void registerToQBETranslation();
LogicalResult translateToQBE(Operation *op, raw_ostream &os);
} // namespace mlir::qbe

#endif // QBE_TARGET_QBEEMITTER_H
