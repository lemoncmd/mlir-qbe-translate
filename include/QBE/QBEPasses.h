//===- QBEPasses.h - QBE passes  --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef QBE_QBEPASSES_H
#define QBE_QBEPASSES_H

#include "QBE/QBEDialect.h"
#include "QBE/QBEOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace qbe {
#define GEN_PASS_DECL
#include "QBE/QBEPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "QBE/QBEPasses.h.inc"
} // namespace qbe
} // namespace mlir

#endif // QBE_QBEPASSES_H
