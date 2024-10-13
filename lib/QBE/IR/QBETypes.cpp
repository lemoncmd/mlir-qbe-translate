//===- QBETypes.cpp - QBE dialect types -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBETypes.h"

#include "QBE/IR/QBEDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::qbe;

#define GET_TYPEDEF_CLASSES
#include "QBE/IR/QBEOpsTypes.cpp.inc"

void QBEDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "QBE/IR/QBEOpsTypes.cpp.inc"
      >();
}
