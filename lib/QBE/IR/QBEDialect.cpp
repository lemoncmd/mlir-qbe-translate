//===- QBEDialect.cpp - QBE dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/IR/QBETypes.h"

using namespace mlir;
using namespace mlir::qbe;

#include "QBE/IR/QBEOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// QBE dialect.
//===----------------------------------------------------------------------===//

void QBEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "QBE/IR/QBEOps.cpp.inc"
      >();
  registerTypes();
}
