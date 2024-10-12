//===- QBEDialect.cpp - QBE dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/QBEDialect.h"
#include "QBE/QBEOps.h"
#include "QBE/QBETypes.h"

using namespace mlir;
using namespace mlir::qbe;

#include "QBE/QBEOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// QBE dialect.
//===----------------------------------------------------------------------===//

void QBEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "QBE/QBEOps.cpp.inc"
      >();
  registerTypes();
}