//===- QBEPasses.td - QBE dialect passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_PASS
#define QBE_PASS

include "mlir/Pass/PassBase.td"

def ConvertArithToQBE: Pass<"convert-arith-to-qbe"> {
  let summary = "Convert Arith dialect to QBE dialect";
  let description = [{
    This pass converts supported Arith ops to QBE ops.
  }];
  let dependentDialects = ["arith::ArithDialect", "qbe::QBEDialect"];
}

#endif // QBE_PASS
