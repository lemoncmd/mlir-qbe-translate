//===- QBEDialect.td - QBE dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_DIALECT
#define QBE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// QBE dialect definition.
//===----------------------------------------------------------------------===//

def QBE_Dialect : Dialect {
    let name = "qbe";
    let summary = "A QBE MLIR dialect.";
    let cppNamespace = "::mlir::qbe";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base QBE operation definition.
//===----------------------------------------------------------------------===//

class QBE_Op<string mnemonic, list<Trait> traits = []> :
        Op<QBE_Dialect, mnemonic, traits>;

#endif // QBE_DIALECT
