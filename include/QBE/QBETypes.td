//===- QBETypes.td - QBE dialect types ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_TYPES
#define QBE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "QBE/QBEDialect.td"

//===----------------------------------------------------------------------===//
// QBE type definitions
//===----------------------------------------------------------------------===//

class QBE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QBE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def QBE_CustomType : QBE_Type<"Custom", "custom"> {
    let summary = "QBE custom type";
    let description = "Custom type in QBE dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // QBE_TYPES
