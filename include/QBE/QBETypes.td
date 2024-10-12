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

class QBE_SimpleType<string name, string mnemonic, string desc>
    : QBE_Type<name, mnemonic>
{
    let summary = "QBE " # mnemonic # " type";
    let description = [{
      A type in QBE dialect which is a }] # desc # [{.
    }];
    let assemblyFormat = "";
}

def QBE_WordType : QBE_SimpleType<"QBEWord", "word", "32-bit integer">;
def QBE_LongType : QBE_SimpleType<"QBELong", "long", "64-bit integer">;
def QBE_SingleType : QBE_SimpleType<"QBESingle", "single", "32-bit floating-point number">;
def QBE_DoubleType : QBE_SimpleType<"QBEDouble", "double", "64-bit floating-point number">;

//===----------------------------------------------------------------------===//
// QBE type constraints
//===----------------------------------------------------------------------===//

def QBE_AnyNumber : AnyTypeOf<[QBE_WordType, QBE_LongType, QBE_SingleType, QBE_DoubleType]>;

def QBE_AnyInteger : AnyTypeOf<[QBE_WordType, QBE_LongType]>;

def QBE_AnyFloat : AnyTypeOf<[QBE_SingleType, QBE_DoubleType]>;

#endif // QBE_TYPES
