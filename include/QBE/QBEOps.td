//===- QBEOps.td - QBE dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_OPS
#define QBE_OPS

include "QBE/QBETypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class QBE_ArithOpBase<Type type, string mnemonic, list<Trait> traits = []>
    : QBE_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {
  let arguments = (ins type:$lhs, type:$rhs);
  let results = (outs type:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

class QBE_ArithAnyNumberOp<string mnemonic, list<Trait> traits = []>
    : QBE_ArithOpBase<QBE_AnyNumber, mnemonic, traits>;

class QBE_ArithAnyIntegerOp<string mnemonic, list<Trait> traits = []>
    : QBE_ArithOpBase<QBE_AnyInteger, mnemonic, traits>;

def QBE_AddOp : QBE_ArithAnyNumberOp<"add", [Commutative]>;
def QBE_SubOp : QBE_ArithAnyNumberOp<"sub">;
def QBE_MulOp : QBE_ArithAnyNumberOp<"mul", [Commutative]>;
def QBE_DivOp : QBE_ArithAnyNumberOp<"div">;

def QBE_NegOp : QBE_ArithAnyNumberOp<"neg"> {
  let arguments = (ins QBE_AnyNumber:$value);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

def QBE_UDivOp : QBE_ArithAnyIntegerOp<"udiv">;
def QBE_RemOp : QBE_ArithAnyIntegerOp<"rem">;
def QBE_URemOp : QBE_ArithAnyIntegerOp<"urem">;

def QBE_OrOp : QBE_ArithAnyIntegerOp<"or", [Commutative]>;
def QBE_XorOp : QBE_ArithAnyIntegerOp<"xor", [Commutative]>;
def QBE_AndOp : QBE_ArithAnyIntegerOp<"and", [Commutative]>;

class QBE_ShiftOpBase<string mnemonic, list<Trait> traits = []>
    : QBE_Op<mnemonic, !listconcat([Pure, AllTypesMatch<["lhs", "res"]>], traits)> {
  let arguments = (ins QBE_AnyInteger:$lhs, QBE_WordType:$rhs);
  let results = (outs QBE_AnyInteger:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def QBE_SarOp : QBE_ShiftOpBase<"sar">;
def QBE_ShrOp : QBE_ShiftOpBase<"shr">;
def QBE_ShlOp : QBE_ShiftOpBase<"shl">;

#endif // QBE_OPS
