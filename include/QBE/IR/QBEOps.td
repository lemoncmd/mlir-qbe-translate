//===- QBEOps.td - QBE dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QBE_OPS
#define QBE_OPS

include "QBE/IR/QBETypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// Function and Jump Ops
//===----------------------------------------------------------------------===//

def QBE_FuncOp : QBE_Op<"func", [
  AutomaticAllocationScope, FunctionOpInterface,
  IsolatedFromAbove
]> {
  let summary = "QBE dialect function.";
  let description = [{
    QBE function op provides QBE compatible function.

    Example:
    ```
    qbe.func @foo(%a: !qbe.word) -> !qbe.word {
      %b = qbe.neg %a : !qbe.word
      qbe.return %b : !qbe.word
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    StrAttr:$sym_name
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

def QBE_JmpOp : QBE_Op<"jmp", [
  //DeclareOpInterfaceMethods<BranchOpInterface>,
  Pure, Terminator
]> {
  let summary = "QBE dialect unconditional jump.";
  let description = [{
    A terminator operation which jumps unconditionally to target block.
  }];
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
}

def QBE_JnzOp : QBE_Op<"jnz", [
  /* AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<BranchOpInterface>,*/ Pure, Terminator
]> {
  let summary = "QBE dialect conditional jump.";
  let description = [{
    A terminator operation which conditionally jumps to target blocks.
    When the argument is non-zero, it jumps to the first block.
    Otherwise it jumps to the second block.
  }];
  let arguments = (ins QBE_WordType:$cond);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$cond `,` $trueDest `,` $falseDest attr-dict";
}

def QBE_HaltOp : QBE_Op<"hlt", [Terminator]> {
  let summary = "QBE dialect halt.";
  let description = [{
    A terminator operation with a target-dependent error.
  }];
  let assemblyFormat = "attr-dict";
}

def QBE_ReturnOp : QBE_Op<"return", [
  HasParent<"FuncOp">, MemRefsNormalizable, Pure, Terminator, ReturnLike
]> {
  let summary = "QBE dialect return.";
  let description = [{
    A terminator operation for regions of qbe function body.
    The operands are the result of the function.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic Ops
//===----------------------------------------------------------------------===//

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
