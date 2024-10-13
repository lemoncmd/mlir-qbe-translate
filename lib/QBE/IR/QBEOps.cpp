//===- QBEOps.cpp - QBE dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/IR/QBETypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

#define GET_OP_CLASSES
#include "QBE/IR/QBEOps.cpp.inc"

using namespace mlir;
using namespace mlir::qbe;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

LogicalResult FuncOp::verify() {
  auto arg_tys = getArgumentTypes();
  auto res_tys = getResultTypes();

  // checks if the arguments are all QBE types
  for (auto arg_ty : arg_tys) {
    if (!isa<QBEDialect>(arg_ty.getDialect())) {
      return emitOpError() << "expected QBE dialect types for arguments, got "
                           << arg_ty;
    }
  }

  // checks if the results are all QBE types
  for (auto res_ty : res_tys) {
    if (!isa<QBEDialect>(res_ty.getDialect())) {
      return emitOpError() << "expected QBE dialect types for results, got "
                           << res_ty;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = (*this)->getParentOfType<FuncOp>();
  auto funType = function.getFunctionType();

  // checks if the result number is the same.
  if (funType.getNumResults() != getOperands().size())
    return emitOpError()
        .append("expected ", funType.getNumResults(), " result operands")
        .attachNote(function.getLoc())
        .append("return type declared here");

  // checks if the result types are the same.
  for (const auto &pair :
       llvm::enumerate(llvm::zip(function.getResultTypes(), getOperands()))) {
    auto [type, operand] = pair.value();
    if (type != operand.getType())
      return emitOpError() << "unexpected type " << operand.getType()
                           << " for operand #" << pair.index();
  }

  return success();
}
