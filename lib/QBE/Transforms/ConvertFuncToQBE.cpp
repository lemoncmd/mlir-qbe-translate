//===- ConvertFuncToQBE.cpp - Func to QBE dialect conversion ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/Transforms/QBEPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

namespace mlir::qbe {
#define GEN_PASS_DEF_CONVERTFUNCTOQBE
#include "QBE/Transforms/QBEPasses.h.inc"

namespace {
struct FuncConversionPattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signatureConverter(
        funcType.getNumInputs());
    for (auto [idx, type] : llvm::enumerate(funcType.getInputs())) {
      signatureConverter.addInputs(idx, {converter->convertType(type)});
    }
    // TODO: support multiple result types
    SmallVector<Type, 1> resultType{};
    if (funcType.getNumResults() == 1) {
      resultType.push_back(converter->convertType(funcType.getResults()[0]));
    }
    auto newFuncType = FunctionType::get(
        getContext(), signatureConverter.getConvertedTypes(), resultType);

    auto newFuncOp =
        rewriter.create<qbe::FuncOp>(op.getLoc(), op.getName(), newFuncType);
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter,
                                      converter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CallConversionPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    // TODO: support multiple result types
    SmallVector<Type, 1> resultType{};
    if (op.getNumResults() == 1) {
      resultType.push_back(converter->convertType(op.getResultTypes()[0]));
    }

    rewriter.replaceOpWithNewOp<qbe::CallOp>(op, resultType, op.getCallee(),
                                             adaptor.getOperands());
    return success();
  }
};

using ReturnConversionPattern =
    ToQBEConversionPatternBase<func::ReturnOp, qbe::ReturnOp>;
} // namespace

struct ConvertFuncToQBE : public impl::ConvertFuncToQBEBase<ConvertFuncToQBE> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    QBETypeConverter converter(&getContext());
    populateFuncToQBEConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    target.addIllegalDialect<func::FuncDialect>();
    target.addLegalDialect<qbe::QBEDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateFuncToQBEConversionPatterns(QBETypeConverter &converter,
                                         RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncConversionPattern,
    CallConversionPattern,
    ReturnConversionPattern
  >(converter, patterns.getContext());
  // clang-format on
}

} // namespace mlir::qbe
