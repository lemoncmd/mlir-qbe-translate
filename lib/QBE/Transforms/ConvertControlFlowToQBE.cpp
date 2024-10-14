//===- ConvertControlFlowToQBE.cpp - CF to QBE dialect conv -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/Transforms/QBEPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <utility>

namespace mlir::qbe {
#define GEN_PASS_DEF_CONVERTCONTROLFLOWTOQBE
#include "QBE/Transforms/QBEPasses.h.inc"

namespace {
struct BranchConversionPattern : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<qbe::JmpOp>(op, op.getDest());
    return success();
  }
};

struct CondBranchConversionPattern
    : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<qbe::JnzOp>(
        op, adaptor.getCondition(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};
} // namespace

struct ConvertControlFlowToQBE
    : public impl::ConvertControlFlowToQBEBase<ConvertControlFlowToQBE> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    QBETypeConverter converter(&getContext());
    populateControlFlowToQBEConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    target.addIllegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<qbe::QBEDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateControlFlowToQBEConversionPatterns(QBETypeConverter &converter,
                                                RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    BranchConversionPattern,
    CondBranchConversionPattern
  >(converter, patterns.getContext());
  // clang-format on
}

} // namespace mlir::qbe
