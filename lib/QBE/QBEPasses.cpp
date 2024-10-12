//===- QBEPasses.cpp - QBE passes -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/QBEPasses.h"
#include "QBE/QBEDialect.h"
#include "QBE/QBEOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::qbe {
#define GEN_PASS_DEF_CONVERTARITHTOQBE
#include "QBE/QBEPasses.h.inc"

namespace {
struct AddIConversionPattern : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<qbe::AddOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

struct ConvertArithToQBE
    : public impl::ConvertArithToQBEBase<ConvertArithToQBE> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    QBETypeConverter converter(&getContext());
    populateArithToQBEConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalDialect<qbe::QBEDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateArithToQBEConversionPatterns(QBETypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.add<AddIConversionPattern>(converter, patterns.getContext());
}

} // namespace mlir::qbe
