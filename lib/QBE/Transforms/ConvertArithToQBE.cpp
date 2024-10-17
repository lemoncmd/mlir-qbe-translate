//===- ConvertArithToQBE.cpp - Arith to QBE dialect conversion --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/Transforms/QBEPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <utility>

namespace mlir::qbe {
#define GEN_PASS_DEF_CONVERTARITHTOQBE
#include "QBE/Transforms/QBEPasses.h.inc"

namespace {
using AddIConversionPattern =
    ToQBEConversionPatternBase<arith::AddIOp, qbe::AddOp>;
using AddFConversionPattern =
    ToQBEConversionPatternBase<arith::AddFOp, qbe::AddOp>;
using SubIConversionPattern =
    ToQBEConversionPatternBase<arith::SubIOp, qbe::SubOp>;
using SubFConversionPattern =
    ToQBEConversionPatternBase<arith::SubFOp, qbe::SubOp>;
using MulIConversionPattern =
    ToQBEConversionPatternBase<arith::MulIOp, qbe::MulOp>;
using MulFConversionPattern =
    ToQBEConversionPatternBase<arith::MulFOp, qbe::MulOp>;
using DivSIConversionPattern =
    ToQBEConversionPatternBase<arith::DivSIOp, qbe::DivOp>;
using DivFConversionPattern =
    ToQBEConversionPatternBase<arith::DivFOp, qbe::DivOp>;
using NegFConversionPattern =
    ToQBEConversionPatternBase<arith::NegFOp, qbe::NegOp>;
using DivUIConversionPattern =
    ToQBEConversionPatternBase<arith::DivUIOp, qbe::UDivOp>;
using RemSIConversionPattern =
    ToQBEConversionPatternBase<arith::RemSIOp, qbe::RemOp>;
using RemUIConversionPattern =
    ToQBEConversionPatternBase<arith::RemUIOp, qbe::URemOp>;
using OrIConversionPattern =
    ToQBEConversionPatternBase<arith::OrIOp, qbe::OrOp>;
using XOrIConversionPattern =
    ToQBEConversionPatternBase<arith::XOrIOp, qbe::XorOp>;
using AndIConversionPattern =
    ToQBEConversionPatternBase<arith::AndIOp, qbe::AndOp>;
using ShRSIConversionPattern =
    ToQBEConversionPatternBase<arith::ShRSIOp, qbe::SarOp>;
using ShRUIConversionPattern =
    ToQBEConversionPatternBase<arith::ShRUIOp, qbe::ShrOp>;
using ShLIConversionPattern =
    ToQBEConversionPatternBase<arith::ShLIOp, qbe::ShlOp>;

struct ConstantConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<qbe::ConstantOp>(
        op, typeConverter->convertType(adaptor.getValue().getType()),
        op.getValue());
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
  // clang-format off
  patterns.add<
    AddIConversionPattern,
    AddFConversionPattern,
    SubIConversionPattern,
    SubFConversionPattern,
    MulIConversionPattern,
    MulFConversionPattern,
    DivSIConversionPattern,
    DivFConversionPattern,
    NegFConversionPattern,
    DivUIConversionPattern,
    RemSIConversionPattern,
    RemUIConversionPattern,
    OrIConversionPattern,
    XOrIConversionPattern,
    AndIConversionPattern,
    ShRSIConversionPattern,
    ShRUIConversionPattern,
    ShLIConversionPattern,
    ConstantConversionPattern
  >(converter, patterns.getContext());
  // clang-format on
}

} // namespace mlir::qbe
