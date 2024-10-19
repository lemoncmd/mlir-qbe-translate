//===- ConvertArithToQBE.cpp - Arith to QBE dialect conversion --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "QBE/IR/QBETypes.h"
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

struct CmpIConversionPattern : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<qbe::CeqOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<qbe::CneOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<qbe::CsltOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::sle:
      rewriter.replaceOpWithNewOp<qbe::CsleOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<qbe::CsgtOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::sge:
      rewriter.replaceOpWithNewOp<qbe::CsgeOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<qbe::CultOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<qbe::CuleOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<qbe::CugtOp>(op, adaptor.getOperands());
      break;
    case arith::CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<qbe::CugeOp>(op, adaptor.getOperands());
      break;
    }
    return success();
  }
};

struct CmpFConversionPattern : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();
    Value comp;
    switch (pred) {
    case arith::CmpFPredicate::AlwaysTrue:
      rewriter.replaceOpWithNewOp<qbe::ConstantOp>(
          op, qbe::QBEWordType::get(rewriter.getContext()),
          rewriter.getI32IntegerAttr(1));
      return success();
    case arith::CmpFPredicate::AlwaysFalse:
      rewriter.replaceOpWithNewOp<qbe::ConstantOp>(
          op, qbe::QBEWordType::get(rewriter.getContext()),
          rewriter.getI32IntegerAttr(0));
      return success();
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      comp = rewriter.create<qbe::CeqOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      comp = rewriter.create<qbe::CgtOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      comp = rewriter.create<qbe::CgeOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      comp = rewriter.create<qbe::CltOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      comp = rewriter.create<qbe::CleOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      comp = rewriter.create<qbe::CneOp>(op.getLoc(), adaptor.getOperands());
      break;
    case arith::CmpFPredicate::ORD:
      rewriter.replaceOpWithNewOp<qbe::CoOp>(op, adaptor.getOperands());
      return success();
    case arith::CmpFPredicate::UNO:
      rewriter.replaceOpWithNewOp<qbe::CuoOp>(op, adaptor.getOperands());
      return success();
    }
    if (pred < arith::CmpFPredicate::ORD) {
      auto ord = rewriter.create<qbe::CoOp>(op.getLoc(), adaptor.getOperands());
      rewriter.replaceOpWithNewOp<qbe::AndOp>(op, comp, ord);
    } else {
      auto uo = rewriter.create<qbe::CuoOp>(op.getLoc(), adaptor.getOperands());
      rewriter.replaceOpWithNewOp<qbe::OrOp>(op, comp, uo);
    }
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
    ConstantConversionPattern,
    CmpIConversionPattern,
    CmpFConversionPattern
  >(converter, patterns.getContext());
  // clang-format on
}

} // namespace mlir::qbe
