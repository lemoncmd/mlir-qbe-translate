//===- QBEPasses.h - QBE passes  --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef QBE_TRANSFORMS_QBEPASSES_H
#define QBE_TRANSFORMS_QBEPASSES_H

#include "QBE/IR/QBEDialect.h"
#include "QBE/IR/QBEOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
namespace qbe {
#define GEN_PASS_DECL
#include "QBE/Transforms/QBEPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "QBE/Transforms/QBEPasses.h.inc"

class QBETypeConverter : public TypeConverter {
public:
  QBETypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](IntegerType type) -> Type {
      switch (type.getWidth()) {
      case 1: // Bool is Word in QBE
      case 32: {
        return QBEWordType::get(ctx);
      } break;
      case 64: {
        return QBELongType::get(ctx);
      } break;
      }
      return type;
    });
    addConversion([ctx](FloatType type) -> Type {
      switch (type.getWidth()) {
      case 32: {
        return QBESingleType::get(ctx);
      } break;
      case 64: {
        return QBEDoubleType::get(ctx);
      } break;
      }
      return type;
    });
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

template <class From, class To>
struct ToQBEConversionPatternBase : public OpConversionPattern<From> {
  using OpConversionPattern<From>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(From op,
                  typename OpConversionPattern<From>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<To>(op, adaptor.getOperands());
    return success();
  }
};

void populateArithToQBEConversionPatterns(QBETypeConverter &converter,
                                          RewritePatternSet &patterns);

void populateControlFlowToQBEConversionPatterns(QBETypeConverter &converter,
                                                RewritePatternSet &patterns);

void populateFuncToQBEConversionPatterns(QBETypeConverter &converter,
                                         RewritePatternSet &patterns);
} // namespace qbe
} // namespace mlir

#endif // QBE_TRANSFORMS_QBEPASSES_H
