//===- TranslateToQBE.cpp - Translate QBE dialect to QBE IR -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QBE/IR/QBEOps.h"
#include "QBE/IR/QBETypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <string>

using namespace mlir;

template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace mlir::qbe {
namespace {
struct Emitter {
  explicit Emitter(raw_ostream &os) : os(os) {}

  LogicalResult emitOperation(Operation &op);

  LogicalResult emitType(Location loc, Type type);

  StringRef getOrCreateName(Value val, StringRef prefix = "val");

  StringRef getOrCreateName(Block &block, StringRef prefix = "block");

  void emitSSEOrConstant(Value val, StringRef prefix = "val");

  raw_indented_ostream &ostream() { return os; }

  void resetValueCount() { valueCount = 0; }

  void resetBlockCount() { blockCount = 0; }

  struct Scope {
    Scope(Emitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper) {}

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
  };

private:
  raw_indented_ostream os;

  llvm::ScopedHashTable<Value, std::string> valueMapper;

  llvm::ScopedHashTable<Block *, std::string> blockMapper;

  int64_t valueCount, blockCount;
};

static LogicalResult printOperation(Emitter &emitter, ModuleOp moduleOp) {
  const Emitter::Scope scope(emitter);
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(Emitter &emitter, FuncOp funcOp) {
  const Emitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  os << "function ";
  if (funcOp.getNumResults() == 1) {
    if (failed(emitter.emitType(funcOp.getLoc(), funcOp.getResultTypes()[0]))) {
      return failure();
    }
    os << " ";
  }

  emitter.resetValueCount();
  os << "$" << funcOp.getSymName() << "(";
  if (failed(interleaveCommaWithError(
          funcOp.getArguments(), os, [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(funcOp.getLoc(), arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg, "arg");
            return success();
          })))
    return failure();
  os << ") {\n";

  Region::BlockListType &blocks = funcOp.getBlocks();

  emitter.resetValueCount();
  emitter.resetBlockCount();
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  for (Block &block : blocks) {
    os << emitter.getOrCreateName(block) << "\n";
    os.indent();

    // Emit phis
    if (!block.hasNoPredecessors()) {
      for (auto [i, arg] : llvm::enumerate(block.getArguments())) {
        // Since structured binded variables can't be captured.
        auto index = i;
        os << emitter.getOrCreateName(arg) << " =";
        if (failed(emitter.emitType(funcOp.getLoc(), arg.getType()))) {
          return failure();
        }
        os << " phi ";
        if (failed(interleaveCommaWithError(
                block.getPredecessors(), os, [&](Block *pred) {
                  os << emitter.getOrCreateName(*pred) << " ";
                  auto *terminator = pred->getTerminator();
                  if (isa<JmpOp>(terminator)) {
                    auto op = cast<JmpOp>(terminator);
                    emitter.emitSSEOrConstant(op.getDestOperands()[index]);
                  } else if (isa<JnzOp>(terminator)) {
                    auto op = cast<JnzOp>(terminator);
                    emitter.emitSSEOrConstant(
                        (op.getTrueDest() == &block
                             ? op.getTrueOperands()
                             : op.getFalseOperands())[index]);
                  } else {
                    return failure();
                  }
                  return success();
                }))) {
          return failure();
        }
        os << "\n";
      }
    }

    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op)))
        return failure();
    }
    os.unindent();
  }

  os << "}\n";
  return success();
}

static LogicalResult printOperation(Emitter &emitter, JmpOp jmpOp) {
  auto &os = emitter.ostream();
  os << "jmp " << emitter.getOrCreateName(*jmpOp.getDest());
  return success();
}

static LogicalResult printOperation(Emitter &emitter, JnzOp jnzOp) {
  auto &os = emitter.ostream();
  os << "jnz ";
  emitter.emitSSEOrConstant(jnzOp.getCond());
  os << ", " << emitter.getOrCreateName(*jnzOp.getTrueDest()) << ", "
     << emitter.getOrCreateName(*jnzOp.getFalseDest());
  return success();
}

static LogicalResult printOperation(Emitter &emitter, HaltOp) {
  auto &os = emitter.ostream();
  os << "hlt";
  return success();
}

static LogicalResult printOperation(Emitter &emitter, ReturnOp returnOp) {
  auto &os = emitter.ostream();
  os << "ret";
  if (returnOp.getNumOperands() == 1) {
    os << " ";
    emitter.emitSSEOrConstant(returnOp.getOperands()[0]);
  }
  return success();
}

template <class T>
static LogicalResult printBinaryOperation(Emitter &emitter, T op) {
  auto &os = emitter.ostream();
  os << emitter.getOrCreateName(op.getRes()) << " =";
  if (failed(emitter.emitType(op.getLoc(), op.getType())))
    return failure();
  os << " " << op->getName().stripDialect() << " ";
  emitter.emitSSEOrConstant(op.getLhs());
  os << ", ";
  emitter.emitSSEOrConstant(op.getRhs());
  return success();
}

static LogicalResult printOperation(Emitter &emitter, AddOp addOp) {
  return printBinaryOperation(emitter, addOp);
}

static LogicalResult printOperation(Emitter &emitter, SubOp subOp) {
  return printBinaryOperation(emitter, subOp);
}

static LogicalResult printOperation(Emitter &emitter, MulOp mulOp) {
  return printBinaryOperation(emitter, mulOp);
}

static LogicalResult printOperation(Emitter &emitter, DivOp divOp) {
  return printBinaryOperation(emitter, divOp);
}

static LogicalResult printOperation(Emitter &emitter, UDivOp udivOp) {
  return printBinaryOperation(emitter, udivOp);
}

static LogicalResult printOperation(Emitter &emitter, RemOp remOp) {
  return printBinaryOperation(emitter, remOp);
}

static LogicalResult printOperation(Emitter &emitter, URemOp uremOp) {
  return printBinaryOperation(emitter, uremOp);
}

static LogicalResult printOperation(Emitter &emitter, OrOp orOp) {
  return printBinaryOperation(emitter, orOp);
}

static LogicalResult printOperation(Emitter &emitter, XorOp xorOp) {
  return printBinaryOperation(emitter, xorOp);
}

static LogicalResult printOperation(Emitter &emitter, AndOp andOp) {
  return printBinaryOperation(emitter, andOp);
}

static LogicalResult printOperation(Emitter &emitter, SarOp sarOp) {
  return printBinaryOperation(emitter, sarOp);
}

static LogicalResult printOperation(Emitter &emitter, ShrOp shrOp) {
  return printBinaryOperation(emitter, shrOp);
}

static LogicalResult printOperation(Emitter &emitter, ShlOp shlOp) {
  return printBinaryOperation(emitter, shlOp);
}

static LogicalResult printOperation(Emitter &emitter, NegOp negOp) {
  auto &os = emitter.ostream();
  os << emitter.getOrCreateName(negOp.getRes()) << " =";
  if (failed(emitter.emitType(negOp.getLoc(), negOp.getType())))
    return failure();
  os << " " << negOp->getName().stripDialect() << " ";
  emitter.emitSSEOrConstant(negOp.getValue());
  return success();
}

StringRef Emitter::getOrCreateName(Value value, StringRef prefix) {
  if (!valueMapper.count(value))
    valueMapper.insert(value, formatv("%{0}{1}", prefix, valueCount++));
  return *valueMapper.begin(value);
}

StringRef Emitter::getOrCreateName(Block &block, StringRef prefix) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("@{0}{1}", prefix, blockCount++));
  return *blockMapper.begin(&block);
}

void Emitter::emitSSEOrConstant(Value value, StringRef prefix) {
  if (auto constOp = dyn_cast_or_null<ConstantOp>(value.getDefiningOp())) {
    auto number = constOp.getValue();
    if (number.getType().isIntOrIndex()) {
      cast<IntegerAttr>(number).print(os, true);
    } else {
      if (number.getType().isF32()) {
        os << "s_";
      } else {
        os << "d_";
      }
      cast<FloatAttr>(number).print(os, true);
    }
  } else {
    os << getOrCreateName(value, prefix);
  }
}

LogicalResult Emitter::emitType(Location loc, Type type) {
  const StringRef name = llvm::TypeSwitch<Type, StringRef>(type)
                             .Case<QBEWordType>([&](auto) { return "w"; })
                             .Case<QBELongType>([&](auto) { return "l"; })
                             .Case<QBESingleType>([&](auto) { return "s"; })
                             .Case<QBEDoubleType>([&](auto) { return "d"; })
                             .Default([&](Type) { return ""; });

  if (name.empty())
    return emitError(loc) << "unable to print type " << type;

  os << name;
  return success();
}

LogicalResult Emitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<ModuleOp, FuncOp, ReturnOp, AddOp, SubOp, MulOp, DivOp, UDivOp,
                RemOp, URemOp, OrOp, XorOp, AndOp, SarOp, ShrOp, ShlOp, NegOp,
                JmpOp, JnzOp, HaltOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<ConstantOp>([](auto) { return success(); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });
  if (failed(status)) {
    return status;
  }
  if (!isa<ConstantOp>(op))
    os << "\n";
  return success();
}
} // namespace

LogicalResult translateToQBE(Operation *op, raw_ostream &os) {
  Emitter emitter(os);
  return emitter.emitOperation(*op);
}
} // namespace mlir::qbe
