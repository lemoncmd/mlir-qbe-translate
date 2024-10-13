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
#include "mlir/IR/BuiltinOps.h"
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
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";

  Region::BlockListType &blocks = funcOp.getBlocks();

  emitter.resetValueCount();
  emitter.resetBlockCount();
  for (Block &block : blocks) {
    os << emitter.getOrCreateName(block) << "\n";
    os.indent();
    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op)))
        return failure();
    }
    os.unindent();
  }

  os << "}\n";
  return success();
}

static LogicalResult printOperation(Emitter &emitter, ReturnOp returnOp) {
  auto &os = emitter.ostream();
  os << "ret";
  if (returnOp.getNumOperands() == 1) {
    os << " " << emitter.getOrCreateName(returnOp.getOperands()[0]);
  }
  return success();
}

static LogicalResult printOperation(Emitter &emitter, AddOp addOp) {
  auto &os = emitter.ostream();
  os << emitter.getOrCreateName(addOp.getRes()) << " =";
  if (failed(emitter.emitType(addOp.getLoc(), addOp.getType())))
    return failure();
  os << " add " << emitter.getOrCreateName(addOp.getLhs()) << ", "
     << emitter.getOrCreateName(addOp.getRhs());
  return success();
}

StringRef Emitter::getOrCreateName(Value value, StringRef prefix) {
  if (!valueMapper.count(value))
    valueMapper.insert(value, formatv("%{0}{1}", prefix, ++valueCount));
  return *valueMapper.begin(value);
}

StringRef Emitter::getOrCreateName(Block &block, StringRef prefix) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("@{0}{1}", prefix, ++blockCount));
  return *blockMapper.begin(&block);
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
          .Case<ModuleOp, FuncOp, ReturnOp, AddOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });
  if (failed(status)) {
    return status;
  }
  os << "\n";
  return success();
}
} // namespace

LogicalResult translateToQBE(Operation *op, raw_ostream &os) {
  Emitter emitter(os);
  return emitter.emitOperation(*op);
}
} // namespace mlir::qbe
