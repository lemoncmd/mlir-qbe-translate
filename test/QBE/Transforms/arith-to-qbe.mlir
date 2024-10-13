// RUN: qbe-opt -convert-arith-to-qbe -split-input-file %s | FileCheck %s

func.func @addi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.add %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.addi %a, %b : i32
  return %c : i32
}

// -----

func.func @addf(%a: f32, %b: f32) -> f32 {
  // CHECK: %{{.*}} = qbe.add %{{.*}}, %{{.*}} : !qbe.single
  %c = arith.addf %a, %b : f32
  return %c : f32
}

// -----

func.func @subi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.sub %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.subi %a, %b : i32
  return %c : i32
}

// -----

func.func @subf(%a: f32, %b: f32) -> f32 {
  // CHECK: %{{.*}} = qbe.sub %{{.*}}, %{{.*}} : !qbe.single
  %c = arith.subf %a, %b : f32
  return %c : f32
}

// -----

func.func @muli(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.mul %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.muli %a, %b : i32
  return %c : i32
}

// -----

func.func @mulf(%a: f32, %b: f32) -> f32 {
  // CHECK: %{{.*}} = qbe.mul %{{.*}}, %{{.*}} : !qbe.single
  %c = arith.mulf %a, %b : f32
  return %c : f32
}

// -----

func.func @divsi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.div %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.divsi %a, %b : i32
  return %c : i32
}

// -----

func.func @divf(%a: f32, %b: f32) -> f32 {
  // CHECK: %{{.*}} = qbe.div %{{.*}}, %{{.*}} : !qbe.single
  %c = arith.divf %a, %b : f32
  return %c : f32
}

// -----

func.func @negf(%a: f32) -> f32 {
  // CHECK: %{{.*}} = qbe.neg %{{.*}} : !qbe.single
  %b = arith.negf %a : f32
  return %b : f32
}

// -----

func.func @divui(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.udiv %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.divui %a, %b : i32
  return %c : i32
}

// -----

func.func @remsi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.rem %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.remsi %a, %b : i32
  return %c : i32
}

// -----

func.func @remui(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.urem %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.remui %a, %b : i32
  return %c : i32
}

// -----

func.func @ori(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.ori %a, %b : i32
  return %c : i32
}

// -----

func.func @xori(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.xor %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.xori %a, %b : i32
  return %c : i32
}

// -----

func.func @andi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.andi %a, %b : i32
  return %c : i32
}

// -----

func.func @shrsi(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.sar %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.shrsi %a, %b : i32
  return %c : i32
}

// -----

func.func @shrui(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.shr %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.shrui %a, %b : i32
  return %c : i32
}

// -----

func.func @shli(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.shl %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.shli %a, %b : i32
  return %c : i32
}

