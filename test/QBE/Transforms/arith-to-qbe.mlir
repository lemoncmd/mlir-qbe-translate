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

// -----

func.func @const() -> (i32, i64, f32, f64) {
  // CHECK: %{{.*}} = qbe.mlir.constant(32 : i32) : !qbe.word
  // CHECK: %{{.*}} = qbe.mlir.constant(64 : i64) : !qbe.long
  // CHECK: %{{.*}} = qbe.mlir.constant(3.2{{0*}}e+01 : f32) : !qbe.single
  // CHECK: %{{.*}} = qbe.mlir.constant(6.4{{0*}}e+01 : f64) : !qbe.double
  %a = arith.constant 32 : i32
  %b = arith.constant 64 : i64
  %c = arith.constant 32.0 : f32
  %d = arith.constant 64.0 : f64
  return %a, %b, %c, %d : i32, i64, f32, f64
}

// -----

func.func @comp_w(%a: i32, %b: i32) {
  // CHECK: %{{.*}} = qbe.ceq %[[A:.*]], %[[B:.*]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.csle %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cslt %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.csge %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.csgt %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cule %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cult %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cuge %[[A]], %[[B]] : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cugt %[[A]], %[[B]] : !qbe.word
  %0 = arith.cmpi eq, %a, %b : i32
  %1 = arith.cmpi ne, %a, %b : i32
  %2 = arith.cmpi sle, %a, %b : i32
  %3 = arith.cmpi slt, %a, %b : i32
  %4 = arith.cmpi sge, %a, %b : i32
  %5 = arith.cmpi sgt, %a, %b : i32
  %6 = arith.cmpi ule, %a, %b : i32
  %7 = arith.cmpi ult, %a, %b : i32
  %8 = arith.cmpi uge, %a, %b : i32
  %9 = arith.cmpi ugt, %a, %b : i32
  return
}

// -----

func.func @comp_s(%a: f32, %b: f32) {
  // CHECK: %{{.*}} = qbe.mlir.constant(0 : i32) : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.ceq %[[ARG0:.*]], %[[ARG1:.*]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cgt %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cge %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.clt %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cle %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.co %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.ceq %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cgt %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cge %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.clt %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cle %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[ARG0]], %[[ARG1]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.mlir.constant(1 : i32) : !qbe.word
  %0 = arith.cmpf false, %a, %b : f32
  %1 = arith.cmpf oeq, %a, %b : f32
  %2 = arith.cmpf ogt, %a, %b : f32
  %3 = arith.cmpf oge, %a, %b : f32
  %4 = arith.cmpf olt, %a, %b : f32
  %5 = arith.cmpf ole, %a, %b : f32
  %6 = arith.cmpf one, %a, %b : f32
  %7 = arith.cmpf ord, %a, %b : f32
  %8 = arith.cmpf ueq, %a, %b : f32
  %9 = arith.cmpf ugt, %a, %b : f32
  %10 = arith.cmpf uge, %a, %b : f32
  %11 = arith.cmpf ult, %a, %b : f32
  %12 = arith.cmpf ule, %a, %b : f32
  %13 = arith.cmpf une, %a, %b : f32
  %14 = arith.cmpf uno, %a, %b : f32
  %15 = arith.cmpf true, %a, %b : f32
  return
}
