// RUN: qbe-opt -split-input-file %s | FileCheck %s

qbe.func @add(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.add %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.add %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sub(%a: !qbe.long, %b: !qbe.long) -> !qbe.long {
  // CHECK: %{{.*}} = qbe.sub %{{.*}}, %{{.*}} : !qbe.long
  %c = qbe.sub %a, %b : !qbe.long
  qbe.return %c : !qbe.long
}

// -----

qbe.func @mul(%a: !qbe.single, %b: !qbe.single) -> !qbe.single {
  // CHECK: %{{.*}} = qbe.mul %{{.*}}, %{{.*}} : !qbe.single
  %c = qbe.mul %a, %b : !qbe.single
  qbe.return %c : !qbe.single
}

// -----

qbe.func @div(%a: !qbe.double, %b: !qbe.double) -> !qbe.double {
  // CHECK: %{{.*}} = qbe.div %{{.*}}, %{{.*}} : !qbe.double
  %c = qbe.div %a, %b : !qbe.double
  qbe.return %c : !qbe.double
}

// -----

qbe.func @neg(%a: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.neg %{{.*}} : !qbe.word
  %b = qbe.neg %a : !qbe.word
  qbe.return %b : !qbe.word
}

// -----

qbe.func @udiv(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.udiv %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.udiv %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @rem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.rem %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.rem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @urem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.urem %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.urem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @or(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.or %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @xor(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.xor %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.xor %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @and(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.and %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sar(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.sar %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.sar %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shr(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.shr %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.shr %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shl(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.shl %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.shl %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @const() -> (!qbe.word, !qbe.long, !qbe.single, !qbe.double) {
  // CHECK: %{{.*}} = qbe.mlir.constant(32 : i32) : !qbe.word
  // CHECK: %{{.*}} = qbe.mlir.constant(64 : i64) : !qbe.long
  // CHECK: %{{.*}} = qbe.mlir.constant(3.2{{0*}}e+01 : f32) : !qbe.single
  // CHECK: %{{.*}} = qbe.mlir.constant(6.4{{0*}}e+01 : f64) : !qbe.double
  %a = qbe.mlir.constant(32 : i32) : !qbe.word
  %b = qbe.mlir.constant(64) : !qbe.long
  %c = qbe.mlir.constant(32.0 : f32) : !qbe.single
  %d = qbe.mlir.constant(64.0 : f64) : !qbe.double
  qbe.return %a, %b, %c, %d : !qbe.word, !qbe.long, !qbe.single, !qbe.double
}

// -----

qbe.func @comp_w(%a: !qbe.word, %b: !qbe.word) {
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
  %0 = qbe.ceq %a, %b : !qbe.word
  %1 = qbe.cne %a, %b : !qbe.word
  %2 = qbe.csle %a, %b : !qbe.word
  %3 = qbe.cslt %a, %b : !qbe.word
  %4 = qbe.csge %a, %b : !qbe.word
  %5 = qbe.csgt %a, %b : !qbe.word
  %6 = qbe.cule %a, %b : !qbe.word
  %7 = qbe.cult %a, %b : !qbe.word
  %8 = qbe.cuge %a, %b : !qbe.word
  %9 = qbe.cugt %a, %b : !qbe.word
  qbe.return
}

// -----

qbe.func @comp_l(%a: !qbe.long, %b: !qbe.long) {
  // CHECK: %{{.*}} = qbe.ceq %[[A:.*]], %[[B:.*]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.csle %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cslt %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.csge %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.csgt %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cule %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cult %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cuge %[[A]], %[[B]] : !qbe.long
  // CHECK-NEXT: %{{.*}} = qbe.cugt %[[A]], %[[B]] : !qbe.long
  %0 = qbe.ceq %a, %b : !qbe.long
  %1 = qbe.cne %a, %b : !qbe.long
  %2 = qbe.csle %a, %b : !qbe.long
  %3 = qbe.cslt %a, %b : !qbe.long
  %4 = qbe.csge %a, %b : !qbe.long
  %5 = qbe.csgt %a, %b : !qbe.long
  %6 = qbe.cule %a, %b : !qbe.long
  %7 = qbe.cult %a, %b : !qbe.long
  %8 = qbe.cuge %a, %b : !qbe.long
  %9 = qbe.cugt %a, %b : !qbe.long
  qbe.return
}

// -----

qbe.func @comp_s(%a: !qbe.single, %b: !qbe.single) {
  // CHECK: %{{.*}} = qbe.ceq %[[A:.*]], %[[B:.*]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cle %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.clt %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cge %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cgt %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.co %[[A]], %[[B]] : !qbe.single
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[A]], %[[B]] : !qbe.single
  %0 = qbe.ceq %a, %b : !qbe.single
  %1 = qbe.cne %a, %b : !qbe.single
  %2 = qbe.cle %a, %b : !qbe.single
  %3 = qbe.clt %a, %b : !qbe.single
  %4 = qbe.cge %a, %b : !qbe.single
  %5 = qbe.cgt %a, %b : !qbe.single
  %6 = qbe.co %a, %b : !qbe.single
  %7 = qbe.cuo %a, %b : !qbe.single
  qbe.return
}

// -----

qbe.func @comp_d(%a: !qbe.double, %b: !qbe.double) {
  // CHECK: %{{.*}} = qbe.ceq %[[A:.*]], %[[B:.*]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.cne %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.cle %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.clt %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.cge %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.cgt %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.co %[[A]], %[[B]] : !qbe.double
  // CHECK-NEXT: %{{.*}} = qbe.cuo %[[A]], %[[B]] : !qbe.double
  %0 = qbe.ceq %a, %b : !qbe.double
  %1 = qbe.cne %a, %b : !qbe.double
  %2 = qbe.cle %a, %b : !qbe.double
  %3 = qbe.clt %a, %b : !qbe.double
  %4 = qbe.cge %a, %b : !qbe.double
  %5 = qbe.cgt %a, %b : !qbe.double
  %6 = qbe.co %a, %b : !qbe.double
  %7 = qbe.cuo %a, %b : !qbe.double
  qbe.return
}
