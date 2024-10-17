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
