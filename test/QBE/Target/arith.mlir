// RUN: qbe-translate -mlir-to-qbeir -split-input-file %s | FileCheck %s

qbe.func @add(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w add %{{.*}}, %{{.*}}
  %c = qbe.add %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sub(%a: !qbe.long, %b: !qbe.long) -> !qbe.long {
  // CHECK: %{{.*}} =l sub %{{.*}}, %{{.*}}
  %c = qbe.sub %a, %b : !qbe.long
  qbe.return %c : !qbe.long
}

// -----

qbe.func @mul(%a: !qbe.single, %b: !qbe.single) -> !qbe.single {
  // CHECK: %{{.*}} =s mul %{{.*}}, %{{.*}}
  %c = qbe.mul %a, %b : !qbe.single
  qbe.return %c : !qbe.single
}

// -----

qbe.func @div(%a: !qbe.double, %b: !qbe.double) -> !qbe.double {
  // CHECK: %{{.*}} =d div %{{.*}}, %{{.*}}
  %c = qbe.div %a, %b : !qbe.double
  qbe.return %c : !qbe.double
}

// -----

qbe.func @neg(%a: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w neg %{{.*}}
  %b = qbe.neg %a : !qbe.word
  qbe.return %b : !qbe.word
}

// -----

qbe.func @udiv(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w udiv %{{.*}}, %{{.*}}
  %c = qbe.udiv %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @rem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w rem %{{.*}}, %{{.*}}
  %c = qbe.rem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @urem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w urem %{{.*}}, %{{.*}}
  %c = qbe.urem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @or(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w or %{{.*}}, %{{.*}}
  %c = qbe.or %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @xor(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w xor %{{.*}}, %{{.*}}
  %c = qbe.xor %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @and(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w and %{{.*}}, %{{.*}}
  %c = qbe.and %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sar(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w sar %{{.*}}, %{{.*}}
  %c = qbe.sar %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shr(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w shr %{{.*}}, %{{.*}}
  %c = qbe.shr %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shl(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w shl %{{.*}}, %{{.*}}
  %c = qbe.shl %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @const() -> !qbe.word {
  %a = qbe.mlir.constant(32 : i32) : !qbe.word
  // CHECK ret 32
  qbe.return %a : !qbe.word
}

// -----

qbe.func @sconst() -> !qbe.single {
  %a = qbe.mlir.constant(32.0 : f32) : !qbe.single
  // CHECK ret s_3.2{{0*}}e+01
  qbe.return %a : !qbe.single
}

// -----

qbe.func @dconst() -> !qbe.double {
  %a = qbe.mlir.constant(64.0 : f64) : !qbe.double
  // CHECK ret d_3.2{{0*}}e+01
  qbe.return %a : !qbe.double
}

// -----

qbe.func @op_prop() -> !qbe.long {
  %a = qbe.mlir.constant(64 : i64) : !qbe.long
  %b = qbe.mlir.constant(46 : i64) : !qbe.long
  // CHECK %{{.*}} = add 64, 46
  %c = qbe.add %a, %b : !qbe.long
  qbe.return %c : !qbe.long
}
