// RUN: qbe-opt -split-input-file %s | FileCheck %s

func.func @add(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.add %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.add %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @sub(%a: !qbe.long, %b: !qbe.long) -> !qbe.long {
  // CHECK: %{{.*}} = qbe.sub %{{.*}}, %{{.*}} : !qbe.long
  %c = qbe.sub %a, %b : !qbe.long
  return %c : !qbe.long
}

// -----

func.func @mul(%a: !qbe.single, %b: !qbe.single) -> !qbe.single {
  // CHECK: %{{.*}} = qbe.mul %{{.*}}, %{{.*}} : !qbe.single
  %c = qbe.mul %a, %b : !qbe.single
  return %c : !qbe.single
}

// -----

func.func @div(%a: !qbe.double, %b: !qbe.double) -> !qbe.double {
  // CHECK: %{{.*}} = qbe.div %{{.*}}, %{{.*}} : !qbe.double
  %c = qbe.div %a, %b : !qbe.double
  return %c : !qbe.double
}

// -----

func.func @neg(%a: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.neg %{{.*}} : !qbe.word
  %b = qbe.neg %a : !qbe.word
  return %b : !qbe.word
}

// -----

func.func @udiv(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.udiv %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.udiv %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @rem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.rem %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.rem %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @urem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.urem %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.urem %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @or(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.or %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.or %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @xor(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.xor %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.xor %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @and(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.and %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.and %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @sar(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.sar %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.sar %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @shr(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.shr %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.shr %a, %b : !qbe.word
  return %c : !qbe.word
}

// -----

func.func @shl(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} = qbe.shl %{{.*}}, %{{.*}} : !qbe.word
  %c = qbe.shl %a, %b : !qbe.word
  return %c : !qbe.word
}

