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
