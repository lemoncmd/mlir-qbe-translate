// RUN: qbe-opt -convert-cf-to-qbe -split-input-file %s | FileCheck %s

func.func @jmp() {
  // CHECK: qbe.jmp ^[[BB1:.*]]
  // CHECK-NEXT: ^[[BB1]]:
  cf.br ^bb1
  ^bb1:
  return
}

// -----

func.func @jnz(%a: i1, %b: i32, %c: i32) -> i32 {
  cf.br ^bb1
  ^bb1:
  // CHECK: qbe.jnz %{{.*}}, ^[[BB2:.*]], ^[[BB3:.*]]
  // CHECK-NEXT: ^[[BB2]]:
  // CHECK-NEXT: return
  // CHECK-NEXT: ^[[BB3]]:
  // CHECK-NEXT: return
  cf.cond_br %a, ^bb2, ^bb3
  ^bb2:
  return %b : i32
  ^bb3:
  return %c : i32
}

// -----

func.func @jmp_with_phi() {
  // CHECK: qbe.jmp ^[[BB1:.*]](%{{.*}}, %{{.*}} : !qbe.long, !qbe.double)
  // CHECK: ^[[BB1]](%[[C:.*]]: !qbe.long, %[[D:.*]]: !qbe.double):
  // CHECK: qbe.jmp ^[[BB1]](%[[C]], %[[D]] : !qbe.long, !qbe.double)
  ^bb0:
  %a = arith.constant 0 : i64
  %b = arith.constant 0.0 : f64
  cf.br ^bb1(%a, %b : i64, f64)
  ^bb1(%c: i64, %d: f64):
  cf.br ^bb1(%c, %d : i64, f64)
}

// -----

func.func @jnz_with_phi(%arg0: i1) -> i32 {
  qbe.jmp ^bb1
^bb1:
  %a = arith.constant 0 : i32
  %b = arith.constant 0.0 : f64
  // CHECK: qbe.jnz %{{.*}}, ^[[BB2:.*]](%[[A:.*]] : !qbe.word), ^[[BB3:.*]](%[[A]], %{{.*}} : !qbe.word, !qbe.double)
  // CHECK: ^[[BB2]](%{{.*}}: !qbe.word):
  // CHECK: ^[[BB3]](%{{.*}}: !qbe.word, %{{.*}}: !qbe.double):
  cf.cond_br %arg0, ^bb2(%a : i32), ^bb3(%a, %b : i32, f64)
^bb2(%c: i32):
  return %c : i32
^bb3(%d: i32, %e: f64):
  return %d : i32
}
