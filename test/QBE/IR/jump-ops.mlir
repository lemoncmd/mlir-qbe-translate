// RUN: qbe-opt -split-input-file -verify-diagnostics %s | FileCheck %s

qbe.func @jmp() {
  // CHECK: qbe.jmp ^[[BB1:.*]]
  // CHECK-NEXT: ^[[BB1]]:
  qbe.jmp ^bb1
  ^bb1:
  qbe.return
}

// -----

qbe.func @jnz(%arg0: !qbe.word, %arg1: !qbe.word, %arg2: !qbe.word) -> !qbe.word {
  qbe.jmp ^bb1
^bb1:
  // CHECK: qbe.jnz %{{.*}}, ^[[BB2:.*]], ^[[BB3:.*]]
  // CHECK-NEXT: ^[[BB2]]:
  // CHECK-NEXT: qbe.return
  // CHECK-NEXT: ^[[BB3]]:
  // CHECK-NEXT: qbe.return
  qbe.jnz %arg0, ^bb2, ^bb3
^bb2:
  qbe.return %arg1 : !qbe.word
^bb3:
  qbe.return %arg2 : !qbe.word
}

// -----

qbe.func @hlt() {
  // CHECK: qbe.hlt
  qbe.hlt
}

// -----

qbe.func @jmp_with_phi() {
  // CHECK: qbe.jmp ^[[BB1:.*]](%{{.*}}, %{{.*}} : !qbe.long, !qbe.double)
  // CHECK-NEXT: ^[[BB1]](%[[C:.*]]: !qbe.long, %[[D:.*]]: !qbe.double):
  // CHECK: qbe.jmp ^[[BB1]](%[[C]], %[[D]] : !qbe.long, !qbe.double)
  ^bb0:
  %a = qbe.mlir.constant(0) : !qbe.long
  %b = qbe.mlir.constant(0.0) : !qbe.double
  qbe.jmp ^bb1(%a, %b : !qbe.long, !qbe.double)
  ^bb1(%c: !qbe.long, %d: !qbe.double):
  qbe.jmp ^bb1(%c, %d : !qbe.long, !qbe.double)
}

// -----

qbe.func @jnz_with_phi(%arg0: !qbe.word) -> !qbe.word {
  qbe.jmp ^bb1
^bb1:
  %a = qbe.mlir.constant(0 : i32) : !qbe.word
  %b = qbe.mlir.constant(0.0) : !qbe.double
  // CHECK: qbe.jnz %{{.*}}, ^[[BB2:.*]](%[[A:.*]] : !qbe.word), ^[[BB3:.*]](%[[A]], %{{.*}} : !qbe.word, !qbe.double)
  // CHECK-NEXT: ^[[BB2]](%[[C:.*]]: !qbe.word):
  // CHECK-NEXT: qbe.return %[[C]] : !qbe.word
  // CHECK-NEXT: ^[[BB3]](%[[D:.*]]: !qbe.word, %{{.*}}: !qbe.double):
  // CHECK-NEXT: qbe.return %[[D]] : !qbe.word
  qbe.jnz %arg0, ^bb2(%a : !qbe.word), ^bb3(%a, %b : !qbe.word, !qbe.double)
^bb2(%c: !qbe.word):
  qbe.return %c : !qbe.word
^bb3(%d: !qbe.word, %e: !qbe.double):
  qbe.return %d : !qbe.word
}
