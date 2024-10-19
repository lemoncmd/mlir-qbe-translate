// RUN: qbe-translate -mlir-to-qbeir -split-input-file %s | FileCheck %s

qbe.func @jmp() {
  // CHECK: jmp @[[BB1:.*]]
  // CHECK-NEXT: @[[BB1]]
  qbe.jmp ^bb1
  ^bb1:
  qbe.return
}

// -----

qbe.func @jnz(%arg0: !qbe.word, %arg1: !qbe.word, %arg2: !qbe.word) -> !qbe.word {
  qbe.jmp ^bb1
^bb1:
  // CHECK: jnz %{{.*}}, @[[BB2:.*]], @[[BB3:.*]]
  // CHECK-NEXT: @[[BB2]]
  // CHECK-NEXT: ret
  // CHECK-NEXT: @[[BB3]]
  // CHECK-NEXT: ret
  qbe.jnz %arg0, ^bb2, ^bb3
^bb2:
  qbe.return %arg1 : !qbe.word
^bb3:
  qbe.return %arg2 : !qbe.word
}

// -----

qbe.func @hlt() {
  // CHECK: hlt
  qbe.hlt
}

// -----

qbe.func @jmp_with_phi() {
  // CKECK: @[[BB0:.*]]
  // CKECK-NEXT: jmp @[[BB1:.*]]
  // CKECK-NEXT: @[[BB1]]
  // CKECK-NEXT: %[[VAL0:.*]] =l phi @[[BB1]] %[[VAL0]], @[[BB0]] 0
  // CKECK-NEXT: %[[VAL1:.*]] =d phi @[[BB1]] %[[VAL1]], @[[BB0]] d_0.{{0+}}e+00
  // CKECK-NEXT: jmp @block1
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
  // CHECK: jnz %{{.*}}, @[[BB2:.*]], @[[BB3:.*]]
  // CHECK-NEXT: @[[BB2]]
  // CHECK-NEXT: %[[VAL0:.*]] =w phi @[[BB1:.*]] 0
  // CHECK-NEXT: ret %[[VAL0]]
  // CHECK-NEXT: @[[BB3]]
  // CHECK-NEXT: %[[VAL1:.*]] =w phi @[[BB1]] 0
  // CHECK-NEXT: %{{.*}} =d phi @[[BB1]] d_0.{{0+}}e+00
  // CHECK-NEXT: ret %[[VAL1]]
  qbe.jnz %arg0, ^bb2(%a : !qbe.word), ^bb3(%a, %b : !qbe.word, !qbe.double)
^bb2(%c: !qbe.word):
  qbe.return %c : !qbe.word
^bb3(%d: !qbe.word, %e: !qbe.double):
  qbe.return %d : !qbe.word
}
