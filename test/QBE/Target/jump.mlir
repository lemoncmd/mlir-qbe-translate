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

