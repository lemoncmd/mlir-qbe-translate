// RUN: qbe-translate -mlir-to-qbeir -split-input-file %s | FileCheck %s

// CHECK: function $none()
qbe.func @none() {
  // CHECK: @block1
  // CHECK: ret
  qbe.return
}

// -----

// CHECK: function $one_operand(w %{{.*}})
qbe.func @one_operand(%a: !qbe.word) {
  // CHECK: @block1
  // CHECK: ret
  qbe.return
}

// -----

// CHECK: function $multi_operands(w %{{.*}}, s %{{.*}}, l %{{.*}})
qbe.func @multi_operands(%a: !qbe.word, %b: !qbe.single, %c: !qbe.long) {
  // CHECK: @block1
  // CHECK: ret
  qbe.return
}

// -----

// CHECK: function w $hasret(w %[[VAR:.*]])
qbe.func @hasret(%a: !qbe.word) -> !qbe.word {
  // CHECK: @block1
  // CHECK: ret %[[VAR]]
  qbe.return %a : !qbe.word
}
