// RUN: qbe-translate -mlir-to-qbeir -split-input-file %s | FileCheck %s

// CHECK: function $none()
qbe.func @none() {
  // CHECK: @block0
  // CHECK: ret
  qbe.return
}

qbe.func @caller() {
  // CHECK: call $none()
  qbe.call @none() : () -> ()
  qbe.return
}

// -----

// CHECK: function $one_operand(w %{{.*}})
qbe.func @one_operand(%a: !qbe.word) {
  // CHECK: @block0
  // CHECK: ret
  qbe.return
}

qbe.func @caller(%a: !qbe.word) {
  // CHECK: call $one_operand(w %{{.*}})
  qbe.call @one_operand(%a) : (!qbe.word) -> ()
  qbe.return
}

// -----

// CHECK: function $multi_operands(w %{{.*}}, s %{{.*}}, l %{{.*}})
qbe.func @multi_operands(%a: !qbe.word, %b: !qbe.single, %c: !qbe.long) {
  // CHECK: @block0
  // CHECK: ret
  qbe.return
}

qbe.func @caller(%a: !qbe.word, %b: !qbe.single, %c: !qbe.long) {
  // CHECK: call $multi_operands(w %{{.*}}, s %{{.*}}, l %{{.*}})
  qbe.call @multi_operands(%a, %b, %c) : (!qbe.word, !qbe.single, !qbe.long) -> ()
  qbe.return
}

// -----

// CHECK: function w $hasret(w %[[VAR:.*]])
qbe.func @hasret(%a: !qbe.word) -> !qbe.word {
  // CHECK: @block0
  // CHECK: ret %[[VAR]]
  qbe.return %a : !qbe.word
}

qbe.func @caller(%a: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w call $hasret(w %{{.*}})
  %b = qbe.call @hasret(%a) : (!qbe.word) -> !qbe.word
  qbe.return %b : !qbe.word
}

