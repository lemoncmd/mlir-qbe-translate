// RUN: qbe-opt -convert-func-to-qbe -split-input-file %s | FileCheck %s

// CHECK: qbe.func @none()
func.func @none() {
  // CHECK: qbe.return
  return
}

func.func @caller() {
  // CHECK: qbe.call @none() : () -> ()
  call @none() : () -> ()
  return
}

// -----

// CHECK: qbe.func @one_operand(%{{.*}}: !qbe.word)
func.func @one_operand(%a: i32) {
  // CHECK: qbe.return
  return
}

func.func @caller(%a: i32) {
  // CHECK: qbe.call @one_operand(%{{.*}}) : (!qbe.word) -> ()
  call @one_operand(%a) : (i32) -> ()
  return
}

// -----

// CHECK: qbe.func @multi_operands(%{{.*}}: !qbe.word, %{{.*}}: !qbe.single, %{{.*}}: !qbe.long)
func.func @multi_operands(%a: i32, %b: f32, %c: i64) {
  // CHECK: qbe.return
  return
}

func.func @caller(%a: i32, %b: f32, %c: i64) {
  // CHECK: qbe.call @multi_operands(%{{.*}}, %{{.*}}, %{{.*}}) : (!qbe.word, !qbe.single, !qbe.long) -> ()
  call @multi_operands(%a, %b, %c) : (i32, f32, i64) -> ()
  return
}

// -----

// CHECK: qbe.func @ret(%[[VAR:.*]]: !qbe.word)
func.func @ret(%a: i32) -> i32 {
  // CHECK: qbe.return %[[VAR]] : !qbe.word
  return %a : i32
}

func.func @caller(%a: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.call @ret(%{{.*}}) : (!qbe.word) -> !qbe.word
  %b = call @ret(%a) : (i32) -> i32
  return %b : i32
}

