// RUN: qbe-opt -convert-func-to-qbe -split-input-file %s | FileCheck %s

// CHECK: qbe.func @none()
func.func @none() {
  // CHECK: qbe.return
  return
}

// -----

// CHECK: qbe.func @one_operand(%{{.*}}: !qbe.word)
func.func @one_operand(%a: i32) {
  // CHECK: qbe.return
  return
}

// -----

// CHECK: qbe.func @one_operand(%{{.*}}: !qbe.word, %{{.*}}: !qbe.single, %{{.*}}: !qbe.long)
func.func @one_operand(%a: i32, %b: f32, %c: i64) {
  // CHECK: qbe.return
  return
}

// -----

// CHECK: qbe.func @ret(%[[VAR:.*]]: !qbe.word)
func.func @ret(%a: i32) -> i32 {
  // CHECK: qbe.return %[[VAR]] : !qbe.word
  return %a : i32
}

