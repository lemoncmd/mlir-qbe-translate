// RUN: qbe-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: qbe.func @none()
qbe.func @none() {
  // CHECK: qbe.return
  qbe.return
}

// -----

// CHECK: qbe.func @one_operand(%{{.*}}: !qbe.word)
qbe.func @one_operand(%a: !qbe.word) {
  // CHECK: qbe.return
  qbe.return
}

// -----

// CHECK: qbe.func @one_operand(%{{.*}}: !qbe.word, %{{.*}}: !qbe.single, %{{.*}}: !qbe.long)
qbe.func @one_operand(%a: !qbe.word, %b: !qbe.single, %c: !qbe.long) {
  // CHECK: qbe.return
  qbe.return
}

// -----

// CHECK: qbe.func @ret(%[[VAR:.*]]: !qbe.word)
qbe.func @ret(%a: !qbe.word) -> !qbe.word {
  // CHECK: qbe.return %[[VAR]] : !qbe.word
  qbe.return %a : !qbe.word
}

// -----

// expected-error @+1 {{'qbe.func' op expected QBE dialect types for arguments, got 'i32'}}
qbe.func @one_operand(%a: !qbe.word, %b: i32, %c: !qbe.long) {
  qbe.return
}

// -----

// expected-error @+1 {{'qbe.func' op expected QBE dialect types for results, got 'i32'}}
qbe.func @one_operand(%a: !qbe.word, %b: !qbe.word, %c: !qbe.long) -> i32 {
  qbe.return
}

// -----

// expected-note @+1 {{return type declared here}}
qbe.func @one_operand(%a: !qbe.word, %b: !qbe.word, %c: !qbe.long) -> !qbe.word {
  // expected-error @+1 {{'qbe.return' op expected 1 result operands}}
  qbe.return
}

// -----

qbe.func @one_operand(%a: !qbe.word, %b: !qbe.word, %c: !qbe.long) -> !qbe.word {
  %d = arith.constant 0 : i32
  // expected-error @+1 {{'qbe.return' op unexpected type 'i32' for operand #0}}
  qbe.return %d : i32
}

