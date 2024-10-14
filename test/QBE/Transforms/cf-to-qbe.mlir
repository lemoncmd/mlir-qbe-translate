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

