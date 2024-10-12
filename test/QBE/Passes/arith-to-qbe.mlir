// RUN: qbe-opt -convert-arith-to-qbe -split-input-file %s | FileCheck %s

func.func @add(%a: i32, %b: i32) -> i32 {
  // CHECK: %{{.*}} = qbe.add %{{.*}}, %{{.*}} : !qbe.word
  %c = arith.addi %a, %b : i32
  return %c : i32
}
