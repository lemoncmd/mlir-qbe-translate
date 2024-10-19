// RUN: qbe-translate -mlir-to-qbeir -split-input-file %s | FileCheck %s

qbe.func @add(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w add %{{.*}}, %{{.*}}
  %c = qbe.add %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sub(%a: !qbe.long, %b: !qbe.long) -> !qbe.long {
  // CHECK: %{{.*}} =l sub %{{.*}}, %{{.*}}
  %c = qbe.sub %a, %b : !qbe.long
  qbe.return %c : !qbe.long
}

// -----

qbe.func @mul(%a: !qbe.single, %b: !qbe.single) -> !qbe.single {
  // CHECK: %{{.*}} =s mul %{{.*}}, %{{.*}}
  %c = qbe.mul %a, %b : !qbe.single
  qbe.return %c : !qbe.single
}

// -----

qbe.func @div(%a: !qbe.double, %b: !qbe.double) -> !qbe.double {
  // CHECK: %{{.*}} =d div %{{.*}}, %{{.*}}
  %c = qbe.div %a, %b : !qbe.double
  qbe.return %c : !qbe.double
}

// -----

qbe.func @neg(%a: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w neg %{{.*}}
  %b = qbe.neg %a : !qbe.word
  qbe.return %b : !qbe.word
}

// -----

qbe.func @udiv(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w udiv %{{.*}}, %{{.*}}
  %c = qbe.udiv %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @rem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w rem %{{.*}}, %{{.*}}
  %c = qbe.rem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @urem(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w urem %{{.*}}, %{{.*}}
  %c = qbe.urem %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @or(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w or %{{.*}}, %{{.*}}
  %c = qbe.or %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @xor(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w xor %{{.*}}, %{{.*}}
  %c = qbe.xor %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @and(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w and %{{.*}}, %{{.*}}
  %c = qbe.and %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @sar(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w sar %{{.*}}, %{{.*}}
  %c = qbe.sar %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shr(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w shr %{{.*}}, %{{.*}}
  %c = qbe.shr %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @shl(%a: !qbe.word, %b: !qbe.word) -> !qbe.word {
  // CHECK: %{{.*}} =w shl %{{.*}}, %{{.*}}
  %c = qbe.shl %a, %b : !qbe.word
  qbe.return %c : !qbe.word
}

// -----

qbe.func @const() -> !qbe.word {
  %a = qbe.mlir.constant(32 : i32) : !qbe.word
  // CHECK ret 32
  qbe.return %a : !qbe.word
}

// -----

qbe.func @sconst() -> !qbe.single {
  %a = qbe.mlir.constant(32.0 : f32) : !qbe.single
  // CHECK ret s_3.2{{0*}}e+01
  qbe.return %a : !qbe.single
}

// -----

qbe.func @dconst() -> !qbe.double {
  %a = qbe.mlir.constant(64.0 : f64) : !qbe.double
  // CHECK ret d_3.2{{0*}}e+01
  qbe.return %a : !qbe.double
}

// -----

qbe.func @op_prop() -> !qbe.long {
  %a = qbe.mlir.constant(64 : i64) : !qbe.long
  %b = qbe.mlir.constant(46 : i64) : !qbe.long
  // CHECK %{{.*}} = add 64, 46
  %c = qbe.add %a, %b : !qbe.long
  qbe.return %c : !qbe.long
}

// -----

qbe.func @comp_w(%a: !qbe.word, %b: !qbe.word) {
  // CHECK: %{{.*}} =w ceqw %[[A:.*]], %[[B:.*]]
  // CHECK-NEXT: %{{.*}} =w cnew %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cslew %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csltw %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csgew %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csgtw %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w culew %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cultw %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cugew %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cugtw %[[A]], %[[B]]
  %0 = qbe.ceq %a, %b : !qbe.word
  %1 = qbe.cne %a, %b : !qbe.word
  %2 = qbe.csle %a, %b : !qbe.word
  %3 = qbe.cslt %a, %b : !qbe.word
  %4 = qbe.csge %a, %b : !qbe.word
  %5 = qbe.csgt %a, %b : !qbe.word
  %6 = qbe.cule %a, %b : !qbe.word
  %7 = qbe.cult %a, %b : !qbe.word
  %8 = qbe.cuge %a, %b : !qbe.word
  %9 = qbe.cugt %a, %b : !qbe.word
  qbe.return
}

// -----

qbe.func @comp_l(%a: !qbe.long, %b: !qbe.long) {
  // CHECK: %{{.*}} =w ceql %[[A:.*]], %[[B:.*]]
  // CHECK-NEXT: %{{.*}} =w cnel %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cslel %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csltl %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csgel %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w csgtl %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w culel %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cultl %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cugel %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cugtl %[[A]], %[[B]]
  %0 = qbe.ceq %a, %b : !qbe.long
  %1 = qbe.cne %a, %b : !qbe.long
  %2 = qbe.csle %a, %b : !qbe.long
  %3 = qbe.cslt %a, %b : !qbe.long
  %4 = qbe.csge %a, %b : !qbe.long
  %5 = qbe.csgt %a, %b : !qbe.long
  %6 = qbe.cule %a, %b : !qbe.long
  %7 = qbe.cult %a, %b : !qbe.long
  %8 = qbe.cuge %a, %b : !qbe.long
  %9 = qbe.cugt %a, %b : !qbe.long
  qbe.return
}

// -----

qbe.func @comp_s(%a: !qbe.single, %b: !qbe.single) {
  // CHECK: %{{.*}} =w ceqs %[[A:.*]], %[[B:.*]]
  // CHECK-NEXT: %{{.*}} =w cnes %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cles %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w clts %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cges %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cgts %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cos %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cuos %[[A]], %[[B]]
  %0 = qbe.ceq %a, %b : !qbe.single
  %1 = qbe.cne %a, %b : !qbe.single
  %2 = qbe.cle %a, %b : !qbe.single
  %3 = qbe.clt %a, %b : !qbe.single
  %4 = qbe.cge %a, %b : !qbe.single
  %5 = qbe.cgt %a, %b : !qbe.single
  %6 = qbe.co %a, %b : !qbe.single
  %7 = qbe.cuo %a, %b : !qbe.single
  qbe.return
}

// -----

qbe.func @comp_d(%a: !qbe.double, %b: !qbe.double) {
  // CHECK: %{{.*}} =w ceqd %[[A:.*]], %[[B:.*]]
  // CHECK-NEXT: %{{.*}} =w cned %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cled %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cltd %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cged %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cgtd %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cod %[[A]], %[[B]]
  // CHECK-NEXT: %{{.*}} =w cuod %[[A]], %[[B]]
  %0 = qbe.ceq %a, %b : !qbe.double
  %1 = qbe.cne %a, %b : !qbe.double
  %2 = qbe.cle %a, %b : !qbe.double
  %3 = qbe.clt %a, %b : !qbe.double
  %4 = qbe.cge %a, %b : !qbe.double
  %5 = qbe.cgt %a, %b : !qbe.double
  %6 = qbe.co %a, %b : !qbe.double
  %7 = qbe.cuo %a, %b : !qbe.double
  qbe.return
}
