# mlir-qbe-translate
Lowers core dialects to qbe dialect, and translate to qbe ir.

## Build

```sh
$ mkdir build && cd build
$ cmake -G Ninja path/to/llvm-project/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=qbe-dialect \
    -DLLVM_EXTERNAL_QBE_DIALECT_SOURCE_DIR=../ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
$ cmake --build . --target check-qbe
```
