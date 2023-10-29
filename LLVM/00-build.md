# Build with LLVM

llvm-project-wrappers

1. https://github.com/llvm/llvm-project/tree/main/cmake
2. https://github.com/llvm/llvm-project/tree/main/mlir/cmake/modules
3. https://llvm.org/docs/CMake.html
4. https://llvm.org/docs/CMakePrimer.html
5. https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
6. https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone



## Sample

Here is an example to build [Ch-2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) out of LLVM tree. The new **CMakeLists.txt**:

```
#Sample

cmake_minimum_required(VERSION 3.20.0)
project(toyc-ch2)

#
set(CMAKE_CXX_STANDARD 17)

set(LLVM_LINK_COMPONENTS
          Support
            )


#
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_subdirectory(include)


add_llvm_executable(toyc-ch2 
        toyc.cpp
        parser/AST.cpp
        mlir/MLIRGen.cpp
        mlir/Dialect.cpp

        DEPENDS
          ToyCh2OpsIncGen)


target_link_libraries(toyc-ch2
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
```

And building commanline:

```
export PREFIX=/home/aslan/workspace/llvm-project/build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir
cmake --build .
```
