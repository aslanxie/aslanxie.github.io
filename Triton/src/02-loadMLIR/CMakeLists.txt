cmake_minimum_required(VERSION 3.20.0)
project(loadMLIR)

set(CMAKE_CXX_STANDARD 17)

set(LLVM_LINK_COMPONENTS Support)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set (TRITON_DIR ${TRITON_DIR})


list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#message(STATUS "dialect_libs: ${dialect_libs}")


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${TRITON_DIR})
include_directories(${TRITON_DIR}/include)
include_directories(${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/include)



add_llvm_executable(loadMLIR
        main.cpp
		)


target_link_libraries(loadMLIR
  PRIVATE
    ${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/Triton/IR/libTritonIR.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/TritonGPU/IR/libTritonGPUIR.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/Triton/Transforms/libTritonTransforms.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/TritonNvidiaGPU/IR/libTritonNvidiaGPUIR.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/TritonNvidiaGPU/Transforms/libTritonNvidiaGPUTransforms.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/NVGPU/IR/libNVGPUIR.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Conversion/TritonGPUToLLVM/libTritonGPUToLLVM.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Analysis/libTritonAnalysis.a
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Conversion/TritonToTritonGPU/libTritonToTritonGPU.a	
	${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a
	${dialect_libs}
    
	MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)