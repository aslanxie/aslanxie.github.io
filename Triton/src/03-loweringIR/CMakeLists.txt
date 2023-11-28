cmake_minimum_required(VERSION 3.20.0)
set(TARGET loweringIR)

project(${TARGET})

set(CMAKE_CXX_STANDARD 17)

set(LLVM_LINK_COMPONENTS Support)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set (TRITON_DIR ${TRITON_DIR})
set (TRITON_LIB_DIR ${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "dialect_libs: ${dialect_libs}")
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "conversion_libs: ${conversion_libs}")


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${TRITON_DIR})
include_directories(${TRITON_DIR}/include)
include_directories(${TRITON_DIR}/python/build/cmake.linux-x86_64-cpython-3.10/include)

add_llvm_executable(${TARGET}  main.cpp )

target_link_libraries(${TARGET}   PRIVATE
	${TRITON_LIB_DIR}/Dialect/Triton/Transforms/libTritonTransforms.a
	${TRITON_LIB_DIR}/Dialect/TritonGPU/Transforms/libTritonGPUTransforms.a	
	${TRITON_LIB_DIR}/Dialect/TritonNvidiaGPU/Transforms/libTritonNvidiaGPUTransforms.a
	${TRITON_LIB_DIR}/Target/LLVMIR/libTritonLLVMIR.a
	${TRITON_LIB_DIR}/Target/PTX/libTritonPTX.a
    ${TRITON_LIB_DIR}/Dialect/Triton/IR/libTritonIR.a
	${TRITON_LIB_DIR}/Dialect/Triton/Transforms/libTritonTransforms.a	
	${TRITON_LIB_DIR}/Dialect/TritonGPU/IR/libTritonGPUIR.a
	${TRITON_LIB_DIR}/Dialect/TritonNvidiaGPU/Transforms/libTritonNvidiaGPUTransforms.a	
	${TRITON_LIB_DIR}/Dialect/TritonNvidiaGPU/IR/libTritonNvidiaGPUIR.a
	${TRITON_LIB_DIR}/Dialect/TritonNvidiaGPU/Transforms/libTritonNvidiaGPUTransforms.a
	${TRITON_LIB_DIR}/Dialect/NVGPU/IR/libNVGPUIR.a
	${dialect_libs}
	${conversion_libs}
	${TRITON_LIB_DIR}/Conversion/TritonToTritonGPU/libTritonToTritonGPU.a
	${TRITON_LIB_DIR}/Conversion/TritonGPUToLLVM/libTritonGPUToLLVM.a	
	${TRITON_LIB_DIR}/Conversion/NVGPUToLLVM/libNVGPUToLLVM.a
	${TRITON_LIB_DIR}/Conversion/TritonGPUToLLVM/libASMBuilder.a	
	${TRITON_LIB_DIR}/Analysis/libTritonAnalysis.a

	MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
	MLIRMathToLLVM
	MLIRLLVMToLLVMIRTranslation
    MLIRNVVMToLLVMIRTranslation
	MLIRROCDLToLLVMIRTranslation
	MLIRGPUToNVVMTransforms
	MLIRGPUToROCDLTransforms
	)