// reference: https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch4/toyc.cpp

// build:
// export PREFIX=/home/aslan/workspace/llvm-project/build
// cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DTRITON_DIR=/home/aslan/workspace/triton
// VERBOSE=1 cmake --build .
//
// run:
// ./bin/loadMLIR triton/test/Triton/vecadd.mlir

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/InitAllPasses.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include "triton/Conversion/NVGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/PTXTranslation.h"
#include "triton/Target/PTX/TmaMetadata.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Tools/Sys/GetPlatform.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));


int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  return 0;
}

int optTTIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
    
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
        llvm::errs() << "failed to applyPassManagerCLOptions pm.\n";
      return 4;
    }
    

    auto printingFlags = mlir::OpPrintingFlags();
    printingFlags.elideLargeElementsAttrs(16);
    printingFlags.enableDebugInfo();
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/nullptr,
        /*shouldPrintAfterPass=*/
        [](mlir::Pass *pass, mlir::Operation *) {
            return false;
        },
        /*printModuleScope=*/false,
        /*printAfterOnlyOnChange=*/true,
        /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    pm.addPass(mlir::triton::createRewriteTensorPointerPass(86/*computeCapability*/));

    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::triton::createCombineOpsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createReorderBroadcastPass());
    
    pm.addPass(mlir::createCSEPass());   
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (mlir::failed(pm.run(*module))){
        llvm::errs() << "failed to run pm.\n";
        return 4;
    }
      
    
    module->dump();
    
    return 0;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  //context.getOrLoadDialect<mlir::triton::TritonDialect>();
  mlir::DialectRegistry registry;
  registry.insert<
    mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
    mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
    mlir::triton::nvgpu::NVGPUDialect, mlir::math::MathDialect,
    mlir::arith::ArithDialect, mlir::index::IndexDialect,
    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
    mlir::LLVM::LLVMDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  //
  mlir::registerTritonPasses();
  mlir::registerTritonGPUPasses();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  //mlir::triton::registerConvertTritonGPUToLLVMPass();
  //mlir::triton::registerConvertNVGPUToLLVMPass();
  //

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error; 

  module->dump();

  if (enableOpt) {
    llvm::errs() << "Optimized:\n";
    optTTIR(context, module);
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "triton compiler\n");

  dumpMLIR();

  return 0;
}