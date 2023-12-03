// reference: https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch4/toyc.cpp

// build:
// export PREFIX=/home/aslan/workspace/llvm-project/build
// cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DTRITON_DIR=/home/aslan/workspace/triton -DLLVM_LIBRARY_DIR=$PREFIX/lib
// VERBOSE=1 cmake --build .
//
// run:
// ./cubin --mlir-print-ir-before-all  ../add_kernel.ttir 

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
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"


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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileUtilities.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <signal.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));


int loadMLIR( mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

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

int optimize_ttir(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
    
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
        llvm::errs() << "failed to applyPassManagerCLOptions pm.\n";
      return 4;
    }
    
    pm.getContext()->disableMultithreading();
    auto printingFlags = mlir::OpPrintingFlags();
    printingFlags.elideLargeElementsAttrs(16);
    printingFlags.enableDebugInfo();
    auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/print_always,
        /*shouldPrintAfterPass=*/print_always,
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);

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


int ttir_to_ttgir(mlir::OwningOpRef<mlir::ModuleOp> &module, 
             int numWarps, int threadsPerWarp, 
             int numCTAs, int computeCapability){

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
      llvm::errs() << "failed to applyPassManagerCLOptions pm.\n";
    return 4;
  }    

  pm.getContext()->disableMultithreading();
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  printingFlags.enableDebugInfo();
  auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/print_always,
      /*shouldPrintAfterPass=*/print_always,
      /*printModuleScope=*/true,
      /*printAfterOnlyOnChange=*/false,
      /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
                numWarps, threadsPerWarp, numCTAs, computeCapability));
  
  if (mlir::failed(pm.run(*module))){
        llvm::errs() << "failed to run pm.\n";
        return 4;
  }

  llvm::outs() << "TTGIR\n"; 
  module->dump();  

  return 0;
}

namespace ttng = mlir::triton::nvidia_gpu;
ttng::ClusterInfo clusterInfo = ttng::ClusterInfo();

int optimize_ttgir(mlir::OwningOpRef<mlir::ModuleOp> &module, int num_stages, int num_warps, int num_ctas, 
  int capability, ttng::ClusterInfo clusterInfo, 
  bool enable_warp_specialization, bool enable_persistent, bool optimize_epilogue){
    mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
      llvm::errs() << "failed to applyPassManagerCLOptions pm.\n";
    return 4;
  }    

  pm.getContext()->disableMultithreading();
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  printingFlags.enableDebugInfo();
  auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/print_always,
      /*shouldPrintAfterPass=*/print_always,
      /*printModuleScope=*/true,
      /*printAfterOnlyOnChange=*/false,
      /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&clusterInfo));

  //CUDA
  pm.addPass(mlir::createTritonGPURewriteTensorPointerPass(capability));
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(&clusterInfo));

  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());

  //is_cuda
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(capability));

  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());

  //optimize_epilogue

  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::createTritonGPUPipelinePass(
                 num_stages, num_warps, num_ctas, capability));
  
  pm.addPass(mlir::createTritonNvidiaGPUMaterializeLoadStorePass(
                 num_warps, capability));
  
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());

  pm.addPass(mlir::createTritonNvidiaGPUWSFixupMissingAttrs());
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  //if capability // 10 >= 9

  pm.addPass(mlir::createTritonNvidiaGPUWSFixupMissingAttrs());

  pm.addPass(mlir::createTritonGPUOptimizeThreadLocalityPass());

  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module))){
        llvm::errs() << "failed to run optimize_ttgir pm.\n";
        return 4;
  }

  module->dump();
    
  return 0;

}

//https://github.com/openai/triton/blob/main/include/triton/Target/PTX/TmaMetadata.h
std::string ttgir_to_llir(mlir::OwningOpRef<mlir::ModuleOp> &module, int computeCapability,
                           mlir::triton::gpu::TMAMetadataTy &tmaInfos,
                           mlir::triton::Target target){
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::triton::translateTritonGPUToLLVMIR(
            &llvmContext, *module, computeCapability, tmaInfos, target);
  if (!llvmModule)
    llvm::report_fatal_error("Failed to translate TritonGPU to LLVM IR.");

  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();

  llvm::outs() << "LLVM IR string\n"; 
  llvm::outs() << str;

  llvm::outs() << "LLVM module dump\n"; 
  llvmModule->dump();

  return str;
}

std::string llir_to_ptx(const std::string llvmIR, int capability, int version,
         bool enable_fp_fusion){
  llvm::LLVMContext context;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), error, context);
  if (!module) {
    llvm::report_fatal_error(
        "failed to parse IR: " + error.getMessage() +
        "lineno: " + std::to_string(error.getLineNo()));
  }

  auto ptxCode = triton::translateLLVMIRToPTX(*module, capability,
                                                    version, enable_fp_fusion);
  return ptxCode;
}

std::string compile_ptx_to_cubin(const std::string &ptxCode, const std::string &ptxasPath,
           int capability, bool enable_fp_fusion){
  
  std::string cubin;
  // compile ptx with ptxas
  llvm::SmallString<64> fsrc;
  llvm::SmallString<64> flog;
  llvm::sys::fs::createTemporaryFile("compile-ptx-src", "", fsrc);
  llvm::sys::fs::createTemporaryFile("compile-ptx-log", "", flog);
  std::string fbin = std::string(fsrc) + ".o";
  llvm::FileRemover logRemover(flog);
  llvm::FileRemover binRemover(fbin);
  const char *_fsrc = fsrc.c_str();
  const char *_flog = flog.c_str();
  const char *_fbin = fbin.c_str();
  std::ofstream ofs(_fsrc);
  ofs << ptxCode << std::endl;
  ofs.close();

  auto lineInfoOption =
      triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO")
          ? ""
          : " -lineinfo";
  auto fmadOption = enable_fp_fusion ? "" : " --fmad=false";
  auto capabilitySuffix = (capability == 90) ? "a " : " ";
  auto outputFileName = std::string(_fsrc) + ".o";
  auto logRedirect = " 2> " + std::string(_flog);
  std::string cmd = ptxasPath + lineInfoOption + fmadOption +
                    " -v --gpu-name=sm_" +
                    std::to_string(capability) + capabilitySuffix +
                    _fsrc + " -o " + outputFileName + logRedirect;

  int err = system(cmd.c_str());
  if (err != 0) {
    err >>= 8;
    std::ifstream _log(_flog);
    std::string log(std::istreambuf_iterator<char>(_log), {});
    if (err == 255) {
      throw std::runtime_error(
          "Internal Triton PTX codegen error: \n" + log);
    } else if (err == 128 + SIGSEGV) {
      throw  std::runtime_error("Please run `ptxas " +
                                fsrc.str().str() +
                                "` to confirm that this is a "
                                "bug in `ptxas`\n" +
                                log);
    } else {
      throw std::runtime_error("`ptxas` failed with error code " +
                                std::to_string(err) + ": \n" + log);
    }
    return {};
  } else {
    llvm::FileRemover srcRemover(fsrc);
    std::ifstream _cubin(_fbin, std::ios::binary);
    cubin = std::string(std::istreambuf_iterator<char>(_cubin), {});
    _cubin.close();
    // Do not return here, exit the gil scope and return below
  }

  return cubin;

}


int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "triton compiler\n");

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

  llvm::outs() << "Load MLIR\n"; 
  loadMLIR(context, module);

  module->dump();

  llvm::outs() << "optimize_ttir\n"; 
  optimize_ttir(context, module);

  llvm::outs() << "ttir_to_ttgir\n"; 
  ttir_to_ttgir(module,  4, 32, 1, 86);
  //module->dump();  

  optimize_ttgir(module, 3, 4, 1, 86, clusterInfo, false, false, false);

  //
  mlir::triton::gpu::TMAMetadataTy tmaInfos;

  auto llvmIR = ttgir_to_llir(module, 86, tmaInfos,  mlir::triton::Target::NVVM);

  auto ptxCode = llir_to_ptx(llvmIR, 86, 80, true);
  llvm::outs() << "PTX Code\n"; 
  llvm::outs() << ptxCode;

  auto cubin = compile_ptx_to_cubin(ptxCode, 
      std::string("/usr/local/cuda-12.0/bin/ptxas"), 86, true);
  llvm::outs() << "cubin\n";
  //binary 
  //llvm::outs() << cubin;

  return 0;
}