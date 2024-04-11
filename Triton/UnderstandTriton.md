# Understand Triton

 As the description from [Triton official site](https://triton-lang.org/main/index.html): "Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware". It shows:
1. Triton is a programming language specifically designed for GPU computing.
2. Triton includes a compiler for its own language, enabling the translation of Triton code into executable GPU code.
3. Triton is Python-based, with both the language and the compiler leveraging Python's syntax and features.
4. Triton enables the efficient and productive creation of custom deep neural network (DNN) compute kernels.
5. Compute kernels written in Triton can achieve maximum throughput on modern GPU hardware, ensuring optimal performance.
6. Triton compute kernels can be seamlessly launched from Python code, functioning like any other library function for ease of use and integration.

## From Python Code to SPV

Python code kernel

```
# Reference: https://github.com/intel/intel-xpu-backend-for-triton/blob/llvm-target/python/tutorials/01-vector-add.py
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```
Compile to SPV
```
import os
import torch
import intel_extension_for_pytorch
from typing import Any, Tuple
from dataclasses import dataclass
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, get_env_vars
from triton._C.libtriton import ir, passes, llvm, intel
from triton.language.extra.intel import convert_custom_float8

#################################################################
# Initialize parameters 
size = 1024
x = torch.rand(size, device='xpu')
y = torch.rand(size, device='xpu')
output = torch.rand(size, device='xpu')

kernel = add_kernel
args = [
    x,  # in_ptr0
    y,  # in_ptr1
    output,  # out_ptr
    size,  # n_elements
    16,  # BLOCK_SIZE
]

signature={i: kernel._type_of(kernel._key_of(arg))
                for i, arg in enumerate(args)
                if i not in kernel.constexprs}
constants={i: arg
                for i, arg in enumerate(args)
                if not isinstance(arg, torch.Tensor)}
attrs=kernel._get_config(*args, )
print(f"signature: {signature}")
print(f"constants: {constants}")
print(f"attrs: {attrs}")
	
src = triton.compiler.compiler.ASTSource(
    fn=kernel,
    signature=signature,
    constants=constants,
    attrs=attrs,
)

# https://github.com/intel/intel-xpu-backend-for-triton/blob/llvm-target/third_party/intel/backend/compiler.py#L34C1-L48C24
@dataclass(frozen=True)
class XPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    threads_per_warp: int = 32
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: int = 0  # `max_num_imprecise_acc` only applies to fp8 -> fp32 dot on sm_90 for cuda
    extern_libs: dict = None
    debug: bool = True

# https://github.com/intel/intel-xpu-backend-for-triton/blob/llvm-target/third_party/intel/backend/compiler.py#L209
codegen_fns = dict()
codegen_fns["convert_custom_types"] = convert_custom_float8

options = XPUOptions()
print(options)

target ={'xpu': {'max_work_group_size': 1024, 'max_num_sub_groups': 64, 'sub_group_sizes': [16, 32]}}
metadata = {
        "hash": "dummy",
        "target": target,
        **options.__dict__,
        **get_env_vars(),
    }
capability = intel.passes.ttgpuir.DEVICE_ARCH.PVC
#################################################################

# Create Triton IR context
context = ir.context()
ir.load_dialects(context)

# Reference: https://github.com/intel/intel-xpu-backend-for-triton/blob/llvm-target/third_party/intel/backend/compiler.py
# Transform to TTIR
module = src.make_ir(options, codegen_fns, context)
print("TTIR: ", module)
#################################################################

# Transform to TTGIR
def make_ttgir(mod, metadata, opt, capability):
	cluster_info = intel.ClusterInfo()
	if opt.cluster_dims is not None:
		cluster_info.clusterDimX = opt.cluster_dims[0]
		cluster_info.clusterDimY = opt.cluster_dims[1]
		cluster_info.clusterDimZ = opt.cluster_dims[2]
	# TTIR -> TTGIR
	pm = ir.pass_manager(mod.context)
	pm.enable_debug()
	passes.ttir.add_convert_to_ttgpuir(pm, opt.num_warps, opt.threads_per_warp, opt.num_ctas, capability)
	# optimize TTGIR
	passes.ttgpuir.add_coalesce(pm)
	# TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
	intel.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
	passes.ttgpuir.add_remove_layout_conversions(pm)
	passes.ttgpuir.add_optimize_thread_locality(pm)
	intel.passes.ttgpuir.add_accelerate_matmul(pm, intel.passes.ttgpuir.DEVICE_ARCH.PVC)
	passes.ttgpuir.add_remove_layout_conversions(pm)
	if opt.optimize_epilogue:
		passes.ttgpuir.add_optimize_epilogue(pm)
	passes.ttgpuir.add_optimize_dot_operands(pm)
	passes.common.add_cse(pm)
	passes.ttgpuir.add_prefetch(pm)
	passes.ttgpuir.add_optimize_dot_operands(pm)
	passes.ttgpuir.add_remove_layout_conversions(pm)
	passes.ttgpuir.add_reduce_data_duplication(pm)
	passes.ttgpuir.add_reorder_instructions(pm)
	passes.common.add_cse(pm)
	passes.common.add_symbol_dce(pm)
	passes.common.add_canonicalizer(pm)
	pm.run(mod)
	metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
	return mod

mod = make_ttgir(module, metadata, options, capability)
print("TTGIR: ", mod)
#################################################################

# Transform to LLIR
def make_llir(src, metadata, options, capability):
	# warp-specialization mutates num_warps
	num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
	if num_warp_groups is not None:
		metadata["num_warps"] *= num_warp_groups
	threads_per_warp = ir.ttgpuir.get_threads_per_warp(src)
	metadata["threads_per_warp"] = threads_per_warp
	mod = src
	# TritonGPU -> LLVM-IR (MLIR)
	pm = ir.pass_manager(mod.context)
	pm.enable_debug()
	intel.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
	passes.convert.add_scf_to_cf(pm)
	passes.convert.add_index_to_llvmir(pm)
	intel.passes.ttgpuir.add_allocate_shared_memory(pm)
	intel.passes.ttgpuir.add_to_llvmir(pm, capability)
	passes.convert.add_arith_to_llvmir(pm)
	passes.common.add_canonicalizer(pm)
	passes.common.add_cse(pm)
	passes.common.add_symbol_dce(pm)
	if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
		passes.llvmir.add_di_scope(pm)
	pm.run(mod)
	# LLVM-IR (MLIR) -> LLVM-IR (LLVM)
	llvm.init_targets()
	context = llvm.context()
	llvm_mod = llvm.to_module(mod, context)
	llvm.set_spv_target_triple(llvm_mod)
	if options.extern_libs:
		paths = [path for (name, path) in options.extern_libs]
		llvm.link_extern_libs(llvm_mod, paths)
	llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
	# Get some metadata
	metadata["shared"] = src.get_int_attr("triton_gpu.shared")
	ret = str(llvm_mod)
	del llvm_mod
	del context
	return ret
 
mod = make_llir(mod, metadata, options, capability)
print("LLIR: ", mod)
#################################################################

# Transform to SPV
def make_spv(src, metadata):
	ret, name = llvm.translate_to_spirv(src)
	metadata["name"] = name
	return ret

mod = make_spv(mod, metadata)
print("SPV: ", mod)
with open("add_kernel.spv", "wb") as file:
    file.write(mod)
    print("Save to add_kernel.spv")
```

The output
```
TTIR:  #loc = loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %1 = arith.muli %0, %c16_i32 : i32 loc(#loc2)
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %3 = tt.splat %1 : i32 -> tensor<16xi32> loc(#loc4)
    %4 = arith.addi %3, %2 : tensor<16xi32> loc(#loc4)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc5)
    %cst = arith.constant dense<1024> : tensor<16xi32> loc(#loc5)
    %5 = arith.cmpi slt, %4, %cst : tensor<16xi32> loc(#loc5)
  ...

TTGIR:  #blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0)
module attributes {"triton_gpu.compute-capability" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/dataset/aslan/workspace/Triton/gen_spv.py":6:0)) attributes {noinline = false} {
    %cst = arith.constant dense<1024> : tensor<16xi32, #blocked> loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c16_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<16xi32, #blocked> loc(#loc5)
...

LLIR:  ; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare !dbg !3 spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr

declare !dbg !6 spir_func i64 @_Z12get_group_idj(i32) local_unnamed_addr

define spir_kernel void @add_kernel(ptr addrspace(1) nocapture readonly %0, ptr addrspace(1) nocapture readonly %1, ptr addrspace(1) nocapture writeonly %2) local_unnamed_addr !dbg !7 !max_work_group_size !8 !intel_reqd_sub_group_size !9 {
  %4 = tail call i64 @_Z12get_group_idj(i32 0)
  %5 = trunc i64 %4 to i32, !dbg !10
  %6 = shl i32 %5, 4, !dbg !11
  %7 = tail call i64 @_Z12get_local_idj(i32 0)
  %8 = trunc i64 %7 to i32, !dbg !12
 
...

SPV:  b'\x03\x02#\x07\x00\x01\x01\x00\x0e\x00\x06\x00R\x00\x00\x00\x00\x00\x00\x00\x11\x00\x02\x00\x04\x00\x00\x00\x11\x00\x02\x00\x05\x00\x00\x00\x...
Save to add_kernel.spv
```

Dissamble spv to text format
```
spirv-dis add_kernel.spv 
; SPIR-V
; Version: 1.1
; Generator: Khronos LLVM/SPIR-V Translator; 14
; Bound: 82
; Schema: 0
               OpCapability Addresses
               OpCapability Linkage
               OpCapability Kernel
               OpCapability Int64
               OpCapability SubgroupDispatch
               OpCapability VectorAnyINTEL
               OpCapability KernelAttributesINTEL
               OpExtension "SPV_INTEL_kernel_attributes"
               OpExtension "SPV_INTEL_vector_compute"
          %2 = OpExtInstImport "OpenCL.DebugInfo.100"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %58 "add_kernel"
               OpExecutionMode %58 ContractionOff
               OpExecutionMode %58 SubgroupSize 32
               OpExecutionMode %58 MaxWorkgroupSizeINTEL 128 1 1
         %64 = OpString "/dataset/aslan/workspace/Triton/gen_spv.py"
         %68 = OpString "_Z12get_local_idj"
         %70 = OpString "_Z12get_group_idj"
         %72 = OpString "add_kernel"
               OpSource Unknown 0
               OpName %_Z12get_local_idj "_Z12get_local_idj"
               OpName %_Z12get_group_idj "_Z12get_group_idj"
               OpName %add_kernel "add_kernel"
               OpModuleProcessed "Debug info producer: triton"
               OpDecorate %_Z12get_local_idj LinkageAttributes "_Z12get_local_idj" Import
               OpDecorate %_Z12get_group_idj LinkageAttributes "_Z12get_group_idj" Import
               OpDecorate %add_kernel LinkageAttributes "add_kernel" Export
              ...
         %63 = OpFunctionCall %void %add_kernel %59 %60 %61
               OpReturn
               OpFunctionEnd
```

disasm device binary
```
ocloc disasm -file add_kernel.bin -device pvc

cat ./dump/.text.add_kernel.asm
L0:
(W)     mov (16|M0)              r127.0<1>:ud  0x0:ud                             
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw             
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x40:uw              {I@2}
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}
...        

```
