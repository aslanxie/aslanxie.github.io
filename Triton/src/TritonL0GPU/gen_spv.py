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