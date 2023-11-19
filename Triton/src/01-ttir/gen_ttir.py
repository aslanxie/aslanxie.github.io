import triton
import triton.language as tl

def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
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


from collections import namedtuple
from dataclasses import dataclass

from triton.runtime.jit import JITFunction
from triton.compiler.code_generator import ast_to_ttir

@dataclass
class CudaTargetDescriptor:
    capability: int
    num_warps: int
    enable_fp_fusion: bool

instance_descriptor = namedtuple("instance_descriptor",
                                 ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
                                 defaults=[set(), set(), set(), set()])

# args
signature = {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}
specialization = instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))
constants = {4: 1024}
debug = True
target = CudaTargetDescriptor(capability=86, num_warps=4, enable_fp_fusion=True)

fn = JITFunction(add_kernel)

ttir = ast_to_ttir(fn, signature,  specialization, constants, debug, target)
print(f"ttir:\n {ttir}")