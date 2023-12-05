# python/triton/tools



## compile.py

It builds trition kernel to GPU kernel and generates C source code (from compile.c and compile.h template) with utilities to load, unload and launch the kernel. 

Take the kernel from [Tutorials: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) as example.

```
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

```

And run complie.py script to build out GPU kernel and utilities codes.
```
python compile.py --kernel-name add_kernel \
                  --signature "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
                  --out-path /home/aslan/workspace/test/add_kernel \
                  --grid "128,1,1" \
                  /home/aslan/workspace/test/kernel.py 
```

## link.py


```
python link.py /home/aslan/workspace/test/*.h -o add_kernel
```
