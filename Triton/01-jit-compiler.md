***@triton.jit*** decorator is parsed by [Triton JIT runtime](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py)
and [compiler](https://github.com/openai/triton/blob/main/python/triton/compiler/compiler.py).


[class JITFunction](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L145) [parse kernle source to AST](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L462) 
and [dynamically generate code](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L306) to compile and launch kernel.

In [compiler.py](https://github.com/openai/triton/blob/main/python/triton/compiler/compiler.py):
1. Function compile describe complier stages and generate a CompiledKernel object.
2. class CompiledKernel provide interface to launch kernel

