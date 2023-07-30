***@triton.jit*** decorator is parsed by [Triton JIT runtime](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py)
and [compiler](https://github.com/openai/triton/blob/main/python/triton/compiler/compiler.py).



[class JITFunction](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L145) [parse kernle source to AST](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L462) 
and [dynamically generate code](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py#L306) to compile kernel.

JITFunction.src contain ***@triton.jit*** decorator code.

JITFunction._make_launcher function obtain the dynamically generate code function object through *exec* and assign to *JITFunction.run*. The dynamically generate code includes detecting and configuring environment to  compile *JITFunction.src*.

*exec(src, scope)* is called to compile and optimize kernel codes, the ***src*** only includes the function def which looks like return the compiled binary object. The ***compile*** function is overloaded by local codes, not default Python Built-in Function. 

JITFunction.parse will parse JITFunction.src to AST.



In [compiler.py](https://github.com/openai/triton/blob/main/python/triton/compiler/compiler.py):
1. Function compile describe complier stages and generate a CompiledKernel object.
2. class CompiledKernel provide interface to launch kernel

