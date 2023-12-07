# Overview

In simply and directly words, Triton directly build Pyhton code function to GPU kernel and execute on GPU. 

Triton depends on LLVM MLIR infrastructrue to build a DSL complier, and intergrates the execution process with Python interpreter. 


## Kernel Launch process

New *python/triton/runtime/* define CudaDriver backend which is a self-contained Python extension module supporting to get device attributes and load GPU kernel.

Another extension module is in *python/triton/compiler/make_launcher.py* which includes kernel launch function. It's a dynamically generated code module depends on kernel.

*class CompiledKernel* in *python/triton/compiler/compiler.py*, read kernel binary, denped on driver backend moduel loading kernel binary and creating kernel function, inintial launch moudle and execute kernel function. 

```
```



