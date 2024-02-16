# MLIR, SPIR-V and L0

convert spir-v to binary and load by leveo zero for running.

- https://mlir.llvm.org/docs/Dialects/SPIR-V/
- https://github.com/intel/intel-graphics-compiler/tree/master
- https://github.com/KhronosGroup/SPIRV-Tools
- https://github.com/KhronosGroup/SPIRV-LLVM-Translator
- https://github.com/intel/intel-graphics-compiler/tree/master
- https://github.com/intel-sandbox/gpu_memory_sharing/tree/main/l0_mem_export


```
 ./spirv-as -o /home/aslan/workspace/L0/add_kernel.spv  /home/aslan/workspace/L0/add_kernel.spirv  --target-env opencl2.2
```
