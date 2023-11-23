# Overview

Notes of Triton kernel compiling and launching process.

Triton extends Python to support Triton kernel operations which will run on GPU. So, roughly speaking, there 3 parts:
1. Add new extending built-in modules to Python to support triton kernel
2. Compile triton kernel to run on device
3. JIT automatically evoke these process   


## JIT Compile


Triton kernel function is decorated by @triton.jit, which will automatically evoke JITFunction to compile Triton kernel source and Python extending module. The module wiil be used for launching the kernel in Python interpreter . Compiler process is described by compile function (in python/triton/compiler/compiler.py).

Here is dynamically generated code for automatically compiling.  bin is a CompiledKernel object. If warmup set, it will run the kernel through c_wrapper (lanuch) entry. Or return the object.
```
# python/triton/runtime/jit.py

class JITFunction(KernelInterface[T]):
  ...
  def _make_launcher(self):
      ...
      # generated code for CUDA
      if not self._call_hook(key, signature, device, constants, num_warps, num_stages, extern_libs, configs):
        bin = compile(self, signature=signature, device=device, constants=constants, num_warps=num_warps, num_stages=num_stages, extern_libs=extern_libs, \
              configs=configs, debug=self.debug, device_type=device_type)
        if not warmup:
            bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, CompiledKernel.launch_enter_hook,\
                CompiledKernel.launch_exit_hook, bin, *args)
            self.cache[device][key] = bin
        return bin

      ...
      exec(src, scope)
      return scope[self.fn.__name__]

  ...

  def __init__(self, fn, version=None, do_not_specialize=None, debug=None, noinline=None):
      ...
      self.run = self._make_launcher()
      ...
    
```

## Kernel Compile

There are several stage to compile(optimize) kernel code, for example on CUDA platform including:ast, ttir, ttgir, llir, ptx and cubin. Here is showing roughly process from source code to ast and ttir. 

In compile function, it build compiler **stages** with IR lowering functions to generate execution binary finally.

```
def compile(fn, **kwargs):
  ...
  for ir_name, (parse, compile_kernel) in list(stages.items())[first_stage:]:
```

At last, the kernel code will be comipled to binary code for hardware paltfrom, such as cubin for CUDA platform.

### AST

It depend Python standard library [ast](https://docs.python.org/3/library/ast.html) to generate Abstract Syntax Trees.

```
# python/triton/runtime/jit.py,
# src is Trition Python code kernel.

class JITFunction(KernelInterface[T]):
  ...
  def parse(self):
      tree = ast.parse(self.src)
      ...
      return tree

```
Here is the file list for understanding AST and creating mapping TTIR.
- https://github.com/openai/triton/blob/main/python/triton/compiler/code_generator.py
- https://github.com/openai/triton/blob/main/python/triton/language/core.py
- https://github.com/openai/triton/blob/main/python/triton/language/semantic.py

Here is mapping C++ class/function to Python code interface. 
- https://github.com/openai/triton/blob/main/python/src/triton.cc

### TTIR

generator.visit is a recursive function and scan all AST node. Local visit_* function in generator will be called by Python AST library, for example visit_FunctionDef.

generator.module is ttir format.

```
# python/triton/compiler/code_generator.py

class CodeGenerator(ast.NodeVisitor):
  def __init__(self, context, prototype, gscope, attributes, constants, function_name, arch,
                 module=None, is_kernel=False, function_types: Optional[Dict] = None,
                 debug=False, noinline=False, file_name: Optional[str] = None, begin_line=0):
        ...


def ast_to_ttir(fn, signature, specialization, constants, debug, arch):
    ...
    context = ir.context()
    context.load_triton()
    ...
    generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants,
                              function_name=function_name, attributes=new_attrs,
                              is_kernel=True, debug=debug, file_name=file_name, begin_line=begin_line,
                              arch=arch)
    try:
        generator.visit(fn.parse())
    ...
    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret
```

[Here](./src/01-ttir/gen_ttir.py) is a simple code with Triton internal interface to generate TTIR. In fact, the output isn't fully based on TTIR operations, other built in dialects is emitted in the output. 

```
 module {
  tt.func public @add_kernel_0d1d2d3de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : (i32) -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %10 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<1024x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }
}

```


### TTGIR
todo

### LLIR
todo

### PTX
PTX is a low-level parallel-thread-execution virtual machine and ISA (Instruction Set Architecture). 
todo

### CUBIN

CUDA binary is a parameter of CompiledKernel. It'll be launched by launch functon in our extending Python module.

```
def compile(fn, **kwargs):
    ...
    if ir_name == "cubin":
            asm[ir_name] = next_module

    ...
    return CompiledKernel(fn, so_path, metadata, asm)
```

todo

## Python Extending 


### Generate and buid Python Module

Generate Python extending source code to luanch built triton kernel.

```
# python/triton/compiler/make_launcher.py
# generated codes for CUDA

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr arg0, CUdeviceptr arg1, CUdeviceptr arg2, int32_t arg3) {
  void *params[] = { &arg0, &arg1, &arg2, &arg3 };
  if(gridX*gridY*gridZ > 0){
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
  }
}

...

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject* _arg0;  PyObject* _arg1;  PyObject* _arg2;  int32_t _arg3; 
  if(!PyArg_ParseTuple(args, "iiiiiKKOOOOOOi", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, &_arg0, &_arg1, &_arg2, &_arg3)) {
    return NULL;
  }

   _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, ptr_info0.dev_ptr, ptr_info1.dev_ptr, ptr_info2.dev_ptr, _arg3);
```


make_stub generate launcher (generate_launcher) C code base on CUDA development interface to launcher kernel function, and follow Python extension inferface to load in Python evnironment. 

The lanucher source code is built by setuptools in python/triton/common/build.py.

```
# python/triton/compiler/make_launcher.py
# generate and build Python Extending codes 

def make_stub(name, signature, constants):
    # name of files that are cached
    so_cache_key = make_so_cache_key(version_key(), signature, constants)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path
```



### Launch Triton Kernel

CompiledKernel load the Python extending module from shared library and assign launch function to c_wrapper. It's the object return by compile and exected by Python interpreter. The built triton kernel binary is passed by asm parameter.


```
# python/triton/compiler/compiler.py
class CompiledKernel:
...
    def __init__(self, fn, so_path, metadata, asm):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        self.fn = fn
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        ...

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                if self.device_type in ["cuda", "rocm"]:
                    stream = get_cuda_stream()
                else:
                    stream = get_backend(self.device_type).get_stream(None)
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args)
        return runner
```

The built binary kernel is loaded through CUDA  Module Management interface. This is a Python extendsion, too.
```
#python/triton/runtime/backends/cuda.c
static PyObject *loadBinary(PyObject *self, PyObject *args) {
   ...
  CUDA_CHECK(cuModuleLoadData(&mod, data));
  CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
  ...
```

