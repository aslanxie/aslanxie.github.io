# Build

As described in source code repo: [Install from source](https://github.com/openai/triton#install-from-source)
```
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build-time dependency
pip install -e .
```

Official describe: https://github.com/openai/triton/blob/main/CONTRIBUTING.md#project-structure

There 2 key parts:
- front end in python folder, most of them are Python code and harness triton kernel parsing, compiling and launching process.
- backend in lib folder, it's a C++ code for python library which is called by front end python codes

The build process is evoked by python/steup.py which will leverage fist level CMakeLists.txt to compile out python/triton/_C/libtriton.so.
