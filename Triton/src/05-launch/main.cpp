// reference: 
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddDrv
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE
// Build
// /usr/local/cuda/bin/nvcc  -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -lcudadevrt -lcuda  -o launch main.cpp 

// Includes
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include <math.h>

// includes, CUDA
#include <builtin_types.h>

using namespace std;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
float *h_A;
float *h_B;
float *h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

// Functions
int CleanupNoFailure();
void RandomInit(float *, int);

// Host code
int main(int argc, char **argv) {
  printf("Vector Addition (Driver API)\n");

  int deviceCount = 0;
  uint32_t N = 1024;
  int devID = 0;
  size_t size = N * sizeof(float);

  // Initialize
  checkCudaErrors(cuInit(0));

  checkCudaErrors(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    std::cout << "cudaDeviceInit error: no devices supporting CUDA\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "There are " << deviceCount << "GPU in system and choose: " << devID << std::endl; 
  checkCudaErrors(cuDeviceGet(&cuDevice, devID));
  char name[100];
  checkCudaErrors(cuDeviceGetName(name, 100, cuDevice));
  std::cout << "Device name: " << std::string(name) << std::endl;

  // Create context
  checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

  // first search for the module path before we load the results
  std::ostringstream cubin;
  std::string filename = "add_kernel.cubin";
  std::ifstream fileIn(filename, std::ios::binary); 

  cubin << fileIn.rdbuf();
  fileIn.close();

  // Create module from binary file (FATBIN)
  checkCudaErrors(cuModuleLoadData(&cuModule, cubin.str().c_str()));

  // Get function handle from module
  checkCudaErrors(
      cuModuleGetFunction(&vecAdd_kernel, cuModule, "add_kernel_0d1d2d3de"));

  // Allocate input vectors h_A and h_B in host memory
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initialize input vectors
  RandomInit(h_A, N);
  RandomInit(h_B, N);

  // Allocate vectors in device memory
  checkCudaErrors(cuMemAlloc(&d_A, size));

  checkCudaErrors(cuMemAlloc(&d_B, size));

  checkCudaErrors(cuMemAlloc(&d_C, size));

  // Copy vectors from host memory to device memory
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));

  checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

  // Grid/Block configuration
  uint32_t bs = 1024;
  int threadsPerBlock = 128;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  void *args[] = {&d_A, &d_B, &d_C, &N};

  // Launch the CUDA kernel
  checkCudaErrors(cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1,
                            threadsPerBlock, 1, 1, 0, NULL, args, NULL));

#ifdef _DEBUG
  checkCudaErrors(cuCtxSynchronize());
#endif

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

  // Verify result
  int i;

  for (i = 0; i < N; ++i) {
    float sum = h_A[i] + h_B[i];

    if (fabs(h_C[i] - sum) > 1e-7f) {
      break;
    }
  }

  CleanupNoFailure();
  printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

  exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure() {
  // Free device memory
  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));

  // Free host memory
  if (h_A) {
    free(h_A);
  }

  if (h_B) {
    free(h_B);
  }

  if (h_C) {
    free(h_C);
  }

  checkCudaErrors(cuCtxDestroy(cuContext));

  return EXIT_SUCCESS;
}

// Allocates an array with random float entries.
void RandomInit(float *data, int n) {
  for (int i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}