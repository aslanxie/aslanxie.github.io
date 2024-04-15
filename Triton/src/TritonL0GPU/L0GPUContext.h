//Reference:
//https://github.com/oneapi-src/level-zero-tests/blob/master/perf_tests/ze_peak/include/ze_peak.h

#ifndef __L0CONTEXT_H__
#define __L0CONTEXT_H__

#include <chrono>
#include <cstdint>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>

/* ze includes */
#include <level_zero/ze_api.h>

#define CHECK_L0_RESULT(func, result)           \
if (result != ZE_RESULT_SUCCESS) {              \
    std::cout << __FILE__ << ": "               \
              << __LINE__ <<": ERROR: "         \
              << func <<  " return "            \
                << result << std::endl;         \
    exit(0);                                    \
}


class L0GPUContext{
public:
    L0GPUContext();
    ~L0GPUContext();

    void init();
    void clean();

    void create_module(std::vector<uint8_t> binary_file, int format);
    void create_kernel(const char *name, void *arg1);
    void run_kernel();
    void* export_buffer();

private:
    ze_driver_handle_t m_driver;
    ze_context_handle_t m_context;
    ze_device_handle_t m_device;
    
    ze_command_queue_handle_t m_command_queue;
    ze_command_list_handle_t m_command_list;

    ze_module_handle_t m_module;
    ze_kernel_handle_t m_kernel;

    uint32_t m_driver_index;
    uint32_t m_device_index;  

    void*  m_buff;
    void* m_mapped_memory;
};

#endif //__L0CONTEXT_H__