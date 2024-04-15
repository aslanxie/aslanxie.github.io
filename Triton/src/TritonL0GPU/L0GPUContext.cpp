#include <sys/mman.h> //mmap

#include "L0GPUContext.h"

L0GPUContext::L0GPUContext(){
    m_driver_index = 0;
    m_device_index = 0; 
}

L0GPUContext::~L0GPUContext(){
    
}

//format 1: SPV, 2 Native binary
void L0GPUContext::create_module(std::vector<uint8_t> binary_file, int format){
    std::cout << "create_module\n";

    ze_result_t result = ZE_RESULT_SUCCESS;
    ze_module_desc_t module_description = {};
    ze_module_build_log_handle_t buildLog;

    module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    module_description.pNext = nullptr;
    if (format == 1)
        module_description.format = ZE_MODULE_FORMAT_IL_SPIRV;
    else  
        module_description.format = ZE_MODULE_FORMAT_NATIVE;
    module_description.inputSize = static_cast<uint32_t>(binary_file.size());
    module_description.pInputModule =
        reinterpret_cast<const uint8_t *>(binary_file.data());
    module_description.pBuildFlags = "";
    std::cout << "zeModuleCreate \n";
    result = zeModuleCreate(m_context, m_device, &module_description, &m_module, &buildLog);
    std::cout << "zeModuleCreate " << result << "\n"<< std::flush;
    if(result !=ZE_RESULT_SUCCESS){
        size_t szLog = 0;
        zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

        char* stringLog = (char*)malloc(szLog);
        zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
        std::cout << "Build log: " << stringLog << std::endl << std::flush;
        return;
    }    

    result = zeModuleBuildLogDestroy(buildLog);
    CHECK_L0_RESULT("zeModuleBuildLogDestroy", result);    
    
    std::cout << "Module created\n" << std::flush;

    size_t szBinary = 0;
    result = zeModuleGetNativeBinary(m_module, &szBinary, nullptr);
    CHECK_L0_RESULT("zeModuleGetNativeBinary", result);  

    if(format == 1){
        std::vector<uint8_t> pBinary;
        pBinary.resize(szBinary);
        result = zeModuleGetNativeBinary(m_module, &szBinary, reinterpret_cast<uint8_t *>(pBinary.data()));
        CHECK_L0_RESULT("zeModuleGetNativeBinary", result);
        std::cout << "Native binary size: " << szBinary << std::endl;

        std::ofstream stream("add_kernel.bin", std::ios::binary);
        stream.write(reinterpret_cast<const char *>(pBinary.data()), szBinary);
        std::cout << "Write binary to add_kernel.bin" << std::endl; 
    }     
}

void L0GPUContext::init(){
    ze_result_t result = ZE_RESULT_SUCCESS;

    //GPU device only
    result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    CHECK_L0_RESULT("zeInit", result);

    //drvier
    uint32_t driver_count = 0;
    result = zeDriverGet(&driver_count, nullptr);
    CHECK_L0_RESULT("zeDriverGet", result);
    if (driver_count == 0) {
        std::cout <<"No level zero driver found!" << std::endl;
        exit(1);
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    result = zeDriverGet(&driver_count, drivers.data());
    CHECK_L0_RESULT("zeDriverGet", result);

    //default driver index 0
    std::cout<<"Found " << driver_count << " drivers in system, choose "
        <<" to work on index: " << m_driver_index << std::endl;
    m_driver = drivers[m_driver_index];

    ze_api_version_t version = ZE_API_VERSION_FORCE_UINT32;
    result = zeDriverGetApiVersion(m_driver, &version);
    CHECK_L0_RESULT("zeDriverGetApiVersion", result);
    std::cout << "Driver API Version " << ZE_MAJOR_VERSION(version)
        << "." << ZE_MINOR_VERSION(version) << std::endl;
    
    //context
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    result = zeContextCreate(m_driver, &context_desc, &m_context);
    CHECK_L0_RESULT("zeContextCreate", result);

    uint32_t device_count = 0;
    result = zeDeviceGet(m_driver, &device_count, nullptr);
    CHECK_L0_RESULT("zeDeviceGet", result);
    if (device_count == 0) {
        std::cout <<"No level zero device found!" << std::endl;
        exit(1);
    }

    std::vector<ze_device_handle_t> devices(device_count);
    result = zeDeviceGet(m_driver, &device_count, devices.data());
    CHECK_L0_RESULT("zeDeviceGet", result);

    //default device index 0
    std::cout<<"Found " << device_count << " GPU device in system, choose "
        <<" to work on index: " << m_device_index << std::endl;
    m_device = devices[m_device_index];

    ze_device_properties_t device_properties{
                    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, };
    result = zeDeviceGetProperties(m_device, &device_properties);
    CHECK_L0_RESULT("zeDeviceGetProperties", result);
    std::cout << "Device: " << device_properties.name << std::endl;

    uint32_t queue_group_count = 0;
    result = zeDeviceGetCommandQueueGroupProperties(m_device, &queue_group_count, nullptr);
    CHECK_L0_RESULT("zeDeviceGetCommandQueueGroupProperties", result);
    if (queue_group_count == 0) {
        std::cout << " No queue groups found\n" << std::endl;
        exit(0);
    }

    std::vector<ze_command_queue_group_properties_t> cmdqueueGroupProperties(queue_group_count);
    result = zeDeviceGetCommandQueueGroupProperties(m_device, &queue_group_count, cmdqueueGroupProperties.data());
    CHECK_L0_RESULT("zeDeviceGetCommandQueueGroupProperties", result);

    // Find a command queue type that support compute
    uint32_t computeQueueGroupOrdinal = queue_group_count;
    for( uint32_t i = 0; i < queue_group_count; ++i ) {
        if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
            computeQueueGroupOrdinal = i;
            break;
        }
    }

    //Create command queue
    ze_command_queue_desc_t command_queue_description{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,//stype
        nullptr,//pNext
        computeQueueGroupOrdinal, //ordinal
        computeQueueGroupOrdinal, // index
        ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY, // flags
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, //mode
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL //priority
    };

    ze_command_list_desc_t command_list_description{
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,//stype
        nullptr, //pNext
        computeQueueGroupOrdinal, //commandQueueGroupOrdinal
        ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY // flags
    };

    result = zeCommandListCreate(m_context, m_device, &command_list_description, &m_command_list);
    CHECK_L0_RESULT("zeCommandListCreate", result);

    result = zeCommandQueueCreate(m_context, m_device, &command_queue_description, &m_command_queue);
    CHECK_L0_RESULT("zeCommandListCreate", result); 
}

void L0GPUContext::clean(){
    ze_result_t result = ZE_RESULT_SUCCESS;

    result = zeCommandListDestroy(m_command_list);
    CHECK_L0_RESULT("zeCommandListDestroy", result); 

    result = zeCommandQueueDestroy(m_command_queue);
    CHECK_L0_RESULT("zeCommandQueueDestroy", result); 

    result = zeContextDestroy(m_context);
    CHECK_L0_RESULT("zeContextDestroy", result); 
}


void L0GPUContext::create_kernel(const char *name, void *arg0){
    ze_result_t result = ZE_RESULT_SUCCESS;
    ze_kernel_desc_t function_description = {};
    function_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    
    function_description.pNext = nullptr;
    function_description.flags = 0;
    function_description.pKernelName = name;

    result = zeKernelCreate(m_module, &function_description, &m_kernel);
    CHECK_L0_RESULT("zeKernelCreate", result);
    std::cout << "Kernel function created\n";

    int elements = 1024;
    int bs = 16;
    void *arg1 = arg0 + 1024 * 4;
    void *arg2 = arg1 + 1024 * 4;

    result = zeKernelSetArgumentValue(m_kernel, 0, 1024 * 4, &arg0);
    CHECK_L0_RESULT("zeKernelSetArgumentValue", result);
    result = zeKernelSetArgumentValue(m_kernel, 1, 1024 * 4, &arg1);
    CHECK_L0_RESULT("zeKernelSetArgumentValue", result);
    result = zeKernelSetArgumentValue(m_kernel, 2, 1024 * 4, &arg2);
    CHECK_L0_RESULT("zeKernelSetArgumentValue", result);

    //
    //result = zeKernelSetArgumentValue(m_kernel, 3, sizeof(int), &elements);
    //CHECK_L0_RESULT("zeKernelSetArgumentValue", result);
    //result = zeKernelSetArgumentValue(m_kernel, 4, sizeof(int), &bs);
    //CHECK_L0_RESULT("zeKernelSetArgumentValue", result);
}

void L0GPUContext::run_kernel(){
    ze_result_t result = ZE_RESULT_SUCCESS;
    uint32_t groupSizeX = 32u;
    uint32_t groupSizeY = 1u;
    uint32_t groupSizeZ = 1u;

    result = zeKernelSetGroupSize(m_kernel, groupSizeX, groupSizeY, groupSizeZ);
    CHECK_L0_RESULT("zeKernelSetGroupSize", result);

     // Kernel thread-dispatch
    ze_group_count_t dispatch;
    dispatch.groupCountX = 4;
    dispatch.groupCountY = 1;
    dispatch.groupCountZ = 1;

    result = zeCommandListAppendLaunchKernel(m_command_list, m_kernel, &dispatch, nullptr, 0, nullptr);
    CHECK_L0_RESULT("zeCommandListAppendLaunchKernel", result);

    result = zeCommandListClose(m_command_list);
    CHECK_L0_RESULT("zeCommandListClose", result);    

    result = zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr);
    CHECK_L0_RESULT("zeCommandQueueExecuteCommandLists", result);

    result = zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max());
    CHECK_L0_RESULT("zeCommandQueueSynchronize", result);

    result = zeCommandListReset(m_command_list);
    CHECK_L0_RESULT("zeCommandListReset", result); 

    std::cout << "Check result:" << std::endl;
    float *buf = (float *)m_mapped_memory + 1024 * 2;
    for(int i = 0; i < 64; i ++)
        std::cout << buf[i] << "\t";
    std::cout << "\n";

}

void* L0GPUContext::export_buffer(){
    ze_result_t result = ZE_RESULT_SUCCESS;

     // allocate memory for results
    ze_device_mem_alloc_desc_t alloc_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        nullptr,
        0, // flags
        0  // ordinal
    };

    // Set up the request for an exportable allocation
    ze_external_memory_export_desc_t export_desc = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF
    };

    int alignment = 32;
    int size = 1024 * 16;
    // Link the request into the allocation descriptor and allocate
    alloc_desc.pNext = &export_desc;
    result = zeMemAllocDevice(m_context, &alloc_desc, size, alignment, m_device, &m_buff);
    CHECK_L0_RESULT("zeMemAllocDevice", result);

     // Set up the request to export the external memory handle
    ze_external_memory_export_fd_t export_fd = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, //ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        0 // [out] fd
    };
    
    // Link the export request into the query
    ze_memory_allocation_properties_t alloc_props = {};
    alloc_props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    alloc_props.pNext = &export_fd;
    result = zeMemGetAllocProperties(m_context, m_buff, &alloc_props, nullptr);
    CHECK_L0_RESULT("zeMemAllocDevice", result);

    m_mapped_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, export_fd.fd, 0);
    if (m_mapped_memory == MAP_FAILED) {
        std::cout <<"failed to mmap" << std::endl;
        exit(0);
    }

    std::cout << "Set buffer value" << std::endl;
    float* buf = (float *)m_mapped_memory;
    for(int i = 0; i < size/8; i ++)
        buf[i] = 1.28 + 32.0/(i+1.0);
    
    return m_buff;
    
}