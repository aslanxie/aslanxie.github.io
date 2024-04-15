#include <unistd.h>

#include "L0GPUContext.h"

std::vector<uint8_t> load_binary_file(const std::string &file_path){
    std::cout << "File path: " << file_path << std::endl;
    std::ifstream stream(file_path, std::ios::in | std::ios::binary);

    std::vector<uint8_t> binary_file;
    if (!stream.good()) {
        std::cerr << "Failed to load binary file: " << file_path << std::endl;
        exit(-1);
    }

    size_t length = 0;
    stream.seekg(0, stream.end);
    length = static_cast<size_t>(stream.tellg());
    stream.seekg(0, stream.beg);
  
    std::cout << "Binary file length: " << length << std::endl;

    binary_file.resize(length);
    stream.read(reinterpret_cast<char *>(binary_file.data()), length);

    std::cout << "Binary file loaded" << std::endl;

    return binary_file;
}

// l0_run kernel type
// l0_run add_kernel.spv 1
// l0_run add_kernel.bin 2
int main(int argc, char **argv){
    
    L0GPUContext l0;    

    int format = 1;
    if (argc > 2) format = atoi(argv[2]);

    std::string fmt = format == 1 ? "SPV" : " Native Binary";
    std::string name = argv[1];
    std::cout << "Input format is " <<  fmt << ", name: " << name << std::endl;

    std::vector<uint8_t> binary_file = load_binary_file(name);    

    l0.init();    
    
    l0.create_module(binary_file, format);

    void* buff = l0.export_buffer();

    l0.create_kernel("add_kernel", buff);

    l0.run_kernel();


    l0.clean();
   
    return 0;
}
