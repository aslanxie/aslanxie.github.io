cmake_minimum_required(VERSION 3.10)

# set the project name
set(TARGET_NAME l0_run)
project(${TARGET_NAME})


set (CMAKE_CXX_FLAGS "-std=c++11 -fpermissive ${CMAKE_CXX_FLAGS}")

include_directories (${TARGET_NAME} /usr/local/include)


link_directories(/usr/local/lib/)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/opt/intel/oneapi/compiler/latest/linux/lib/)

add_executable(${TARGET_NAME}
    main.cpp
    L0GPUContext.cpp
)

target_link_libraries( ${TARGET_NAME} ze_loader
)

add_custom_command(TARGET ${TARGET_NAME}
          PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_SOURCE_DIR}/add_kernel.spv  ${PROJECT_BINARY_DIR}/
)