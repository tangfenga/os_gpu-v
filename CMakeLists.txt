cmake_minimum_required(VERSION 3.20)

project(os_hyj_vgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/vgpu.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(GENERATED_PROTO_CC ${GENERATED_DIR}/vgpu.pb.cc)
set(GENERATED_PROTO_H ${GENERATED_DIR}/vgpu.pb.h)
set(GENERATED_GRPC_CC ${GENERATED_DIR}/vgpu.grpc.pb.cc)
set(GENERATED_GRPC_H ${GENERATED_DIR}/vgpu.grpc.pb.h)

add_custom_command(
    OUTPUT
        ${GENERATED_PROTO_CC}
        ${GENERATED_PROTO_H}
        ${GENERATED_GRPC_CC}
        ${GENERATED_GRPC_H}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
        --cpp_out=${GENERATED_DIR}
        --grpc_out=${GENERATED_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC/protobuf sources from proto/vgpu.proto"
    VERBATIM
)

add_library(vgpu_proto
    ${GENERATED_PROTO_CC}
    ${GENERATED_GRPC_CC}
)
set_target_properties(vgpu_proto PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(vgpu_proto
    PUBLIC
        ${GENERATED_DIR}
)
target_link_libraries(vgpu_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

add_executable(vgpu_server
    server/main.cpp
)
target_link_libraries(vgpu_server
    PRIVATE
        vgpu_proto
        CUDA::cuda_driver
)

add_executable(vgpu_client_probe
    client/probe.cpp
)
target_link_libraries(vgpu_client_probe
    PRIVATE
        vgpu_proto
)

add_library(cudart_proxy SHARED
    client/libcudart_proxy.cpp
)
set_target_properties(cudart_proxy PROPERTIES
    OUTPUT_NAME cudart_proxy
)
target_link_libraries(cudart_proxy
    PRIVATE
        vgpu_proto
)
