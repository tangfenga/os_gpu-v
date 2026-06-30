cmake_minimum_required(VERSION 3.20)

project(os_gpu_v LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VGPU_BUILD_CLIENT_PROBE "Build the small gRPC client probe utility" ON)
option(VGPU_BUILD_CUDA_TESTS "Build CUDA smoke tests and benchmarks with nvcc" OFF)

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

if(VGPU_BUILD_CLIENT_PROBE)
    add_executable(vgpu_client_probe
        client/probe.cpp
    )
    target_link_libraries(vgpu_client_probe
        PRIVATE
            vgpu_proto
    )
endif()

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

if(VGPU_BUILD_CUDA_TESTS)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "VGPU_BUILD_CUDA_TESTS=ON requires a CUDA compiler such as nvcc")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    function(add_vgpu_cuda_test target source)
        add_executable(${target} ${source})
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
        )
        target_link_libraries(${target}
            PRIVATE
                CUDA::cudart
        )
    endfunction()

    add_vgpu_cuda_test(vgpu_vector_add_smoke tools/vector_add_smoke.cu)
    add_vgpu_cuda_test(vgpu_runtime_query_smoke tools/runtime_query_smoke.cu)
    add_vgpu_cuda_test(vgpu_memcpy_baseline tools/memcpy_baseline.cu)
    add_vgpu_cuda_test(vgpu_matrix_mul_benchmark tools/matrix_mul_benchmark.cu)
    add_vgpu_cuda_test(vgpu_concurrent_worker tools/concurrent_worker.cu)
    add_vgpu_cuda_test(vgpu_cross_session_sync_test tools/cross_session_sync_test.cu)
    add_vgpu_cuda_test(vgpu_fire_forget_stress tools/fire_forget_stress.cu)
    add_vgpu_cuda_test(vgpu_stability_negative_smoke tools/stability_negative_smoke.cu)
    add_vgpu_cuda_test(vgpu_stream_async_smoke tools/stream_async_smoke.cu)
    add_vgpu_cuda_test(vgpu_event_timing_smoke tools/event_timing_smoke.cu)
    add_vgpu_cuda_test(vgpu_host_alloc_smoke tools/host_alloc_smoke.cu)
endif()

install(TARGETS vgpu_server cudart_proxy
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(VGPU_BUILD_CLIENT_PROBE)
    install(TARGETS vgpu_client_probe
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
