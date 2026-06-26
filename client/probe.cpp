#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "vgpu.grpc.pb.h"

namespace {

std::string ServerAddress(int argc, char **argv) {
    if (argc > 1) {
        return argv[1];
    }
    const char *env = std::getenv("VGPU_SERVER");
    if (env && env[0] != '\0') {
        return env;
    }
    return "127.0.0.1:50051";
}

}  // namespace

int main(int argc, char **argv) {
    const std::string server_addr = ServerAddress(argc, argv);
    auto channel = grpc::CreateChannel(server_addr, grpc::InsecureChannelCredentials());
    auto stub = vgpu::VgpuRuntime::NewStub(channel);

    vgpu::CreateSessionRequest create_request;
    create_request.set_client_pid(static_cast<uint32_t>(getpid()));
    create_request.set_client_name("vgpu_client_probe");
    create_request.set_requested_memory_limit(0);

    vgpu::CreateSessionReply create_reply;
    grpc::ClientContext create_context;
    grpc::Status status = stub->CreateSession(&create_context, create_request, &create_reply);
    if (!status.ok()) {
        std::cerr << "CreateSession RPC failed: " << status.error_message() << std::endl;
        return 1;
    }
    if (create_reply.cuda_error() != 0) {
        std::cerr << "CreateSession CUDA error " << create_reply.cuda_error()
                  << ": " << create_reply.message() << std::endl;
        return 1;
    }

    const uint64_t session_id = create_reply.session_id();
    std::cout << "created session: " << session_id << std::endl;

    vgpu::GetDeviceCountRequest count_request;
    count_request.set_session_id(session_id);
    vgpu::GetDeviceCountReply count_reply;
    grpc::ClientContext count_context;
    status = stub->GetDeviceCount(&count_context, count_request, &count_reply);
    if (!status.ok() || count_reply.cuda_error() != 0) {
        std::cerr << "GetDeviceCount failed" << std::endl;
        return 1;
    }
    std::cout << "device count: " << count_reply.count() << std::endl;

    vgpu::GetDevicePropertiesRequest props_request;
    props_request.set_session_id(session_id);
    props_request.set_device(0);
    vgpu::GetDevicePropertiesReply props_reply;
    grpc::ClientContext props_context;
    status = stub->GetDeviceProperties(&props_context, props_request, &props_reply);
    if (!status.ok() || props_reply.cuda_error() != 0) {
        std::cerr << "GetDeviceProperties failed" << std::endl;
        return 1;
    }
    std::cout << "device name: " << props_reply.name() << std::endl;

    vgpu::DestroySessionRequest destroy_request;
    destroy_request.set_session_id(session_id);
    vgpu::StatusReply destroy_reply;
    grpc::ClientContext destroy_context;
    status = stub->DestroySession(&destroy_context, destroy_request, &destroy_reply);
    if (!status.ok() || destroy_reply.cuda_error() != 0) {
        std::cerr << "DestroySession failed" << std::endl;
        return 1;
    }
    std::cout << "destroyed session: " << session_id << std::endl;
    return 0;
}

