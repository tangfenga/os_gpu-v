# os_gpu-v

Selected topic: `proj43` - lightweight GPU virtualization.

`os_gpu-v` is a lightweight CUDA Runtime API virtualization prototype. It intercepts CUDA Runtime calls from a client process, forwards the required GPU process state and data to a server, and executes the work on a host NVIDIA GPU through the CUDA Driver API.

The design focuses on process-level CUDA resource virtualization: only the CUDA Runtime state and data needed by a running GPU process are represented and transferred.

The contest task asks for a lightweight GPU virtualization design that takes a running GPU process as the migration/virtualization unit, captures the operator and CUDA runtime information needed by the client process, forwards the work to a host-side NVIDIA GPU, and reduces the data-transfer overhead with concurrency and synchronization optimizations.

## Features

- CUDA Runtime API interception with `LD_PRELOAD`
- Per-process `session` resource isolation
- Session-local virtual device pointers
- CUDA Driver API execution on the server
- H2D/D2H data transfer through POSIX shared memory
- Kernel launch, D2D copy, and sync fast path through per-session SPSC rings
- Stream/event virtualization
- Client-side keepalive to prevent premature session reclamation
- Ring operation timeout with automatic fallback to gRPC path
- O(log n) virtual pointer lookup via ordered map (vs O(n) linear scan)
- Server-side shared memory cleanup on session teardown
- Multi-process concurrency tests
- Performance and stability validation scripts

## Repository Layout

```text
client/     CUDA Runtime proxy library
server/     vGPU server, session manager, ring worker
proto/      gRPC/protobuf service definition
shared/     shared-memory ring structures
tools/
  smoke/      basic smoke tests (vector add, memcpy, stream, event, ...)
  stress/     stress and negative tests
  concurrency/  multi-process concurrency and cross-session tests
  benchmark/    performance benchmarks
  scripts/      Python acceptance validation scripts
docs/       project report
```

## Submission Scope

The submitted project entry points are:

- `vgpu_server`
- `libcudart_proxy.so`
- `proto/vgpu.proto`
- `client/`, `server/`, `shared/`
- acceptance tests and benchmarks under `tools/`
- project report under `docs/`

The repository excludes local probes, one-off debugging programs, generated build output, and learning notes.

## Dependencies

- Linux
- CMake 3.20+
- C++17 compiler
- protobuf and gRPC C++ development packages
- NVIDIA CUDA Toolkit
- NVIDIA driver available to the server process
- Python 3 for acceptance scripts

On Ubuntu-like systems, install protobuf/gRPC through your package manager or a vcpkg/conda/toolchain setup that exposes CMake package config files.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To also build CUDA test binaries when `nvcc` is available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVGPU_BUILD_CUDA_TESTS=ON
cmake --build build -j
```

Main artifacts:

```text
build/vgpu_server
build/libcudart_proxy.so
```

## Quick Start

Start the server:

```bash
./build/vgpu_server 127.0.0.1:50052
```

Run a CUDA program through the proxy. This command assumes the project was configured with `-DVGPU_BUILD_CUDA_TESTS=ON`:

```bash
VGPU_DATA_PLANE=shm \
NO_PROXY=127.0.0.1,localhost \
LD_PRELOAD="$PWD/build/libcudart_proxy.so" \
VGPU_SERVER=127.0.0.1:50052 \
./build/vgpu_vector_add_smoke
```

If CUDA test binaries were not built by CMake, compile one manually with shared CUDA Runtime:

```bash
nvcc -std=c++17 -cudart shared tools/smoke/vector_add_smoke.cu -o /tmp/vgpu_vector_add_smoke
```

Then run:

```bash
VGPU_DATA_PLANE=shm \
NO_PROXY=127.0.0.1,localhost \
LD_PRELOAD="$PWD/build/libcudart_proxy.so" \
VGPU_SERVER=127.0.0.1:50052 \
/tmp/vgpu_vector_add_smoke
```

## Acceptance Validation

Build the acceptance binaries (only needed when building without CMake CUDA tests):

```bash
nvcc -std=c++17 -cudart shared tools/concurrency/concurrent_worker.cu -o /tmp/vgpu_concurrent_worker
nvcc -std=c++17 -cudart shared tools/benchmark/matrix_mul_benchmark.cu -o /tmp/vgpu_matrix_mul_benchmark
nvcc -std=c++17 -cudart shared tools/concurrency/cross_session_sync_test.cu -o /tmp/vgpu_cross_session_sync_test
```

Run the acceptance suite (using CMake-built binaries):

```bash
./tools/scripts/run_acceptance_validation.py \
  --server 127.0.0.1:50052 \
  --proxy-lib "$PWD/build/libcudart_proxy.so" \
  --skip-stability
```

Run the cross-session synchronization test:

```bash
./tools/scripts/run_cross_session_sync_test.py \
  --server 127.0.0.1:50052 \
  --proxy-lib "$PWD/build/libcudart_proxy.so"
```

For the full 10-minute stability run:

```bash
./tools/scripts/run_acceptance_validation.py \
  --server 127.0.0.1:50052 \
  --proxy-lib "$PWD/build/libcudart_proxy.so" \
  --stability-seconds 600
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VGPU_SERVER` | `127.0.0.1:50051` | Server address |
| `VGPU_DATA_PLANE` | (empty, uses gRPC) | Set to `shm` to enable shared memory data path |
| `VGPU_SHM_SIZE` | 67108864 (64 MB) | Shared memory arena size |
| `VGPU_SHM_THRESHOLD` | 65536 (64 KB) | Minimum transfer size to use SHM path |
| `VGPU_MEMORY_LIMIT` | 0 (unlimited) | Per-session GPU memory limit |
| `VGPU_SESSION_TIMEOUT_MS` | 30000 | Session idle timeout before server reclamation (ms) |
| `VGPU_RING_TIMEOUT_US` | 5000000 | Ring operation timeout (us); on expiry ring is disabled and falls back to gRPC |
| `VGPU_PERF_DETAIL` | 0 | Set to `1` for detailed performance logging |
| `VGPU_INIT_TRACE` | 0 | Set to `1` for initialization timing trace |

## Report

- [Project Report](docs/project_report.md)
