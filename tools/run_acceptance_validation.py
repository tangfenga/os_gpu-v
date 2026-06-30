#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import subprocess
import sys
import time


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def env_with_ld_library_path(env):
    out = dict(env)
    for key in PROXY_ENV_KEYS:
        out.pop(key, None)
    ld = out.get("LD_LIBRARY_PATH", "")
    wsl_cuda = "/usr/lib/wsl/lib"
    if os.path.isdir(wsl_cuda) and wsl_cuda not in ld.split(":"):
        out["LD_LIBRARY_PATH"] = wsl_cuda + (":" + ld if ld else "")
    return out


def proxy_env(args):
    env = env_with_ld_library_path(os.environ)
    env.update(
        {
            "LD_PRELOAD": args.proxy_lib,
            "VGPU_SERVER": args.server,
            "VGPU_DATA_PLANE": "shm",
            "VGPU_SHM_SIZE": str(args.shm_size),
        }
    )
    if args.ring_trace:
        env["VGPU_RING_TRACE"] = "1"
    return env


def run_cmd(cmd, env=None, timeout=120, allow_fail=False):
    begin = time.monotonic()
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.monotonic() - begin
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(
            "command failed rc={} cmd={}\nstdout:\n{}\nstderr:\n{}".format(
                proc.returncode, " ".join(cmd), proc.stdout, proc.stderr
            )
        )
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def parse_line(stdout, prefix):
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return {k: v for k, v in KV_RE.findall(line)}
    raise RuntimeError("missing {} in output:\n{}".format(prefix, stdout))


def parse_worker(stdout):
    return parse_line(stdout, "worker_result")


def run_worker(args, mode, iterations, n, timeout=180, allow_fail=False):
    cmd = [args.worker, mode, str(iterations), str(n)]
    rc, stdout, stderr, elapsed = run_cmd(cmd, env=proxy_env(args), timeout=timeout, allow_fail=allow_fail)
    return rc, stdout, stderr, elapsed


def run_worker_async(args, mode, iterations, n):
    return subprocess.Popen(
        [args.worker, mode, str(iterations), str(n)],
        env=proxy_env(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_worker(proc, timeout=180):
    begin = time.monotonic()
    stdout, stderr = proc.communicate(timeout=timeout)
    elapsed = time.monotonic() - begin
    if proc.returncode != 0:
        raise RuntimeError(
            "worker failed rc={}\nstdout:\n{}\nstderr:\n{}".format(proc.returncode, stdout, stderr)
        )
    return parse_worker(stdout), stdout, stderr, elapsed


def worker_cfg(mode):
    if mode == "vector":
        return 200, 262144
    if mode == "matmul":
        return 30, 128
    raise ValueError(mode)


def worker_total_us(result):
    return float(result["total_us"])


def matrix_compare(args):
    native_env = env_with_ld_library_path(os.environ)
    cmd = [args.matrix, str(args.matrix_n), str(args.matrix_trials), str(args.matrix_warmup)]
    native_rc, native_out, native_err, _ = run_cmd(cmd, env=native_env, timeout=180, allow_fail=True)
    proxy_rc, proxy_out, proxy_err, _ = run_cmd(cmd, env=proxy_env(args), timeout=180, allow_fail=True)
    if native_rc != 0 or proxy_rc != 0:
        print("matrix_compare pass=0 native_rc={} proxy_rc={}".format(native_rc, proxy_rc))
        if native_err:
            print("matrix_native_stderr {}".format(native_err.strip().replace("\n", " | ")))
        if proxy_err:
            print("matrix_proxy_stderr {}".format(proxy_err.strip().replace("\n", " | ")))
        return False
    native = parse_line(native_out, "matrix_benchmark")
    proxy = parse_line(proxy_out, "matrix_benchmark")
    ratio = float(proxy["median_us"]) / float(native["median_us"])
    print(
        "matrix_compare n={} native_median_us={} proxy_median_us={} ratio={:.3f} pass={}".format(
            args.matrix_n, native["median_us"], proxy["median_us"], ratio, int(ratio <= 1.5)
        )
    )
    return ratio <= 1.5


def concurrency_pair(args, left, right):
    left_iter, left_n = worker_cfg(left)
    right_iter, right_n = worker_cfg(right)
    _, left_single_out, _, _ = run_worker(args, left, left_iter, left_n)
    _, right_single_out, _, _ = run_worker(args, right, right_iter, right_n)
    left_single = parse_worker(left_single_out)
    right_single = parse_worker(right_single_out)

    left_proc = run_worker_async(args, left, left_iter, left_n)
    right_proc = run_worker_async(args, right, right_iter, right_n)
    left_result, _, _, _ = wait_worker(left_proc)
    right_result, _, _, _ = wait_worker(right_proc)

    left_slowdown = worker_total_us(left_result) / worker_total_us(left_single)
    right_slowdown = worker_total_us(right_result) / worker_total_us(right_single)
    same_vptr = int(left_result.get("first_vptr") == right_result.get("first_vptr"))
    pair = "{}+{}".format(left, right)
    print(
        "concurrency_pair pair={} left_single_us={:.3f} right_single_us={:.3f} "
        "left_concurrent_us={:.3f} right_concurrent_us={:.3f} "
        "left_slowdown={:.3f} right_slowdown={:.3f} same_first_vptr={} pass=1".format(
            pair,
            worker_total_us(left_single),
            worker_total_us(right_single),
            worker_total_us(left_result),
            worker_total_us(right_result),
            left_slowdown,
            right_slowdown,
            same_vptr,
        )
    )
    return True


def error_isolation(args):
    iter_count, n = worker_cfg("vector")
    long_proc = run_worker_async(args, "vector", iter_count * 5, n)
    rc, err_out, err_stderr, _ = run_worker(args, "error", 1, 1, timeout=60)
    long_result, _, _, _ = wait_worker(long_proc, timeout=180)
    err_result = parse_worker(err_out)
    ok = rc == 0 and err_result.get("pass") == "1" and long_result.get("pass") == "1"
    print(
        "error_isolation long_mode=vector error_mode=error long_total_us={} error_pass={} pass={}".format(
            long_result.get("total_us", "na"), err_result.get("pass", "0"), int(ok)
        )
    )
    if err_stderr:
        print("error_isolation_stderr {}".format(err_stderr.strip().replace("\n", " | ")))
    return ok


def crash_isolation(args):
    rc, crash_out, crash_err, _ = run_worker(args, "crash", 1, 262144, timeout=60, allow_fail=True)
    _, ok_out, _, _ = run_worker(args, "vector", 20, 262144, timeout=120)
    ok_result = parse_worker(ok_out)
    ok = rc == 2 and ok_result.get("pass") == "1"
    print(
        "crash_isolation crash_rc={} followup_mode=vector followup_pass={} pass={}".format(
            rc, ok_result.get("pass", "0"), int(ok)
        )
    )
    if crash_err:
        print("crash_isolation_stderr {}".format(crash_err.strip().replace("\n", " | ")))
    return ok


def run_stability(args):
    deadline = time.monotonic() + args.stability_seconds
    rounds = 0
    failures = 0
    latencies = []
    while time.monotonic() < deadline:
        begin = time.monotonic()
        try:
            left_proc = run_worker_async(args, "vector", 100, 262144)
            right_proc = run_worker_async(args, "vector", 100, 262144)
            left_result, _, _, _ = wait_worker(left_proc, timeout=120)
            right_result, _, _, _ = wait_worker(right_proc, timeout=120)
            if left_result.get("pass") != "1" or right_result.get("pass") != "1":
                raise RuntimeError("dual vector worker reported failure")
            rc, err_out, _, _ = run_worker(args, "error", 1, 1, timeout=60)
            err_result = parse_worker(err_out)
            if rc != 0 or err_result.get("pass") != "1":
                raise RuntimeError("error worker reported failure")
        except Exception as exc:
            failures += 1
            print("stability_round_error round={} error={}".format(rounds, str(exc).replace("\n", " | ")))
        latencies.append((time.monotonic() - begin) * 1000.0)
        rounds += 1
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    print(
        "stability_sustained seconds={} rounds={} failures={} mean_round_ms={:.3f} pass={}".format(
            args.stability_seconds, rounds, failures, mean_ms, int(failures == 0 and rounds > 0)
        )
    )
    return failures == 0 and rounds > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:50052")
    parser.add_argument("--proxy-lib", default="./build/libcudart_proxy.so")
    parser.add_argument("--worker", default="/tmp/vgpu_concurrent_worker")
    parser.add_argument("--matrix", default="/tmp/vgpu_matrix_mul_benchmark")
    parser.add_argument("--matrix-n", type=int, default=256)
    parser.add_argument("--matrix-trials", type=int, default=10)
    parser.add_argument("--matrix-warmup", type=int, default=20)
    parser.add_argument("--shm-size", type=int, default=268435456)
    parser.add_argument("--stability-seconds", type=int, default=600)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--ring-trace", action="store_true")
    args = parser.parse_args()

    required = [args.proxy_lib, args.worker, args.matrix]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print("missing required files: {}".format(", ".join(missing)), file=sys.stderr)
        return 2

    checks = []
    checks.append(matrix_compare(args))
    for left, right in (("vector", "vector"), ("vector", "matmul"), ("matmul", "matmul")):
        checks.append(concurrency_pair(args, left, right))
    checks.append(error_isolation(args))
    checks.append(crash_isolation(args))
    if not args.skip_stability:
        checks.append(run_stability(args))
    print("acceptance_validation pass={}".format(int(all(checks))))
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
