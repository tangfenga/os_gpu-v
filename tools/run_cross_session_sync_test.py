#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
import time


def proxy_env(args):
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    if os.path.isdir("/usr/lib/wsl/lib") and "/usr/lib/wsl/lib" not in ld.split(":"):
        env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (":" + ld if ld else "")
    env.update(
        {
            "LD_PRELOAD": args.proxy_lib,
            "VGPU_SERVER": args.server,
            "VGPU_DATA_PLANE": "shm",
            "VGPU_SHM_SIZE": str(args.shm_size),
        }
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:50052")
    parser.add_argument("--proxy-lib", default="./build/libcudart_proxy.so")
    parser.add_argument("--binary", default="/tmp/vgpu_cross_session_sync_test")
    parser.add_argument("--cycles", default="1500000000")
    parser.add_argument("--hold-ms", default="1500")
    parser.add_argument("--threshold-us", default="200000")
    parser.add_argument("--shm-size", type=int, default=268435456)
    args = parser.parse_args()

    signal_path = os.path.join(tempfile.gettempdir(), "vgpu_cross_session_sync_{}.ready".format(os.getpid()))
    try:
        os.unlink(signal_path)
    except FileNotFoundError:
        pass

    env = proxy_env(args)
    holder = subprocess.Popen(
        [args.binary, "holder", signal_path, args.cycles, args.hold_ms],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.1)
    probe = subprocess.Popen(
        [args.binary, "sync_probe", signal_path, args.threshold_us],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    probe_out, probe_err = probe.communicate(timeout=20)
    holder_out, holder_err = holder.communicate(timeout=20)

    ok = holder.returncode == 0 and probe.returncode == 0
    print(holder_out, end="")
    print(probe_out, end="")
    if holder_err:
        print("holder_stderr {}".format(holder_err.strip().replace("\n", " | ")))
    if probe_err:
        print("probe_stderr {}".format(probe_err.strip().replace("\n", " | ")))
    print("cross_session_sync_result holder_rc={} probe_rc={} pass={}".format(
        holder.returncode, probe.returncode, 1 if ok else 0))
    try:
        os.unlink(signal_path)
    except FileNotFoundError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

