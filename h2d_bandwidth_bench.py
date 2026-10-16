#!/usr/bin/env python3
"""
h2d_bandwidth_bench.py

Measures real Host-to-Device (H2D) transfer bandwidth on Ascend 910c NPUs
using torch_npu.

What it does
------------
1. Single-device sweep:
   For each requested device, for a log sweep of sizes (4KB -> 1GB),
   measures H2D transfer bandwidth for both pinned and pageable host
   memory (3 warm-up + 20 timed iterations per size, buffers allocated
   once and reused).

2. Multi-device sequential-vs-concurrent sweep:
   Same log sweep of sizes, across all requested devices at once.
   For each size, transfers are launched (a) sequentially, one device
   after another, and (b) concurrently, via one Python thread per
   device with a barrier-synchronized start. 3 warm-up + 10 timed
   iterations per size. Reports aggregate throughput (sum across
   devices) for both pinned and pageable memory.

Outputs (written to a timestamped directory):
   - summary.csv       : one row per (test, size, device(s), mode) with
                          median/mean/min/max bandwidth (decimal GB/s)
   - raw_timings.csv    : one row per individual timed iteration
   - single_device_bandwidth.png : pinned vs pageable bandwidth vs size
   - multi_device_scaling.png    : sequential vs concurrent aggregate
                                    throughput vs size (pinned & pageable)

Usage
-----
    python h2d_bandwidth_bench.py --devices 0,1,2,3
    python h2d_bandwidth_bench.py --devices 0,2 --dtype float32
    python h2d_bandwidth_bench.py --devices 0 --output-dir ./my_run

Notes
-----
- Devices are NOT auto-detected: if a requested device id is invalid,
  the script fails immediately with a clear error.
- Bandwidth is reported in decimal GB/s (1 GB = 1e9 bytes).
- The dtype used for test buffers defaults to uint8 (raw bytes, exact
  byte-count sizing) but can be changed via --dtype.
"""

import argparse
import csv
import os
import statistics
import sys
import threading
import time
from datetime import datetime

import torch
import torch_npu  # noqa: F401  (registers the 'npu' backend with torch)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Log sweep of transfer sizes, in bytes: 4KB -> 1GB
# SIZES_BYTES = [
#     4 * 1024,          # 4 KB
#     16 * 1024,         # 16 KB
#     64 * 1024,         # 64 KB
#     256 * 1024,        # 256 KB
#     1 * 1024 * 1024,   # 1 MB
#     4 * 1024 * 1024,   # 4 MB
#     16 * 1024 * 1024,  # 16 MB
#     64 * 1024 * 1024,  # 64 MB
#     256 * 1024 * 1024,  # 256 MB
#     1024 * 1024 * 1024,  # 1 GB
# ]

# Dense
SIZES_BYTES = [
    4 * 1024 * 1024,   # 4 MB
    8 * 1024 * 1024,   # 8 MB
    12 * 1024 * 1024,  # 12 MB
    16 * 1024 * 1024,  # 16 MB
    24 * 1024 * 1024,  # 24 MB
    32 * 1024 * 1024,  # 32 MB
    48 * 1024 * 1024,  # 48 MB
    64 * 1024 * 1024,  # 64 MB
]

SINGLE_DEVICE_WARMUP = 3
SINGLE_DEVICE_ITERS = 20

MULTI_DEVICE_WARMUP = 3
MULTI_DEVICE_ITERS = 10

# Map of accepted --dtype CLI strings to torch dtypes and their byte width.
# To add a new dtype: add an entry here (element_size must match torch's).
DTYPE_MAP = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def human_size(n_bytes):
    """Pretty-print a byte count, e.g. 1048576 -> '1MB'."""
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n_bytes >= size and n_bytes % size == 0:
            return f"{n_bytes // size}{unit}"
    return f"{n_bytes}B"


def num_elements_for_bytes(n_bytes, dtype):
    """How many elements of `dtype` are needed to hold exactly n_bytes."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    if n_bytes % elem_size != 0:
        raise ValueError(
            f"Size {n_bytes} bytes is not evenly divisible by element size "
            f"{elem_size} bytes for dtype {dtype}."
        )
    return n_bytes // elem_size


# --------------------------------------------------------------------------
# Device validation
# --------------------------------------------------------------------------

def validate_devices(requested_devices):
    """Fail immediately and clearly if any requested device id is invalid."""
    try:
        available = torch.npu.device_count()
    except Exception as exc:
        print(f"ERROR: could not query NPU device count via torch_npu: {exc}")
        sys.exit(1)

    if available == 0:
        print("ERROR: no NPU devices detected on this system.")
        sys.exit(1)

    bad = [d for d in requested_devices if d < 0 or d >= available]
    if bad:
        print(
            f"ERROR: invalid device id(s) {bad}. "
            f"This system has {available} NPU device(s): 0..{available - 1}."
        )
        sys.exit(1)

    print(f"Detected {available} NPU device(s). Using requested devices: "
          f"{requested_devices}")
    for d in requested_devices:
        try:
            name = torch.npu.get_device_name(d)
        except Exception:
            name = "<name unavailable>"
        print(f"  device {d}: {name}")


# --------------------------------------------------------------------------
# Core single-transfer timing primitive
# --------------------------------------------------------------------------

def time_one_h2d_transfer(host_tensor, device_tensor, device_id):
    """Copy host_tensor -> device_tensor on device_id, return elapsed seconds."""
    torch.npu.synchronize(device_id)
    start = time.perf_counter()
    device_tensor.copy_(host_tensor, non_blocking=False)
    torch.npu.synchronize(device_id)
    end = time.perf_counter()
    return end - start


def make_buffers(n_elements, dtype, device_id, pinned):
    host = torch.empty(n_elements, dtype=dtype, pin_memory=pinned)
    device = torch.empty(n_elements, dtype=dtype, device=f"npu:{device_id}")
    if pinned:
        host = host.contiguous()
    return host, device


# --------------------------------------------------------------------------
# Single-device sweep
# --------------------------------------------------------------------------

def run_single_device_sweep(devices, dtype, raw_rows, summary_rows):
    print("\n=== Single-device H2D bandwidth sweep ===")
    for device_id in devices:
        torch.npu.set_device(device_id)
        print(f"\n-- Device {device_id} --")
        print(f"{'Size':>10} | {'Pinned GB/s':>12} | {'Pageable GB/s':>14}")
        print("-" * 42)

        for size_bytes in SIZES_BYTES:
            n_elem = num_elements_for_bytes(size_bytes, dtype)
            row_result = {}

            for pinned in (True, False):
                mode = "pinned" if pinned else "pageable"
                host_buf, dev_buf = make_buffers(n_elem, dtype, device_id, pinned)

                # Warm-up (untimed)
                for _ in range(SINGLE_DEVICE_WARMUP):
                    time_one_h2d_transfer(host_buf, dev_buf, device_id)

                # Timed iterations
                times = []
                for it in range(SINGLE_DEVICE_ITERS):
                    t = time_one_h2d_transfer(host_buf, dev_buf, device_id)
                    times.append(t)
                    bw = size_bytes / t / 1e9
                    raw_rows.append({
                        "test": "single_device",
                        "device": device_id,
                        "size_bytes": size_bytes,
                        "mode": mode,
                        "scenario": "n/a",
                        "iteration": it,
                        "time_s": t,
                        "bandwidth_gbs": bw,
                    })

                del host_buf, dev_buf

                bws = [size_bytes / t / 1e9 for t in times]
                median_bw = statistics.median(bws)
                mean_bw = statistics.mean(bws)
                min_bw = min(bws)
                max_bw = max(bws)

                summary_rows.append({
                    "test": "single_device",
                    "device": device_id,
                    "size_bytes": size_bytes,
                    "mode": mode,
                    "scenario": "n/a",
                    "median_bandwidth_gbs": median_bw,
                    "mean_bandwidth_gbs": mean_bw,
                    "min_bandwidth_gbs": min_bw,
                    "max_bandwidth_gbs": max_bw,
                })
                row_result[mode] = median_bw

            print(f"{human_size(size_bytes):>10} | "
                  f"{row_result['pinned']:>12.2f} | "
                  f"{row_result['pageable']:>14.2f}")


# --------------------------------------------------------------------------
# Multi-device sequential vs concurrent sweep
# --------------------------------------------------------------------------

def run_multi_device_sweep(devices, dtype, raw_rows, summary_rows):
    if len(devices) < 2:
        print("\n(Skipping multi-device comparison: fewer than 2 devices requested.)")
        return

    print("\n=== Multi-device sequential vs concurrent H2D sweep ===")
    print(f"Devices: {devices}")
    print(f"{'Size':>10} | {'Mode':>9} | {'Seq GB/s (sum)':>15} | {'Conc GB/s (sum)':>16}")
    print("-" * 60)

    for size_bytes in SIZES_BYTES:
        n_elem = num_elements_for_bytes(size_bytes, dtype)

        for pinned in (True, False):
            mode = "pinned" if pinned else "pageable"

            # Allocate one host/device buffer pair per device, reused across
            # warm-up and timed iterations, for both scenarios.
            buffers = {}
            for device_id in devices:
                host_buf, dev_buf = make_buffers(n_elem, dtype, device_id, pinned)
                buffers[device_id] = (host_buf, dev_buf)

            # --- Warm-up (both scenarios use the same warm-up) ---
            for _ in range(MULTI_DEVICE_WARMUP):
                for device_id in devices:
                    host_buf, dev_buf = buffers[device_id]
                    time_one_h2d_transfer(host_buf, dev_buf, device_id)

            # --- Sequential timed iterations ---
            seq_aggregate_bws = []
            for it in range(MULTI_DEVICE_ITERS):
                per_device_times = {}
                for device_id in devices:
                    host_buf, dev_buf = buffers[device_id]
                    t = time_one_h2d_transfer(host_buf, dev_buf, device_id)
                    per_device_times[device_id] = t
                    bw = size_bytes / t / 1e9
                    raw_rows.append({
                        "test": "multi_device",
                        "device": device_id,
                        "size_bytes": size_bytes,
                        "mode": mode,
                        "scenario": "sequential",
                        "iteration": it,
                        "time_s": t,
                        "bandwidth_gbs": bw,
                    })
                # Aggregate throughput for this iteration: total bytes moved
                # across all devices divided by total wall time spent.
                total_bytes = size_bytes * len(devices)
                total_time = sum(per_device_times.values())
                seq_aggregate_bws.append(total_bytes / total_time / 1e9)

            # --- Concurrent timed iterations (threads + barrier) ---
            conc_aggregate_bws = []
            for it in range(MULTI_DEVICE_ITERS):
                barrier = threading.Barrier(len(devices))
                results = {}

                def worker(dev_id):
                    host_buf, dev_buf = buffers[dev_id]
                    torch.npu.set_device(dev_id)
                    torch.npu.synchronize(dev_id)
                    barrier.wait()  # all threads start together
                    start = time.perf_counter()
                    dev_buf.copy_(host_buf, non_blocking=False)
                    torch.npu.synchronize(dev_id)
                    end = time.perf_counter()
                    results[dev_id] = end - start

                threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
                wall_start = time.perf_counter()
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                wall_end = time.perf_counter()

                for device_id in devices:
                    t = results[device_id]
                    bw = size_bytes / t / 1e9
                    raw_rows.append({
                        "test": "multi_device",
                        "device": device_id,
                        "size_bytes": size_bytes,
                        "mode": mode,
                        "scenario": "concurrent",
                        "iteration": it,
                        "time_s": t,
                        "bandwidth_gbs": bw,
                    })

                # Aggregate throughput for the concurrent case: total bytes
                # moved across all devices divided by the wall-clock time
                # for the whole concurrent batch (not summed per-device time).
                total_bytes = size_bytes * len(devices)
                total_wall_time = wall_end - wall_start
                conc_aggregate_bws.append(total_bytes / total_wall_time / 1e9)

            for buf_pair in buffers.values():
                del buf_pair
            buffers.clear()

            seq_median = statistics.median(seq_aggregate_bws)
            conc_median = statistics.median(conc_aggregate_bws)

            summary_rows.append({
                "test": "multi_device",
                "device": ",".join(str(d) for d in devices),
                "size_bytes": size_bytes,
                "mode": mode,
                "scenario": "sequential",
                "median_bandwidth_gbs": seq_median,
                "mean_bandwidth_gbs": statistics.mean(seq_aggregate_bws),
                "min_bandwidth_gbs": min(seq_aggregate_bws),
                "max_bandwidth_gbs": max(seq_aggregate_bws),
            })
            summary_rows.append({
                "test": "multi_device",
                "device": ",".join(str(d) for d in devices),
                "size_bytes": size_bytes,
                "mode": mode,
                "scenario": "concurrent",
                "median_bandwidth_gbs": conc_median,
                "mean_bandwidth_gbs": statistics.mean(conc_aggregate_bws),
                "min_bandwidth_gbs": min(conc_aggregate_bws),
                "max_bandwidth_gbs": max(conc_aggregate_bws),
            })

            print(f"{human_size(size_bytes):>10} | {mode:>9} | "
                  f"{seq_median:>15.2f} | {conc_median:>16.2f}")


# --------------------------------------------------------------------------
# Output: CSVs and plots
# --------------------------------------------------------------------------

def write_csvs(output_dir, raw_rows, summary_rows):
    raw_path = os.path.join(output_dir, "raw_timings.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "device", "size_bytes", "mode", "scenario",
            "iteration", "time_s", "bandwidth_gbs",
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "device", "size_bytes", "mode", "scenario",
            "median_bandwidth_gbs", "mean_bandwidth_gbs",
            "min_bandwidth_gbs", "max_bandwidth_gbs",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")


def make_plots(output_dir, devices, summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes_kb = [s / 1024 for s in SIZES_BYTES]
    size_labels = [human_size(s) for s in SIZES_BYTES]

    def set_size_ticks(ax):
        """Force x-axis ticks to be exactly the tested sizes, not powers of 10."""
        ax.set_xticks(sizes_kb)
        ax.set_xticklabels(size_labels, rotation=45, ha="right")
        # Turn off log-scale minor ticks so they don't clutter the axis
        # alongside the explicit major ticks above.
        ax.minorticks_off()

    # --- Figure 1: single-device pinned vs pageable ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for device_id in devices:
        for mode, style in (("pinned", "-o"), ("pageable", "--s")):
            rows = [r for r in summary_rows
                    if r["test"] == "single_device"
                    and r["device"] == device_id
                    and r["mode"] == mode]
            rows.sort(key=lambda r: r["size_bytes"])
            xs = [r["size_bytes"] / 1024 for r in rows]
            ys = [r["median_bandwidth_gbs"] for r in rows]
            ax1.plot(xs, ys, style, label=f"dev{device_id} {mode}")
    ax1.set_xscale("log")
    set_size_ticks(ax1)
    ax1.set_xlabel("Transfer size")
    ax1.set_ylabel("H2D bandwidth (GB/s)")
    ax1.set_title("Single-device H2D bandwidth: pinned vs pageable")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "single_device_bandwidth.png")
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print(f"Wrote {fig1_path}")

    # --- Figure 2: multi-device sequential vs concurrent (pinned & pageable) ---
    if len(devices) >= 2:
        fig2, (axp, axg) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        for ax, mode in ((axp, "pinned"), (axg, "pageable")):
            for scenario, style in (("sequential", "-o"), ("concurrent", "--s")):
                rows = [r for r in summary_rows
                        if r["test"] == "multi_device"
                        and r["mode"] == mode
                        and r["scenario"] == scenario]
                rows.sort(key=lambda r: r["size_bytes"])
                xs = [r["size_bytes"] / 1024 for r in rows]
                ys = [r["median_bandwidth_gbs"] for r in rows]
                ax.plot(xs, ys, style, label=scenario)
            ax.set_xscale("log")
            set_size_ticks(ax)
            ax.set_xlabel("Transfer size")
            ax.set_ylabel("Aggregate bandwidth (GB/s)")
            ax.set_title(f"Multi-device ({mode})")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
        fig2.suptitle(f"Sequential vs concurrent aggregate H2D throughput, "
                       f"devices {devices}")
        fig2.tight_layout()
        fig2_path = os.path.join(output_dir, "multi_device_scaling.png")
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print(f"Wrote {fig2_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_devices(s):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --devices value '{s}'. Expected comma-separated integers, "
            f"e.g. --devices 0,2,4"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Test real Host-to-Device transfer bandwidth on Ascend 910c."
    )
    parser.add_argument(
        "--devices", type=parse_devices, required=True,
        help="Comma-separated NPU device ids to test, e.g. --devices 0,2,4",
    )
    parser.add_argument(
        "--dtype", type=str, default="uint8", choices=sorted(DTYPE_MAP.keys()),
        help="Element dtype for test buffers (default: uint8, raw bytes).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write results into. Default: "
             "./h2d_bench_results/<timestamp>/",
    )
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]

    validate_devices(args.devices)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("h2d_bench_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    raw_rows = []
    summary_rows = []

    run_single_device_sweep(args.devices, dtype, raw_rows, summary_rows)
    run_multi_device_sweep(args.devices, dtype, raw_rows, summary_rows)

    write_csvs(output_dir, raw_rows, summary_rows)
    make_plots(output_dir, args.devices, summary_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
