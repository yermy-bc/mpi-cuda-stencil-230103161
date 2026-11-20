#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
false_sharing_lab.py
====================
Cache Line & False Sharing Mechanics - Instructor: Sufyan bin Uzayr

Student : Ainat Yermurat (ID 230103161)
Git     : yermy-bc

Measures genuine hardware cache-coherency (MESI/MOESI RFO ping-pong) cost from Python
by running GIL-free Numba kernels (@njit(nogil=True)) on 4 pinned threading.Thread
workers that hammer 64-byte-aligned uint64 slots placed at strides 1, 4, 8 and 16.

Tasks
-----
Task 1  init_system_context        - topology, cache-line size, SHA-256 INSTANCE_SEED, affinity
Task 2  worker_strided_accumulate  - nogil RMW kernel, real load+store to buffer[slot] every iter
Task 3  run_stride_benchmark       - 4 threads x 60,000,000 updates, strides {1,4,8,16},
                                     perf_counter_ns timing + getrusage context switches
Task 4  worker_private_accumulate  - register-privatized control (one shared store at the end)
Task 5  audit_llvm_and_parity      - bit-exact parity, LLVM IR i64 load/store audit,
                                     cache_lab_manifest.json

Run:  python false_sharing_lab.py
"""

import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import statistics
import datetime
import numpy as np
import numba
from numba import njit, uint64

try:
    import resource
except ImportError:  # Windows
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "yermy-bc"
STUDENT_NAME = "Ainat Yermurat"
STUDENT_ID = "230103161"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
REPEATS = 3                      # each configuration is run REPEATS times, median reported
GIL_ITERS_PER_THREAD = 2_000_000  # pure-Python (GIL) comparison for Q5 (60M would take minutes)
MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15

LAYOUT_NAMES = {
    1: "Layout A: Contiguous (8 B)",
    4: "Layout B: Half-Line (32 B)",
    8: "Layout C: Line-Padded (64 B)",
    16: "Layout D: Prefetch-Safe (128 B)",
}


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"],
                                      encoding="utf-8")
        return int(out.strip())
    return 64


# ==============================================================================
# Helpers: CPU model, physical-core selection, per-thread affinity
# ==============================================================================
def _cpu_model_string() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware", "processor\t: arm")):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        elif sys.platform == "win32":
            try:
                import winreg
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            except Exception:
                pass
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def _linux_physical_cores(allowed):
    """Pick one logical CPU per distinct physical core (avoid SMT siblings sharing L1)."""
    seen, picks = set(), []
    for cpu in sorted(allowed):
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(f"{base}/core_id") as f:
                core_id = int(f.read())
            with open(f"{base}/physical_package_id") as f:
                pkg = int(f.read())
        except Exception:
            return None
        if (pkg, core_id) not in seen:
            seen.add((pkg, core_id))
            picks.append(cpu)
    return picks


def _choose_cores(logical_cores):
    """Return (list of NUM_THREADS logical CPU ids, affinity_mode string, physical core count)."""
    if hasattr(os, "sched_setaffinity"):                 # Linux
        allowed = sorted(os.sched_getaffinity(0))
        phys = _linux_physical_cores(allowed)
        if phys and len(phys) >= NUM_THREADS:
            return phys[:NUM_THREADS], "linux:sched_setaffinity (distinct physical cores)", len(phys)
        if len(allowed) >= NUM_THREADS:
            return allowed[:NUM_THREADS], "linux:sched_setaffinity (logical cores)", (len(phys) if phys else None)
        cores = [allowed[t % len(allowed)] for t in range(NUM_THREADS)]
        return cores, (f"linux:sched_setaffinity (WARNING: only {len(allowed)} CPUs allowed, "
                       f"threads share cores)"), (len(phys) if phys else None)
    if sys.platform == "win32":
        # Windows enumerates SMT siblings adjacently (0/1, 2/3, ...): use even ids if possible
        phys = None
        try:
            import psutil  # optional
            phys = psutil.cpu_count(logical=False)
        except Exception:
            pass
        smt = (phys is not None and logical_cores >= 2 * phys) or (phys is None and logical_cores >= 2 * NUM_THREADS)
        if smt and logical_cores >= 2 * NUM_THREADS:
            return [2 * i for i in range(NUM_THREADS)], "win32:SetThreadAffinityMask (even logical ids)", phys
        return list(range(NUM_THREADS)), "win32:SetThreadAffinityMask (logical ids)", phys
    return list(range(NUM_THREADS)), "unsupported (OS scheduler places threads)", None


def _pin_current_thread(core):
    """Pin the calling OS thread to one logical CPU. Returns True on success."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})          # pid 0 == calling thread on Linux
            return True
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            return k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << core) != 0
    except Exception:
        return False
    return False


def _rusage():
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_nvcsw, r.ru_nivcsw)


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")
    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # [Task 1] deterministic 64-bit seed: SHA-256(git_user + cpu_sig), first 16 hex digits
    seed_material = f"{git_user.strip()}{cpu_sig}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cores, affinity_mode, physical = _choose_cores(logical_cores)
    thread_seeds = [((instance_seed ^ ((t + 1) * GOLDEN64)) & MASK64) for t in range(NUM_THREADS)]

    return {
        "git_username": git_user.strip(),
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "logical_cores": logical_cores,
        "physical_cores": physical,
        "cache_line_size": line_size,
        "elements_per_line": line_size // 8,
        "cpu_sig": cpu_sig,
        "cpu_model": _cpu_model_string(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "seed_sha256": digest,
        "instance_seed": instance_seed,
        "thread_seeds": thread_seeds,
        "affinity_cores": cores,
        "affinity_mode": affinity_mode,
        "resource_available": resource is not None,
    }


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel (real memory RMW each iteration)
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    """buffer[slot] += inc, iters times, with a genuine i64 load + store every iteration.

    idx = slot + (v >> 63) is always == slot at runtime (v < 2**63 throughout), but the
    compiler cannot prove it: the address of iteration k depends on the value loaded in
    iteration k-1, so LLVM's LICM / scalar promotion cannot keep buffer[slot] in a register.
    """
    inc = (seed & uint64(0xFF)) | uint64(1)
    s = uint64(slot)
    sh = uint64(63)
    v = uint64(0)
    for _ in range(iters):
        idx = s + (v >> sh)
        v = buffer[idx] + inc          # real load (store-forwarded or coherence miss)
        buffer[idx] = v                # real store -> RFO when the line is not Modified here
    return v


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Same arithmetic in a register; exactly one store to shared memory at the end."""
    inc = (seed & uint64(0xFF)) | uint64(1)
    sh = uint64(63)
    acc = uint64(0)
    for _ in range(iters):
        acc = acc + inc + (acc >> sh)  # (acc>>63)==0 at runtime; blocks closed-form folding
    buffer[uint64(slot)] = acc
    return acc


def _aligned_buffer(n_elems, align=64):
    """uint64 array whose first element starts on an `align`-byte boundary."""
    raw = np.zeros(n_elems + align // 8, dtype=np.uint64)
    off = (-raw.ctypes.data % align) // 8
    buf = raw[off:off + n_elems]
    assert buf.ctypes.data % align == 0
    return buf


def _thread_body(fn, buf, slot, iters, seed, core, barrier, results, pinned, t):
    pinned[t] = _pin_current_thread(core)
    barrier.wait()
    results[t] = int(fn(buf, np.uint64(slot), np.int64(iters), np.uint64(seed)))


def _run_threads(ctx, fn, stride, iters):
    """One timed run: NUM_THREADS concurrent workers. Returns dict with timing & state."""
    n_elems = stride * NUM_THREADS + 16
    buf = _aligned_buffer(n_elems)
    slots = [t * stride for t in range(NUM_THREADS)]
    results = [None] * NUM_THREADS
    pinned = [False] * NUM_THREADS
    barrier = threading.Barrier(NUM_THREADS + 1)
    threads = [threading.Thread(target=_thread_body,
                                args=(fn, buf, slots[t], iters, ctx["thread_seeds"][t],
                                      ctx["affinity_cores"][t], barrier, results, pinned, t))
               for t in range(NUM_THREADS)]
    for th in threads:
        th.start()
    ru0 = _rusage()
    barrier.wait()                       # all workers pinned and ready
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    t1 = time.perf_counter_ns()
    ru1 = _rusage()
    return {
        "elapsed_ns": t1 - t0,
        "slots": slots,
        "byte_offsets": [s * 8 for s in slots],
        "cache_lines": [(buf.ctypes.data + 8 * s) // ctx["cache_line_size"]
                        - buf.ctypes.data // ctx["cache_line_size"] for s in slots],
        "final_values": [int(buf[s]) for s in slots],
        "returned_values": results,
        "untouched_zero": bool(np.all(np.delete(buf, slots) == 0)),
        "pinned": pinned,
        "voluntary_ctx_switches": None if ru0 is None else ru1[0] - ru0[0],
        "involuntary_ctx_switches": None if ru0 is None else ru1[1] - ru0[1],
    }


def _summarize(runs, iters):
    times = [r["elapsed_ns"] for r in runs]
    med = int(statistics.median(times))
    last = runs[-1]
    out = {
        "elapsed_ns_runs": times,
        "elapsed_ns_median": med,
        "elapsed_ns_min": min(times),
        "ns_per_update": med / iters,                       # per thread (threads run concurrently)
        "ns_per_update_aggregate": med / (iters * NUM_THREADS),
        "updates_per_sec_total": (iters * NUM_THREADS) / (med / 1e9),
        "byte_offsets": last["byte_offsets"],
        "cache_lines": last["cache_lines"],
        "final_values": last["final_values"],
        "returned_values": last["returned_values"],
        "untouched_zero": all(r["untouched_zero"] for r in runs),
        "pinned": last["pinned"],
        "voluntary_ctx_switches": [r["voluntary_ctx_switches"] for r in runs],
        "involuntary_ctx_switches": [r["involuntary_ctx_switches"] for r in runs],
        "all_runs_final_values": [r["final_values"] for r in runs],
    }
    return out


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx) -> dict:
    iters = ITERATIONS_PER_THREAD
    results = {}
    for stride in STRIDES:
        runs = [_run_threads(ctx, worker_strided_accumulate, stride, iters) for _ in range(REPEATS)]
        s = _summarize(runs, iters)
        s["stride"] = stride
        s["stride_bytes"] = stride * 8
        s["layout"] = LAYOUT_NAMES[stride]
        results[str(stride)] = s
        print(f"  stride={stride:2d} ({stride * 8:3d} B) | lines={s['cache_lines']} | "
              f"median={s['elapsed_ns_median']:>14,d} ns | {s['ns_per_update']:.3f} ns/update | "
              f"runs={[f'{t / 1e6:.1f}ms' for t in s['elapsed_ns_runs']]} | "
              f"ctxsw vol={s['voluntary_ctx_switches']} invol={s['involuntary_ctx_switches']}",
              flush=True)
    return results


def run_private_baseline(ctx) -> dict:
    iters = ITERATIONS_PER_THREAD
    runs = [_run_threads(ctx, worker_private_accumulate, 1, iters) for _ in range(REPEATS)]
    s = _summarize(runs, iters)
    s["stride"] = 1
    s["layout"] = "Register-privatized (stride 1, single final store)"
    print(f"  private (stride 1)   | median={s['elapsed_ns_median']:>14,d} ns | "
          f"{s['ns_per_update']:.3f} ns/update | runs={[f'{t / 1e6:.1f}ms' for t in s['elapsed_ns_runs']]}",
          flush=True)
    return s


# ------------------------------------------------------------------------------
# Q5 support: identical stride-1 experiment with pure-Python (GIL-bound) threads
# ------------------------------------------------------------------------------
def _py_worker(buf, slot, iters, seed):
    inc = (seed & 0xFF) | 1
    v = 0
    for _ in range(iters):
        idx = slot + (v >> 63)
        v = (buf[idx] + inc) & MASK64
        buf[idx] = v
    return v


def run_gil_experiment(ctx) -> dict:
    out = {"iters_per_thread": GIL_ITERS_PER_THREAD}
    for label, nthreads, stride in (("1_thread_stride1", 1, 1), ("4_threads_stride1", 4, 1),
                                    ("4_threads_stride8", 4, 8)):
        buf = [0] * (stride * NUM_THREADS + 16)
        ths = [threading.Thread(target=_py_worker,
                                args=(buf, t * stride, GIL_ITERS_PER_THREAD, ctx["thread_seeds"][t]))
               for t in range(nthreads)]
        t0 = time.perf_counter_ns()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        el = time.perf_counter_ns() - t0
        total = GIL_ITERS_PER_THREAD * nthreads
        out[label] = {"threads": nthreads, "stride": stride, "elapsed_ns": el,
                      "ns_per_update_aggregate": el / total}
        print(f"  pure-Python GIL {label:<18s} | {el:>14,d} ns | {el / total:.2f} ns/update (aggregate)",
              flush=True)
    out["gil_4t_vs_1t_wall_ratio"] = out["4_threads_stride1"]["elapsed_ns"] / out["1_thread_stride1"]["elapsed_ns"]
    out["gil_stride1_vs_stride8_ratio"] = (out["4_threads_stride1"]["elapsed_ns"]
                                           / out["4_threads_stride8"]["elapsed_ns"])
    out["switch_interval_s"] = sys.getswitchinterval()
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def _main_function_ir(llvm_text, name):
    """Extract the native body of `name` (not the CPython wrapper / cfunc)."""
    best = None
    for chunk in llvm_text.split("\ndefine ")[1:]:
        header = chunk.split("{", 1)[0]
        if name in header and "cpython" not in header and "cfunc" not in header:
            body = chunk.split("\n}\n", 1)[0]
            if best is None or len(body) > len(best):
                best = body
    return best or ""


def _llvm_stats(fn, name):
    ir = fn.inspect_llvm()
    text = "\n".join(ir.values()) if isinstance(ir, dict) else str(ir)
    body = _main_function_ir(text, name)
    return {
        "signatures": [str(s) for s in fn.signatures],
        "store_i64": body.count("store i64"),
        "load_i64": body.count("load i64"),
        "loop_backedges_br": body.count("br i1"),
        "phi_nodes": body.count(" phi "),
        "pyobject_refs_in_body": body.count("PyObject") + body.count("@Py"),
        "nrt_calls_in_body": body.count("@NRT_"),
        "body_lines": body.count("\n") + 1 if body else 0,
        "whole_module_lines": text.count("\n") + 1,
    }, body


def audit_llvm_and_parity(ctx, stride_results, private_result) -> dict:
    iters = ITERATIONS_PER_THREAD
    expected = []
    for seed in ctx["thread_seeds"]:
        inc = (seed & 0xFF) | 1
        expected.append((inc * iters) & MASK64)

    parity = {}
    ok = True
    for key, r in list(stride_results.items()) + [("private", private_result)]:
        this_ok = (all(v == expected for v in r["all_runs_final_values"])
                   and r["returned_values"] == expected and r["untouched_zero"])
        parity[key] = this_ok
        ok = ok and this_ok

    strided_stats, strided_body = _llvm_stats(worker_strided_accumulate, "worker_strided_accumulate")
    private_stats, _ = _llvm_stats(worker_private_accumulate, "worker_private_accumulate")
    llvm_ok = (strided_stats["store_i64"] >= 1 and strided_stats["load_i64"] >= 1
               and strided_stats["pyobject_refs_in_body"] == 0)
    # Show the core loop instructions of the strided kernel
    loop_lines = [l.strip() for l in strided_body.splitlines()
                  if ("load i64" in l or "store i64" in l or " phi " in l
                      or "lshr" in l or "add i64" in l or "br i1" in l)]
    return {
        "expected_final_values": expected,
        "parity_per_layout": parity,
        "parity_passed": ok,
        "llvm_strided": strided_stats,
        "llvm_private": private_stats,
        "llvm_strided_loop_excerpt": loop_lines[:40],
        "llvm_passed": llvm_ok,
    }


def main():
    print("=" * 78)
    print(" Cache-Line False Sharing Lab (Numba nogil + threading)")
    print(f" Student: {STUDENT_NAME} | ID: {STUDENT_ID} | Git: {GIT_USERNAME}")
    print("=" * 78, flush=True)

    ctx = init_system_context(GIT_USERNAME)
    print(f"[Task 1] CPU model        : {ctx['cpu_model']}")
    print(f"[Task 1] platform         : {ctx['platform']} | python {ctx['python']} | "
          f"numpy {ctx['numpy']} | numba {ctx['numba']}")
    print(f"[Task 1] logical cores    : {ctx['logical_cores']} | physical cores: {ctx['physical_cores']}")
    print(f"[Task 1] cache line size  : {ctx['cache_line_size']} B ({ctx['elements_per_line']} uint64 per line)")
    print(f"[Task 1] cpu_sig          : {ctx['cpu_sig']}")
    print(f"[Task 1] SHA-256          : {ctx['seed_sha256']}")
    print(f"[Task 1] INSTANCE_SEED    : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    print(f"[Task 1] affinity         : cores {ctx['affinity_cores']} via {ctx['affinity_mode']}")
    print(f"[Task 1] getrusage        : {'available' if ctx['resource_available'] else 'not available on this OS'}")

    # JIT warm-up (compile both kernels with the exact runtime signature)
    t = time.perf_counter()
    wb = _aligned_buffer(32)
    worker_strided_accumulate(wb, np.uint64(0), np.int64(1000), np.uint64(ctx["thread_seeds"][0]))
    worker_private_accumulate(wb, np.uint64(1), np.int64(1000), np.uint64(ctx["thread_seeds"][0]))
    print(f"[Task 2] Numba JIT warm-up: {time.perf_counter() - t:.2f} s", flush=True)

    print(f"[Task 3] {NUM_THREADS} threads x {ITERATIONS_PER_THREAD:,} updates, "
          f"{REPEATS} runs per layout (median reported)", flush=True)
    stride_results = run_stride_benchmark(ctx)
    print("[Task 4] register-privatized control", flush=True)
    private_result = run_private_baseline(ctx)
    print("[Q5]     pure-Python threads (GIL held), "
          f"{GIL_ITERS_PER_THREAD:,} updates per thread", flush=True)
    gil = run_gil_experiment(ctx)

    audit = audit_llvm_and_parity(ctx, stride_results, private_result)

    t8 = stride_results["8"]["elapsed_ns_median"]
    t1 = stride_results["1"]["elapsed_ns_median"]
    t16 = stride_results["16"]["elapsed_ns_median"]
    tp = private_result["elapsed_ns_median"]
    ratios = {
        "stride1_over_stride8": t1 / t8,
        "stride4_over_stride8": stride_results["4"]["elapsed_ns_median"] / t8,
        "stride16_over_stride8": t16 / t8,
        "stride8_over_private": t8 / tp,
        "stride1_over_private": t1 / tp,
    }

    print("-" * 78)
    print("[Task 5] Summary (median of runs, per-thread ns/update):")
    for s in STRIDES:
        r = stride_results[str(s)]
        print(f"   {r['layout']:<34s} {r['elapsed_ns_median']:>14,d} ns  {r['ns_per_update']:7.3f} ns/update  "
              f"x{r['elapsed_ns_median'] / t8:6.2f} vs stride 8")
    print(f"   {'Register-privatized baseline':<34s} {tp:>14,d} ns  {private_result['ns_per_update']:7.3f} ns/update")
    print(f"[Task 5] slowdown stride1/stride8 = {ratios['stride1_over_stride8']:.2f}x | "
          f"stride16/stride8 = {ratios['stride16_over_stride8']:.3f} | "
          f"stride8/private = {ratios['stride8_over_private']:.2f}x")
    print(f"[Task 5] expected final values : {audit['expected_final_values']}")
    print(f"[Task 5] parity per layout     : {audit['parity_per_layout']} -> {audit['parity_passed']}")
    ls = audit["llvm_strided"]
    print(f"[Task 5] LLVM worker_strided_accumulate: signatures={ls['signatures']}")
    print(f"           store i64={ls['store_i64']} load i64={ls['load_i64']} phi={ls['phi_nodes']} "
          f"br i1={ls['loop_backedges_br']} PyObject refs={ls['pyobject_refs_in_body']} "
          f"NRT calls={ls['nrt_calls_in_body']} body lines={ls['body_lines']}")
    lp = audit["llvm_private"]
    print(f"[Task 5] LLVM worker_private_accumulate: store i64={lp['store_i64']} load i64={lp['load_i64']} "
          f"phi={lp['phi_nodes']} PyObject refs={lp['pyobject_refs_in_body']}")
    print("[Task 5] strided kernel IR excerpt (loads/stores/phi/shift/add/branch):")
    for l in audit["llvm_strided_loop_excerpt"]:
        print("           " + l)
    verification = bool(audit["parity_passed"] and audit["llvm_passed"])
    print(f"[Task 5] llvm_passed = {audit['llvm_passed']} | verification_passed = {verification}")

    manifest = {
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "git_username": GIT_USERNAME,
        "GIT_USERNAME": GIT_USERNAME,
        "repository": f"cache-line-false-sharing-{GIT_USERNAME}",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system": {k: ctx[k] for k in ("cpu_model", "cpu_sig", "platform", "python", "numpy", "numba",
                                       "logical_cores", "physical_cores", "cache_line_size",
                                       "elements_per_line", "affinity_cores", "affinity_mode",
                                       "resource_available")},
        "seed_sha256": ctx["seed_sha256"],
        "INSTANCE_SEED": str(ctx["instance_seed"]),
        "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
        "thread_seeds": [str(s) for s in ctx["thread_seeds"]],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "stride_results": stride_results,
        "private_baseline": private_result,
        "slowdown_ratios": ratios,
        "gil_experiment": gil,
        "parity": {"expected_final_values": [str(v) for v in audit["expected_final_values"]],
                   "per_layout": audit["parity_per_layout"], "passed": audit["parity_passed"]},
        "llvm_audit": {"worker_strided_accumulate": audit["llvm_strided"],
                       "worker_private_accumulate": audit["llvm_private"],
                       "strided_loop_excerpt": audit["llvm_strided_loop_excerpt"],
                       "passed": audit["llvm_passed"]},
        "verification_passed": verification,
    }
    # JSON cannot hold Python ints > 2**63 safely for every consumer -> stringify big values
    for r in list(stride_results.values()) + [private_result]:
        for k in ("final_values", "returned_values"):
            r[k] = [str(v) for v in r[k]]
        r["all_runs_final_values"] = [[str(v) for v in run] for run in r["all_runs_final_values"]]
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[Task 5] cache_lab_manifest.json written")
    print("ALL CHECKS PASSED" if verification else "SOME CHECKS FAILED")
    return 0 if verification else 1


if __name__ == "__main__":
    sys.exit(main())
