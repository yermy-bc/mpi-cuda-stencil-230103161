"""
Hybrid MPI-CUDA & Cache-Line Coherency Lab
Student : Ainat Yermurat (ID 230103161)
Instructor: Sufyan bin Uzayr

Run:  mpirun --allow-run-as-root --oversubscribe -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>
"""
import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64
from numba.core import types, cgutils
from numba.core.extending import intrinsic

STUDENT_NAME = "Ainat Yermurat"
STUDENT_ID = "230103161"

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_N = 256              # interior points per rank per dimension
PAD_N = LOCAL_N + 2        # 258 with ghost layer
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
SHM_SLOTS = 64
CFL_SAFETY = 0.8           # dt = 0.8 * dt_max
ENERGY_STEPS = (0, 100, 200, 300)

# message tags = direction the payload travels
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 11, 12, 13, 14


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def log0(ctx, msg):
    if ctx["rank"] == 0:
        print(msg, flush=True)


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    # --- Cartesian 2x2, non-periodic -------------------------------------
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)           # (row, col)
    up, down = cart_comm.Shift(0, 1)                   # row-1 / row+1
    left, right = cart_comm.Shift(1, 1)                # col-1 / col+1

    # --- GPU binding (single T4 shared by all 4 ranks) --------------------
    cuda.select_device(0)

    # --- Deterministic instance seed --------------------------------------
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}::{get_cpu_model()}"
    cpu_sig = comm.bcast(cpu_sig if rank == 0 else None, root=0)
    line_bytes = detect_cpu_line_size()
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}::{line_bytes}".encode("utf-8")
    ).hexdigest()
    instance_seed = int(digest[:16], 16)  # 64-bit

    # --- Node-local shared memory window ----------------------------------
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    itemsize = 8
    # over-allocate by one cache line so we can align the view to 64 bytes
    nbytes = (SHM_SLOTS * itemsize + 64) if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buf, qitemsize = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint64, shape=(SHM_SLOTS + 8,))
    base_addr = raw.ctypes.data
    # mmap'd segments are page-aligned in every process, so the offset is identical;
    # compute it once on node rank 0 and broadcast to be safe.
    offset_words = ((64 - (base_addr % 64)) % 64) // itemsize
    offset_words = node_comm.bcast(offset_words if node_rank == 0 else None, root=0)
    shm_view = raw[offset_words: offset_words + SHM_SLOTS]
    aligned_addr = shm_view.ctypes.data
    assert aligned_addr % 64 == 0, "shared window view is not 64-byte aligned"
    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    ctx = dict(
        comm=comm, rank=rank, size=size,
        cart_comm=cart_comm, cart_rank=cart_rank, coords=tuple(coords),
        up=up, down=down, left=left, right=right,
        node_comm=node_comm, node_rank=node_rank, node_size=node_comm.Get_size(),
        win=win, shm_view=shm_view, shm_base_addr=base_addr,
        shm_aligned_addr=aligned_addr, shm_offset_words=offset_words,
        gpu_uuid=gpu_uuid, cpu_sig=cpu_sig, line_bytes=line_bytes,
        instance_seed=instance_seed, github_user=github_user.strip(),
    )
    return ctx


# ==============================================================================
# TASK 2: Rank-local fields + halo pack/unpack kernels
# ==============================================================================
def initialize_fields(ctx: dict) -> dict:
    """Build rank-local (258,258) arrays from global coordinates only."""
    row, col = ctx["coords"]
    h = 1.0 / (GLOBAL_NX + 1)                       # dx = 1/513
    rng = np.random.default_rng(ctx["instance_seed"])
    # seed-dependent (but rank-identical) Gaussian bump parameters
    x0 = 0.35 + 0.30 * rng.random()
    y0 = 0.35 + 0.30 * rng.random()
    amp = 0.8 + 0.4 * rng.random()
    sigma = 0.08 + 0.04 * rng.random()
    phase = 2.0 * math.pi * rng.random()

    # local index i (0..257) -> global index gi = row*256 + i (0..513)
    gi = row * LOCAL_N + np.arange(PAD_N, dtype=np.float64)
    gj = col * LOCAL_N + np.arange(PAD_N, dtype=np.float64)
    Y, X = np.meshgrid(gi * h, gj * h, indexing="ij")

    kappa = 0.5 + 0.4 * np.sin(2.0 * math.pi * X + phase) * np.cos(2.0 * math.pi * Y)
    source = 0.5 * np.sin(math.pi * X) * np.sin(math.pi * Y)
    u = amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))

    # Dirichlet u = 0 on the physical boundary (global index 0 and 513),
    # and ghost cells start at zero (filled later by halo exchange).
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    source[0, :] = source[-1, :] = 0.0
    source[:, 0] = source[:, -1] = 0.0

    u = np.ascontiguousarray(u)
    kappa = np.ascontiguousarray(kappa)
    source = np.ascontiguousarray(source)

    local_kmax = float(kappa[1:-1, 1:-1].max())
    kappa_max = ctx["cart_comm"].allreduce(local_kmax, op=MPI.MAX)
    dt_max = h * h / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    return dict(
        u0=u, kappa=kappa, source=source, h=h, dt=dt, dt_max=dt_max,
        kappa_max=kappa_max,
        init_params=dict(x0=x0, y0=y0, amp=amp, sigma=sigma, phase=phase),
    )


@cuda.jit
def pack_halos_kernel(u, send_buf):
    """send_buf layout (4*256): [top row | bottom row | left col | right col]."""
    t = cuda.grid(1)
    if t < 4 * LOCAL_N:
        side = t // LOCAL_N
        k = t % LOCAL_N + 1
        if side == 0:
            send_buf[t] = u[1, k]               # contiguous row 1
        elif side == 1:
            send_buf[t] = u[LOCAL_N, k]         # contiguous row 256
        elif side == 2:
            send_buf[t] = u[k, 1]               # strided column 1
        else:
            send_buf[t] = u[k, LOCAL_N]         # strided column 256


@cuda.jit
def unpack_halos_kernel(u, recv_buf, has_up, has_down, has_left, has_right):
    """recv_buf layout: [from up -> row 0 | from down -> row 257 |
                         from left -> col 0 | from right -> col 257]."""
    t = cuda.grid(1)
    if t < 4 * LOCAL_N:
        side = t // LOCAL_N
        k = t % LOCAL_N + 1
        if side == 0:
            if has_up:
                u[0, k] = recv_buf[t]
        elif side == 1:
            if has_down:
                u[LOCAL_N + 1, k] = recv_buf[t]
        elif side == 2:
            if has_left:
                u[k, 0] = recv_buf[t]
        else:
            if has_right:
                u[k, LOCAL_N + 1] = recv_buf[t]


# ==============================================================================
# TASK 3: 16x16 threadblock stencil with 18x18 shared tile + dual streams
# ==============================================================================
STENCIL_SIG = ("void(float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[:, ::1], "
               "float64, float64, float64, int64, int64, int64, int64)")


@cuda.jit(STENCIL_SIG)
def tiled_stencil_kernel(u, kappa, source, u_new, dt, inv_h2, lam, i0, i1, j0, j1):
    """Updates u_new[i,j] for i in [i0,i1], j in [j0,j1] (inclusive).
    u_t = div(kappa grad u) - lam*u + S, explicit Euler, face kappa = arithmetic mean."""
    s_u = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    s_k = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    base_i = i0 + cuda.blockIdx.y * TILE_DIM
    base_j = j0 + cuda.blockIdx.x * TILE_DIM
    tid = ty * TILE_DIM + tx

    # cooperative 18x18 load (324 cells by 256 threads), restricted to region +/- 1
    idx = tid
    while idx < SHARED_DIM * SHARED_DIM:
        si = idx // SHARED_DIM
        sj = idx % SHARED_DIM
        gi = base_i - 1 + si
        gj = base_j - 1 + sj
        if gi >= i0 - 1 and gi <= i1 + 1 and gj >= j0 - 1 and gj <= j1 + 1:
            s_u[si, sj] = u[gi, gj]
            s_k[si, sj] = kappa[gi, gj]
        else:
            s_u[si, sj] = 0.0
            s_k[si, sj] = 0.0
        idx += TILE_DIM * TILE_DIM

    cuda.syncthreads()

    i = base_i + ty
    j = base_j + tx
    if i <= i1 and j <= j1:
        si = ty + 1
        sj = tx + 1
        uc = s_u[si, sj]
        kc = s_k[si, sj]
        k_n = 0.5 * (kc + s_k[si - 1, sj])
        k_s = 0.5 * (kc + s_k[si + 1, sj])
        k_w = 0.5 * (kc + s_k[si, sj - 1])
        k_e = 0.5 * (kc + s_k[si, sj + 1])
        flux = (k_e * (s_u[si, sj + 1] - uc) - k_w * (uc - s_u[si, sj - 1])
                + k_s * (s_u[si + 1, sj] - uc) - k_n * (uc - s_u[si - 1, sj]))
        u_new[i, j] = uc + dt * (flux * inv_h2 - lam * uc + source[i, j])


def launch_stencil(region, stream, d_u, d_k, d_s, d_unew, dt, inv_h2):
    i0, i1, j0, j1 = region
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    grid = ((nj + TILE_DIM - 1) // TILE_DIM, (ni + TILE_DIM - 1) // TILE_DIM)
    block = (TILE_DIM, TILE_DIM)
    tiled_stencil_kernel[grid, block, stream](
        d_u, d_k, d_s, d_unew, dt, inv_h2, LAMBDA_DAMP, i0, i1, j0, j1)


FULL_REGION = (1, LOCAL_N, 1, LOCAL_N)
DEEP_REGION = (2, LOCAL_N - 1, 2, LOCAL_N - 1)            # [2..255, 2..255]
BORDER_REGIONS = (
    (1, 1, 1, LOCAL_N),                                    # top row
    (LOCAL_N, LOCAL_N, 1, LOCAL_N),                        # bottom row
    (2, LOCAL_N - 1, 1, 1),                                # left column
    (2, LOCAL_N - 1, LOCAL_N, LOCAL_N),                    # right column
)
HALO_THREADS = 256
HALO_BLOCKS = (4 * LOCAL_N + HALO_THREADS - 1) // HALO_THREADS


class GPUState:
    def __init__(self, ctx, fields):
        self.ctx = ctx
        self.f = fields
        self.inv_h2 = 1.0 / (fields["h"] ** 2)
        self.dt = fields["dt"]
        self.d_k = cuda.to_device(fields["kappa"])
        self.d_s = cuda.to_device(fields["source"])
        self.d_u = cuda.to_device(fields["u0"])
        self.d_unew = cuda.to_device(fields["u0"])
        self.d_send = cuda.device_array(4 * LOCAL_N, dtype=np.float64)
        self.d_recv = cuda.device_array(4 * LOCAL_N, dtype=np.float64)
        self.h_send = cuda.pinned_array(4 * LOCAL_N, dtype=np.float64)
        self.h_recv = cuda.pinned_array(4 * LOCAL_N, dtype=np.float64)
        self.h_send[:] = 0.0
        self.h_recv[:] = 0.0
        self.stream_interior = cuda.stream()
        self.stream_boundary = cuda.stream()
        c = ctx
        self.flags = (int(c["up"] != MPI.PROC_NULL), int(c["down"] != MPI.PROC_NULL),
                      int(c["left"] != MPI.PROC_NULL), int(c["right"] != MPI.PROC_NULL))

    def reset(self):
        self.d_u.copy_to_device(self.f["u0"])
        self.d_unew.copy_to_device(self.f["u0"])
        cuda.synchronize()

    def swap(self):
        self.d_u, self.d_unew = self.d_unew, self.d_u

    def exchange_host(self):
        """Non-blocking tag-matched MPI exchange over pinned host buffers."""
        c = self.ctx
        cart = c["cart_comm"]
        n = LOCAL_N
        hs, hr = self.h_send, self.h_recv
        reqs = [
            cart.Irecv([hr[0 * n:1 * n], MPI.DOUBLE], source=c["up"], tag=TAG_DOWN),
            cart.Irecv([hr[1 * n:2 * n], MPI.DOUBLE], source=c["down"], tag=TAG_UP),
            cart.Irecv([hr[2 * n:3 * n], MPI.DOUBLE], source=c["left"], tag=TAG_RIGHT),
            cart.Irecv([hr[3 * n:4 * n], MPI.DOUBLE], source=c["right"], tag=TAG_LEFT),
            cart.Isend([hs[0 * n:1 * n], MPI.DOUBLE], dest=c["up"], tag=TAG_UP),
            cart.Isend([hs[1 * n:2 * n], MPI.DOUBLE], dest=c["down"], tag=TAG_DOWN),
            cart.Isend([hs[2 * n:3 * n], MPI.DOUBLE], dest=c["left"], tag=TAG_LEFT),
            cart.Isend([hs[3 * n:4 * n], MPI.DOUBLE], dest=c["right"], tag=TAG_RIGHT),
        ]
        return reqs

    def step_serialized(self):
        sb = self.stream_boundary
        pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](self.d_u, self.d_send)
        self.d_send.copy_to_host(self.h_send, stream=sb)
        sb.synchronize()
        MPI.Request.Waitall(self.exchange_host())
        self.d_recv.copy_to_device(self.h_recv, stream=sb)
        unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](self.d_u, self.d_recv, *self.flags)
        launch_stencil(FULL_REGION, sb, self.d_u, self.d_k, self.d_s, self.d_unew,
                       self.dt, self.inv_h2)
        sb.synchronize()
        self.swap()

    def step_overlapped(self):
        si, sb = self.stream_interior, self.stream_boundary
        # 1) boundary stream: pack strided/contiguous edges + async D2H into pinned memory
        pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](self.d_u, self.d_send)
        self.d_send.copy_to_host(self.h_send, stream=sb)
        # 2) interior stream: deep interior [2..255]^2 needs no ghost data
        launch_stencil(DEEP_REGION, si, self.d_u, self.d_k, self.d_s, self.d_unew,
                       self.dt, self.inv_h2)
        # 3) MPI exchange while the interior kernel runs
        sb.synchronize()
        MPI.Request.Waitall(self.exchange_host())
        # 4) H2D + unpack ghosts + 4 border strips on the boundary stream
        self.d_recv.copy_to_device(self.h_recv, stream=sb)
        unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](self.d_u, self.d_recv, *self.flags)
        for reg in BORDER_REGIONS:
            launch_stencil(reg, sb, self.d_u, self.d_k, self.d_s, self.d_unew,
                           self.dt, self.inv_h2)
        sb.synchronize()
        si.synchronize()
        self.swap()

    def global_energy(self):
        u = self.d_u.copy_to_host()
        h = self.f["h"]
        local = 0.5 * h * h * float(np.sum(u[1:-1, 1:-1] ** 2))
        return self.ctx["cart_comm"].allreduce(local, op=MPI.SUM)


def run_mode(state: GPUState, mode: str, steps: int):
    state.reset()
    ctx = state.ctx
    step_fn = state.step_overlapped if mode == "overlapped" else state.step_serialized
    energies = {}
    compute_ms = 0.0
    if 0 in ENERGY_STEPS:
        energies[0] = state.global_energy()
    ctx["cart_comm"].Barrier()
    for s in range(1, steps + 1):
        t0 = time.perf_counter()
        step_fn()
        compute_ms += (time.perf_counter() - t0) * 1e3
        if s in ENERGY_STEPS:
            energies[s] = state.global_energy()
    cuda.synchronize()
    ctx["cart_comm"].Barrier()
    total_ms = ctx["cart_comm"].allreduce(compute_ms, op=MPI.MAX)
    final = state.d_u.copy_to_host()
    return final, energies, total_ms


# ==============================================================================
# TASK 4: Host false-sharing sweep on the MPI-3 shared window
# ==============================================================================
@intrinsic
def _relaxed_rmw_add(typingctx, arr, idx, inc):
    """In-place RMW through memory: monotonic (relaxed) atomic load + add + monotonic
    atomic store. Not a locked RMW, but LLVM may NOT promote it to a register
    (LICM only promotes simple loads/stores), so every iteration touches the line."""
    if not isinstance(arr, types.Array) or arr.dtype != types.uint64:
        return None
    sig = types.uint64(arr, idx, inc)

    def codegen(context, builder, signature, args):
        aryty, idxty, incty = signature.args
        ary = context.make_array(aryty)(context, builder, args[0])
        index = context.cast(builder, args[1], idxty, types.intp)
        ptr = cgutils.get_item_pointer(context, builder, aryty, ary, [index],
                                       wraparound=False)
        delta = context.cast(builder, args[2], incty, types.uint64)
        old = builder.load_atomic(ptr, "monotonic", 8)
        new = builder.add(old, delta)
        builder.store_atomic(new, ptr, "monotonic", 8)
        return new

    return sig, codegen


@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters):
    one = np.uint64(1)
    last = np.uint64(0)
    for _ in range(iters):
        last = _relaxed_rmw_add(shm_view, slot, one)
    return last


def benchmark_shm_strides(ctx: dict) -> dict:
    node_comm = ctx["node_comm"]
    view = ctx["shm_view"]
    rank = ctx["node_rank"]
    nsz = ctx["node_size"]

    # JIT warm-up on a private slot pattern (not timed)
    host_shm_accumulate(view, rank * 16, 1000)
    node_comm.Barrier()

    results = {}
    for stride in (1, 8, 16):
        if rank == 0:
            view[:] = 0
        node_comm.Barrier()
        slot = rank * stride
        t0 = time.perf_counter()
        host_shm_accumulate(view, slot, SHM_ITERS)
        dt_s = time.perf_counter() - t0
        node_comm.Barrier()
        t_max = node_comm.allreduce(dt_s, op=MPI.MAX)
        t_all = node_comm.gather(dt_s, root=0)
        ok = int(view[slot]) == SHM_ITERS
        all_ok = node_comm.allreduce(int(ok), op=MPI.MIN) == 1
        lines = sorted({(ctx["shm_aligned_addr"] + 8 * r * stride) // 64 -
                        ctx["shm_aligned_addr"] // 64 for r in range(nsz)})
        results[stride] = dict(
            stride_words=stride, stride_bytes=8 * stride,
            max_rank_seconds=t_max,
            per_rank_seconds=t_all if rank == 0 else None,
            ns_per_update=t_max / SHM_ITERS * 1e9,
            distinct_cpu_lines=len(lines), cpu_line_indices=lines,
            counters_correct=all_ok,
        )
        node_comm.Barrier()
    base = results[8]["max_rank_seconds"]
    out = {
        "stride_1_seconds": results[1]["max_rank_seconds"],
        "stride_8_seconds": results[8]["max_rank_seconds"],
        "stride_16_seconds": results[16]["max_rank_seconds"],
        "false_sharing_slowdown_s1_over_s8": results[1]["max_rank_seconds"] / base,
        "s16_over_s8_ratio": results[16]["max_rank_seconds"] / base,
        "details": {str(k): v for k, v in results.items()},
    }
    return out


# ==============================================================================
# Audits
# ==============================================================================
def _as_text(obj):
    if isinstance(obj, dict):
        return "\n".join(str(v) for v in obj.values())
    return str(obj)


def ptx_audit():
    try:
        ptx = _as_text(tiled_stencil_kernel.inspect_asm())
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}
    with open("tiled_stencil_kernel.ptx", "w", encoding="utf-8") as f:
        f.write(ptx)
    bar = ptx.count("bar.sync")
    barrier = ptx.count("barrier.sync")
    return {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": bar + barrier,
        "bar.sync_literal": bar,
        "barrier.sync_literal": barrier,
        "ld.global.f64": ptx.count("ld.global.f64"),
        "st.global.f64": ptx.count("st.global.f64"),
        "ptx_lines": ptx.count("\n"),
    }


def llvm_audit():
    try:
        ir = _as_text(host_shm_accumulate.inspect_llvm())
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}
    with open("host_shm_accumulate.ll", "w", encoding="utf-8") as f:
        f.write(ir)
    try:
        asm = _as_text(host_shm_accumulate.inspect_asm())
        with open("host_shm_accumulate.s", "w", encoding="utf-8") as f:
            f.write(asm)
    except Exception:
        asm = ""
    la = ir.count("load atomic")
    sa = ir.count("store atomic")
    return {
        "load_atomic_monotonic": ir.count("load atomic i64"),
        "store_atomic_monotonic": ir.count("store atomic i64"),
        "load_atomic_total": la,
        "store_atomic_total": sa,
        "monotonic_tokens": ir.count("monotonic"),
        "atomicrmw": ir.count("atomicrmw"),
        "nogil": True,
        "register_hoisting_defeated": bool(la >= 1 and sa >= 1),
        "ir_lines": ir.count("\n"),
    }


# ==============================================================================
# TASK 5: main
# ==============================================================================
def main():
    github_user = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_USERNAME", "")
    if not github_user or github_user.startswith("REPLACE"):
        raise SystemExit("Pass your GitHub username as argv[1].")

    ctx = init_hybrid_context(github_user)
    rank = ctx["rank"]
    cart = ctx["cart_comm"]

    log0(ctx, "=" * 78)
    log0(ctx, f"Hybrid MPI-CUDA & Cache-Line Coherency Lab | {STUDENT_NAME} ({STUDENT_ID})")
    log0(ctx, f"GitHub user      : {ctx['github_user']}")
    log0(ctx, f"GPU              : {get_gpu_name()} | {ctx['gpu_uuid']}")
    log0(ctx, f"CPU signature    : {ctx['cpu_sig']} | coherency line {ctx['line_bytes']} B")
    log0(ctx, f"INSTANCE_SEED    : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    log0(ctx, "=" * 78)

    topo = cart.gather(dict(rank=rank, coords=ctx["coords"], up=ctx["up"], down=ctx["down"],
                            left=ctx["left"], right=ctx["right"]), root=0)
    if rank == 0:
        print("[Task 1] 2x2 Cartesian topology (PROC_NULL = %d):" % MPI.PROC_NULL)
        for t in topo:
            print(f"  rank {t['rank']} coords={t['coords']} up={t['up']} down={t['down']} "
                  f"left={t['left']} right={t['right']}")
        print(f"[Task 1] node_comm size={ctx['node_size']} | shm base=0x{ctx['shm_base_addr']:x} "
              f"aligned=0x{ctx['shm_aligned_addr']:x} (offset {ctx['shm_offset_words']} words, "
              f"addr%64={ctx['shm_aligned_addr'] % 64})", flush=True)

    # ---------------- Task 2/3: stencil -----------------------------------
    fields = initialize_fields(ctx)
    log0(ctx, f"[Task 2] local arrays {fields['u0'].shape} float64 | h = 1/513 = {fields['h']:.10e}")
    log0(ctx, f"[Task 2] kappa_max = {fields['kappa_max']:.12f} | dt_max = h^2/(4 kappa_max) = "
              f"{fields['dt_max']:.6e} | dt = {fields['dt']:.6e} (CFL ratio {fields['dt'] / fields['dt_max']:.3f})")

    state = GPUState(ctx, fields)
    # JIT warm-up (not timed)
    state.step_serialized()
    state.step_overlapped()
    cuda.synchronize()
    cart.Barrier()

    u_ser, e_ser, t_ser = run_mode(state, "serialized", STENCIL_STEPS)
    u_ovl, e_ovl, t_ovl = run_mode(state, "overlapped", STENCIL_STEPS)

    local_diff = float(np.max(np.abs(u_ser[1:-1, 1:-1] - u_ovl[1:-1, 1:-1])))
    max_diff = cart.allreduce(local_diff, op=MPI.MAX)
    per_rank_diff = cart.gather(local_diff, root=0)
    parity_ok = max_diff < 1e-12
    energy_rel = max(abs(e_ser[s] - e_ovl[s]) / max(abs(e_ser[s]), 1e-300) for s in ENERGY_STEPS)
    energy_parity_ok = energy_rel < 1e-12
    finite_ok = all(math.isfinite(e_ovl[s]) for s in ENERGY_STEPS)
    monotone = all(e_ovl[ENERGY_STEPS[k + 1]] <= e_ovl[ENERGY_STEPS[k]]
                   for k in range(len(ENERGY_STEPS) - 1))

    if rank == 0:
        print(f"[Task 3] gpu_serialized_ms = {t_ser:.3f} | gpu_overlapped_ms = {t_ovl:.3f} | "
              f"speedup = {t_ser / t_ovl:.4f}x")
        print(f"[Task 5] per-rank max|u_ser-u_ovl| = {per_rank_diff}")
        print(f"[Task 5] global max_diff = {max_diff:.3e} -> {'PASS' if parity_ok else 'FAIL'} (< 1e-12)")
        for s in ENERGY_STEPS:
            print(f"[Task 5] step {s:3d}: E_serialized = {e_ser[s]:.15e} | E_overlapped = {e_ovl[s]:.15e}")
        print(f"[Task 5] energy parity rel = {energy_rel:.3e} -> {'PASS' if energy_parity_ok else 'FAIL'} | "
              f"monotone dissipation = {monotone}", flush=True)

    # ---------------- Task 4: shared-memory false sharing -----------------
    shm = benchmark_shm_strides(ctx)
    if rank == 0:
        for k in ("1", "8", "16"):
            d = shm["details"][k]
            print(f"[Task 4] stride={d['stride_words']:2d} ({d['stride_bytes']:3d} B): "
                  f"{d['max_rank_seconds'] * 1e3:10.3f} ms | {d['ns_per_update']:7.3f} ns/update | "
                  f"CPU lines used={d['distinct_cpu_lines']} {d['cpu_line_indices']} | "
                  f"per-rank s={['%.4f' % x for x in d['per_rank_seconds']]} | "
                  f"counters ok={d['counters_correct']}")
        print(f"[Task 4] false-sharing slowdown stride1/stride8 = {shm['false_sharing_slowdown_s1_over_s8']:.3f}x | "
              f"stride16/stride8 = {shm['s16_over_s8_ratio']:.3f}x", flush=True)

    # ---------------- Audits ----------------------------------------------
    if rank == 0:
        ptx = ptx_audit()
        llvm = llvm_audit()
        print(f"[Audit] PTX tiled_stencil_kernel: {json.dumps(ptx)}")
        print(f"[Audit] LLVM host_shm_accumulate: {json.dumps(llvm)}", flush=True)

        manifest = {
            "student": {"name": STUDENT_NAME, "id": STUDENT_ID, "github_username": ctx["github_user"]},
            "instructor": "Sufyan bin Uzayr",
            "environment": {
                "gpu_name": get_gpu_name(), "gpu_uuid": ctx["gpu_uuid"],
                "cpu_signature": ctx["cpu_sig"], "cpu_coherency_line_bytes": ctx["line_bytes"],
                "python": platform.python_version(), "mpi_vendor": list(map(str, MPI.get_vendor())),
                "numba": __import__("numba").__version__, "numpy": np.__version__,
            },
            "instance_seed": ctx["instance_seed"],
            "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
            "topology": topo,
            "grid": {"global": [GLOBAL_NY, GLOBAL_NX], "local_interior": [LOCAL_N, LOCAL_N],
                     "local_padded": [PAD_N, PAD_N], "tile": [TILE_DIM, TILE_DIM],
                     "shared_tile": [SHARED_DIM, SHARED_DIM]},
            "physics": {"dx": fields["h"], "kappa_max": fields["kappa_max"],
                        "dt": fields["dt"], "dt_max": fields["dt_max"],
                        "cfl_ratio": fields["dt"] / fields["dt_max"],
                        "cfl_satisfied": fields["dt"] <= fields["dt_max"],
                        "lambda_damp": LAMBDA_DAMP, "steps": STENCIL_STEPS,
                        "init_params": fields["init_params"]},
            "timing": {"gpu_serialized_ms": t_ser, "gpu_overlapped_ms": t_ovl,
                       "overlap_speedup": t_ser / t_ovl},
            "verification": {"max_diff": max_diff, "per_rank_max_diff": per_rank_diff,
                             "parity_pass": parity_ok,
                             "energy_serialized": {str(k): v for k, v in e_ser.items()},
                             "energy_overlapped": {str(k): v for k, v in e_ovl.items()},
                             "energy_parity_rel": energy_rel,
                             "energy_parity_pass": energy_parity_ok,
                             "energy_finite": finite_ok,
                             "energy_monotone_dissipation": monotone},
            "shm_window": {"slots": SHM_SLOTS, "aligned_addr_mod_64": ctx["shm_aligned_addr"] % 64,
                           "iters_per_rank": SHM_ITERS, **shm},
            "ptx_audit": ptx,
            "llvm_audit": llvm,
        }
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        print("[Task 5] wrote hybrid_lab_manifest.json")
        status = parity_ok and energy_parity_ok and finite_ok
        print(f"[Task 5] OVERALL: {'PASS' if status else 'FAIL'}", flush=True)

    cart.Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    main()
