#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distributed_cuda_solver.py
==========================
4-rank distributed 2D anisotropic (heterogeneous-kappa) reaction-diffusion solver
built with mpi4py (distributed memory) + numba.cuda (SIMT GPU kernels).

Student : Ainat Yermurat (ID 230103161)
GitHub  : yermy-bc
Course  : MPI Implementation in CUDA (Python) - Instructor: Sufyan bin Uzayr

Implemented tasks
-----------------
Task 1  init_distributed_context   - 2x2 non-periodic Cartesian topology, GPU binding,
                                     SHA-256 hardware-bound INSTANCE_SEED
Task 2  initialize_subdomains      - rank-local (258,258) kappa / u0 / S fields,
                                     seeded Gaussian pulses, Dirichlet on PROC_NULL edges
Task 3  pack_halos_kernel / unpack_halos_kernel / exchange_halos_async
                                   - GPU strided packing, pinned host staging,
                                     tag-matched Irecv/Isend on the Cartesian communicator
Task 4  tiled_stencil_kernel / step_overlapped
                                   - 16x16 blocks, 18x18 shared tile + syncthreads,
                                     dual-stream compute/communication overlap
Task 5  run_solver                 - 400 steps serialized vs overlapped, Allreduce
                                     energy / L2 diagnostics, parity, PTX audit, manifest

Launch:  mpirun -np 4 python distributed_cuda_solver.py
(GITHUB_USERNAME must be exported in the environment.)
"""

import os

# Silence numba low-occupancy warnings (thin boundary-ring launches are intentional).
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")

import sys
import re
import json
import math
import time
import hashlib
import platform
import datetime
import subprocess
import warnings

import numpy as np
from mpi4py import MPI
import numba
from numba import cuda

warnings.filterwarnings("ignore", message=".*[Uu]nder-utilization.*")
warnings.filterwarnings("ignore", message=".*deprecat.*")
try:
    from numba.core.errors import NumbaPerformanceWarning, NumbaDeprecationWarning
    warnings.simplefilter("ignore", NumbaPerformanceWarning)
    warnings.simplefilter("ignore", NumbaDeprecationWarning)
except Exception:  # pragma: no cover
    pass

# ----------------------------------------------------------------------------
# Global problem constants
# ----------------------------------------------------------------------------
STUDENT_NAME = os.environ.get("STUDENT_NAME", "Ainat Yermurat")
STUDENT_ID = os.environ.get("STUDENT_ID", "230103161")

NY_GLOBAL, NX_GLOBAL = 512, 512          # interior global grid (no physical boundary)
PROC_DIMS = [2, 2]                        # 2 x 2 Cartesian process grid
PERIODS = [False, False]                  # non-periodic
NY, NX = NY_GLOBAL // PROC_DIMS[0], NX_GLOBAL // PROC_DIMS[1]   # 256 x 256 owned
LOCAL_SHAPE = (NY + 2, NX + 2)            # 258 x 258 incl. 1-cell ghost apron

DX = 1.0 / (NX_GLOBAL + 1)                # 1/513 (boundaries sit at x=0 and x=1)
DY = 1.0 / (NY_GLOBAL + 1)                # 1/513
LAMBDA = 0.15                             # cubic damping coefficient
KAPPA_LO, KAPPA_HI = 0.08, 0.24           # diffusivity range
CFL_SAFETY = 0.80                         # dt = 0.8 * dt_max
N_STEPS = 400
DIAG_EVERY = 100
TILE = 16                                 # 16 x 16 threadblock
TILE_HALO = 18                            # 18 x 18 shared tile (TILE + 2), literal for numba-cuda
PACK_THREADS = 128
PARITY_TOL = 1.0e-12
assert TILE_HALO == TILE + 2

# Message tags: tag names the direction the message TRAVELS.
TAG_NORTHWARD = 101   # my top owned row  -> neighbour above  (arrives in its bottom ghost)
TAG_SOUTHWARD = 102   # my bottom owned row -> neighbour below (arrives in its top ghost)
TAG_WESTWARD = 103    # my left owned col  -> neighbour left   (arrives in its right ghost)
TAG_EASTWARD = 104    # my right owned col -> neighbour right  (arrives in its left ghost)

# Layout of the packed 1-D halo vectors (length 2*NX + 2*NY):
#   [0      : NX)          top    strip  (send: row 1   | recv: ghost row 0)
#   [NX     : 2NX)         bottom strip  (send: row NY  | recv: ghost row NY+1)
#   [2NX    : 2NX+NY)      left   strip  (send: col 1   | recv: ghost col 0)
#   [2NX+NY : 2NX+2NY)     right  strip  (send: col NX  | recv: ghost col NX+1)
OFF_TOP, OFF_BOT, OFF_LEFT, OFF_RIGHT = 0, NX, 2 * NX, 2 * NX + NY
HALO_LEN = 2 * NX + 2 * NY


class Ctx:
    """Plain container for the distributed context."""
    pass


def root_print(ctx, *args):
    if ctx.rank == 0:
        print(*args, flush=True)


def ordered_print(ctx, msg):
    for r in range(ctx.size):
        if r == ctx.rank:
            print(msg, flush=True)
        ctx.cart.Barrier()


# ============================================================================
# TASK 1 - Distributed topology & hardware-bound seed
# ============================================================================
def _query_gpu_uuid(dev):
    """Return the active GPU UUID (e.g. 'GPU-1a2b...'). Never returns a placeholder."""
    uuid = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=30,
        )
        cand = out.stdout.strip().splitlines()
        if out.returncode == 0 and cand and cand[0].strip().startswith("GPU-"):
            uuid = cand[0].strip()
    except Exception:
        uuid = None
    if uuid is None:
        try:
            u = getattr(dev, "uuid", None)
            if u:
                u = u.decode() if isinstance(u, bytes) else str(u)
                uuid = u if u.startswith("GPU-") else "GPU-" + u
        except Exception:
            uuid = None
    if not uuid:
        raise RuntimeError("Could not determine the active GPU UUID - refusing to use a placeholder.")
    return uuid


def init_distributed_context(github_username):
    if (not github_username or not github_username.strip()
            or github_username.strip() == "REPLACE_WITH_YOUR_GITHUB_USERNAME"):
        raise RuntimeError("GITHUB_USERNAME is not set to a real GitHub handle.")
    github_username = github_username.strip()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size == 4, f"This solver requires exactly 4 MPI ranks, got {size}"

    cart = comm.Create_cart(dims=PROC_DIMS, periods=PERIODS, reorder=False)
    rank = cart.Get_rank()
    coords = cart.Get_coords(rank)
    nbr_up, nbr_down = cart.Shift(0, 1)       # axis 0: rows   (up = row_rank-1)
    nbr_left, nbr_right = cart.Shift(1, 1)    # axis 1: columns (left = col_rank-1)

    # Bind every rank to the single physical GPU before any device allocation.
    cuda.select_device(0)
    dev = cuda.get_current_device()
    gpu_name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    cc = tuple(dev.compute_capability)
    gpu_uuid = _query_gpu_uuid(dev)
    dev_attrs = {}
    for a in ("MULTIPROCESSOR_COUNT", "ASYNC_ENGINE_COUNT", "CONCURRENT_KERNELS",
              "MAX_SHARED_MEMORY_PER_BLOCK", "COMPUTE_MODE"):
        try:
            dev_attrs[a] = int(getattr(dev, a))
        except Exception:
            dev_attrs[a] = None

    # INSTANCE_SEED = first 64 bits of SHA-256(github_username || gpu_uuid)
    seed_material = f"{github_username}{gpu_uuid}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)      # 64-bit unsigned integer

    # All ranks must agree on the seed (same user, same GPU).
    s_min = cart.allreduce(instance_seed, op=MPI.MIN)
    s_max = cart.allreduce(instance_seed, op=MPI.MAX)
    assert s_min == s_max == instance_seed, "INSTANCE_SEED mismatch across ranks"

    ctx = Ctx()
    ctx.comm, ctx.cart, ctx.rank, ctx.size = comm, cart, rank, size
    ctx.coords = (int(coords[0]), int(coords[1]))
    ctx.nbr_up, ctx.nbr_down = nbr_up, nbr_down
    ctx.nbr_left, ctx.nbr_right = nbr_left, nbr_right
    ctx.github_username = github_username
    ctx.gpu_name, ctx.gpu_uuid, ctx.cc = gpu_name, gpu_uuid, cc
    ctx.dev_attrs = dev_attrs
    ctx.seed_digest = digest
    ctx.instance_seed = instance_seed
    ctx.hostname = MPI.Get_processor_name()
    return ctx


# ============================================================================
# TASK 2 - Seeded heterogeneous subdomain initialisation (rank-local only)
# ============================================================================
def initialize_subdomains(ctx):
    row_rank, col_rank = ctx.coords
    # Local index l in 0..257 -> global node index g = rank_offset + l in 0..513
    gi = row_rank * NY + np.arange(NY + 2, dtype=np.int64)   # rows  -> y
    gj = col_rank * NX + np.arange(NX + 2, dtype=np.int64)   # cols  -> x
    y = gi * DY
    x = gj * DX
    X, Y = np.meshgrid(x, y)          # both (258, 258); X varies along axis 1

    # Identical RNG stream on every rank -> globally consistent fields.
    rng = np.random.default_rng(ctx.instance_seed)

    # Diffusivity kappa(x,y) in [0.08, 0.24]
    ph_x, ph_y = rng.uniform(0.0, 2.0 * np.pi, size=2)
    kmid = 0.5 * (KAPPA_LO + KAPPA_HI)       # 0.16
    kamp = 0.5 * (KAPPA_HI - KAPPA_LO)       # 0.08
    kappa = kmid + kamp * np.sin(3.0 * np.pi * X + ph_x) * np.cos(2.0 * np.pi * Y + ph_y)

    # Initial Gaussian thermal pulses (centres / amplitudes / widths from seed)
    n_pulses = int(rng.integers(3, 5))       # 3 or 4 pulses
    pulses = []
    u0 = np.zeros(LOCAL_SHAPE, dtype=np.float64)
    for _ in range(n_pulses):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        amp = rng.uniform(0.6, 1.0)
        sig = rng.uniform(0.05, 0.09)
        u0 += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig))
        pulses.append({"cx": float(cx), "cy": float(cy), "amp": float(amp), "sigma": float(sig)})

    # Localised (weak) source S(x,y)
    sx, sy = rng.uniform(0.3, 0.7, size=2)
    s_amp, s_sig = 0.05, 0.03
    src = s_amp * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2.0 * s_sig * s_sig))

    # Homogeneous Dirichlet u = 0 on outer global edges (neighbour == PROC_NULL)
    if ctx.nbr_up == MPI.PROC_NULL:
        u0[0, :] = 0.0
    if ctx.nbr_down == MPI.PROC_NULL:
        u0[-1, :] = 0.0
    if ctx.nbr_left == MPI.PROC_NULL:
        u0[:, 0] = 0.0
    if ctx.nbr_right == MPI.PROC_NULL:
        u0[:, -1] = 0.0

    meta = {
        "kappa_phase": [float(ph_x), float(ph_y)],
        "pulses": pulses,
        "source": {"cx": float(sx), "cy": float(sy), "amp": s_amp, "sigma": s_sig},
        "local_x_range": [float(x[0]), float(x[-1])],
        "local_y_range": [float(y[0]), float(y[-1])],
    }
    return (np.ascontiguousarray(u0), np.ascontiguousarray(kappa),
            np.ascontiguousarray(src), meta)


# ============================================================================
# TASK 3 - GPU halo pack / unpack kernels and pinned asynchronous exchange
# ============================================================================
@cuda.jit
def pack_halos_kernel(u, send, ny, nx):
    """Collect the 4 owned border strips (strided columns!) into one contiguous vector."""
    k = cuda.grid(1)
    if k < nx:
        send[k] = u[1, k + 1]                       # top owned row    (row 1)
        send[nx + k] = u[ny, k + 1]                 # bottom owned row (row ny)
    if k < ny:
        send[2 * nx + k] = u[k + 1, 1]              # left owned col   (col 1)  - stride 258
        send[2 * nx + ny + k] = u[k + 1, nx]        # right owned col  (col nx) - stride 258


@cuda.jit
def unpack_halos_kernel(u, recv, ny, nx, has_up, has_down, has_left, has_right):
    """Scatter received contiguous vectors into ghost rows/cols 0 and n+1."""
    k = cuda.grid(1)
    if k < nx:
        if has_up != 0:
            u[0, k + 1] = recv[k]                   # ghost row 0
        if has_down != 0:
            u[ny + 1, k + 1] = recv[nx + k]         # ghost row ny+1
    if k < ny:
        if has_left != 0:
            u[k + 1, 0] = recv[2 * nx + k]          # ghost col 0
        if has_right != 0:
            u[k + 1, nx + 1] = recv[2 * nx + ny + k]  # ghost col nx+1


def _pack_grid():
    return (max(NX, NY) + PACK_THREADS - 1) // PACK_THREADS


def start_halo_exchange(ctx, u_d, st):
    """Stage A + B(MPI part): pack on GPU, async D2H to pinned, post Irecv/Isend."""
    b = ctx.bufs
    pack_halos_kernel[_pack_grid(), PACK_THREADS, st](u_d, b["d_send"], NY, NX)
    b["d_send"].copy_to_host(b["h_send"], stream=st)     # async DMA -> page-locked
    st.synchronize()                                       # host send buffers ready

    cart, hs, hr = ctx.cart, b["h_send"], b["h_recv"]
    reqs = [
        # receives first (pre-posted), tag-matched to the sender's direction
        cart.Irecv([hr[OFF_TOP:OFF_TOP + NX], MPI.DOUBLE], source=ctx.nbr_up, tag=TAG_SOUTHWARD),
        cart.Irecv([hr[OFF_BOT:OFF_BOT + NX], MPI.DOUBLE], source=ctx.nbr_down, tag=TAG_NORTHWARD),
        cart.Irecv([hr[OFF_LEFT:OFF_LEFT + NY], MPI.DOUBLE], source=ctx.nbr_left, tag=TAG_EASTWARD),
        cart.Irecv([hr[OFF_RIGHT:OFF_RIGHT + NY], MPI.DOUBLE], source=ctx.nbr_right, tag=TAG_WESTWARD),
        cart.Isend([hs[OFF_TOP:OFF_TOP + NX], MPI.DOUBLE], dest=ctx.nbr_up, tag=TAG_NORTHWARD),
        cart.Isend([hs[OFF_BOT:OFF_BOT + NX], MPI.DOUBLE], dest=ctx.nbr_down, tag=TAG_SOUTHWARD),
        cart.Isend([hs[OFF_LEFT:OFF_LEFT + NY], MPI.DOUBLE], dest=ctx.nbr_left, tag=TAG_WESTWARD),
        cart.Isend([hs[OFF_RIGHT:OFF_RIGHT + NY], MPI.DOUBLE], dest=ctx.nbr_right, tag=TAG_EASTWARD),
    ]
    return reqs


def finish_halo_exchange(ctx, u_d, reqs, st):
    """Stage C(MPI part): Waitall, async H2D from pinned, unpack into ghost cells."""
    b = ctx.bufs
    MPI.Request.Waitall(reqs)
    b["d_recv"].copy_to_device(b["h_recv"], stream=st)   # async DMA <- page-locked
    unpack_halos_kernel[_pack_grid(), PACK_THREADS, st](
        u_d, b["d_recv"], NY, NX,
        ctx.has_up, ctx.has_down, ctx.has_left, ctx.has_right)


def exchange_halos_async(ctx, u_d, st):
    """Complete pinned, stream-ordered halo exchange (used by the serialized path)."""
    reqs = start_halo_exchange(ctx, u_d, st)
    finish_halo_exchange(ctx, u_d, reqs, st)


# ============================================================================
# TASK 4 - Shared-memory tiled stencil kernel and dual-stream orchestrator
# ============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kap, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    """Update cells (i,j) with r_start<=i<=r_end, c_start<=j<=c_end.

    16x16 threads, each block stages an 18x18 tile of u (and of kappa) incl. a 1-cell apron.
    Only cells inside [r_start-1, r_end+1] x [c_start-1, c_end+1] are ever read, so the
    deep-interior launch never touches ghost cells that stream_boundary is filling.
    """
    su = cuda.shared.array(shape=(TILE_HALO, TILE_HALO), dtype=np.float64)
    sk = cuda.shared.array(shape=(TILE_HALO, TILE_HALO), dtype=np.float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE + tx      # column (x direction)
    i = r_start + cuda.blockIdx.y * TILE + ty      # row    (y direction)
    sx = tx + 1
    sy = ty + 1
    r_lim = r_end + 1
    c_lim = c_end + 1

    if i <= r_lim and j <= c_lim:
        su[sy, sx] = u[i, j]                        # centre cell
        sk[sy, sx] = kap[i, j]
        if ty == 0:                                 # top apron row
            su[0, sx] = u[i - 1, j]
            sk[0, sx] = kap[i - 1, j]
        if ty == TILE - 1 and i + 1 <= r_lim:       # bottom apron row
            su[TILE + 1, sx] = u[i + 1, j]
            sk[TILE + 1, sx] = kap[i + 1, j]
        if tx == 0:                                 # left apron column
            su[sy, 0] = u[i, j - 1]
            sk[sy, 0] = kap[i, j - 1]
        if tx == TILE - 1 and j + 1 <= c_lim:       # right apron column
            su[sy, TILE + 1] = u[i, j + 1]
            sk[sy, TILE + 1] = kap[i, j + 1]

    cuda.syncthreads()                              # tile complete before any stencil math

    if i <= r_end and j <= c_end:
        uc = su[sy, sx]
        kc = sk[sy, sx]
        k_e = 0.5 * (sk[sy, sx + 1] + kc)           # kappa_{i,j+1/2}
        k_w = 0.5 * (sk[sy, sx - 1] + kc)           # kappa_{i,j-1/2}
        k_s = 0.5 * (sk[sy + 1, sx] + kc)           # kappa_{i+1/2,j}
        k_n = 0.5 * (sk[sy - 1, sx] + kc)           # kappa_{i-1/2,j}
        flux_x = (k_e * (su[sy, sx + 1] - uc) - k_w * (uc - su[sy, sx - 1])) * inv_dx2
        flux_y = (k_s * (su[sy + 1, sx] - uc) - k_n * (uc - su[sy - 1, sx])) * inv_dy2
        u_new[i, j] = uc + dt * (flux_x + flux_y - lam * uc * uc * uc + src[i, j])


BOX_FULL = (1, NY, 1, NX)
BOX_DEEP = (2, NY - 1, 2, NX - 1)                   # [2..255, 2..255]
BOX_RING = [
    (1, 1, 1, NX),                                  # row 1
    (NY, NY, 1, NX),                                # row ny
    (2, NY - 1, 1, 1),                              # col 1   (corners already covered)
    (2, NY - 1, NX, NX),                            # col nx
]


def launch_stencil(ctx, st, u_d, un_d, box):
    r0, r1, c0, c1 = box
    rows, cols = r1 - r0 + 1, c1 - c0 + 1
    grid = ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE)
    tiled_stencil_kernel[grid, (TILE, TILE), st](
        u_d, un_d, ctx.kappa_d, ctx.src_d, r0, r1, c0, c1,
        ctx.dt, ctx.inv_dx2, ctx.inv_dy2, LAMBDA)


def step_serialized(ctx):
    """Reference path: exchange -> full stencil, everything on one stream, no overlap."""
    u_d, un_d = ctx.fields[ctx.cur], ctx.fields[1 - ctx.cur]
    sb = ctx.stream_boundary
    exchange_halos_async(ctx, u_d, sb)
    launch_stencil(ctx, sb, u_d, un_d, BOX_FULL)
    sb.synchronize()
    ctx.cur = 1 - ctx.cur


def step_overlapped(ctx):
    """Dual-stream pipeline (Stages A/B/C)."""
    u_d, un_d = ctx.fields[ctx.cur], ctx.fields[1 - ctx.cur]
    sb, si = ctx.stream_boundary, ctx.stream_interior
    # Stage A + B: pack, D2H, sync, post Irecv/Isend
    reqs = start_halo_exchange(ctx, u_d, sb)
    # Stage B: deep interior compute runs concurrently with the MPI exchange
    launch_stencil(ctx, si, u_d, un_d, BOX_DEEP)
    # Stage C: Waitall, H2D, unpack, then the 4 perimeter rings on stream_boundary
    finish_halo_exchange(ctx, u_d, reqs, sb)
    for box in BOX_RING:
        launch_stencil(ctx, sb, u_d, un_d, box)
    sb.synchronize()
    si.synchronize()
    ctx.cur = 1 - ctx.cur


# ============================================================================
# TASK 5 - Reductions, PTX audit, benchmark, manifest
# ============================================================================
def global_sum(ctx, value):
    out = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(np.array([value], dtype=np.float64), out, op=MPI.SUM)
    return float(out[0])


def global_max(ctx, value):
    out = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(np.array([value], dtype=np.float64), out, op=MPI.MAX)
    return float(out[0])


def global_min(ctx, value):
    out = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(np.array([value], dtype=np.float64), out, op=MPI.MIN)
    return float(out[0])


def diagnostics(ctx):
    """Global energy E = sum u dx dy and L2 norm of the last increment ||u^n - u^{n-1}||_2."""
    cur = ctx.fields[ctx.cur].copy_to_host()
    prev = ctx.fields[1 - ctx.cur].copy_to_host()   # holds u^{n-1} after the swap
    own = cur[1:NY + 1, 1:NX + 1]
    own_prev = prev[1:NY + 1, 1:NX + 1]
    e_loc = float(np.sum(own) * DX * DY)
    r_loc = float(np.sum((own - own_prev) ** 2))
    energy = global_sum(ctx, e_loc)
    resid = math.sqrt(global_sum(ctx, r_loc))
    umax = global_max(ctx, float(np.max(own)))
    return energy, resid, umax


def reset_fields(ctx):
    ctx.fields[0].copy_to_device(ctx.u0_host)
    ctx.fields[1].copy_to_device(ctx.u0_host)
    ctx.bufs["h_recv"][:] = 0.0
    ctx.cur = 0
    cuda.synchronize()


def run_mode(ctx, mode):
    step = step_serialized if mode == "serialized" else step_overlapped
    reset_fields(ctx)
    trace = []
    e0, _, umax0 = diagnostics(ctx)
    trace.append({"step": 0, "energy": e0, "l2_residual": 0.0, "u_max": umax0})
    root_print(ctx, f"  [{mode:>10}] step {0:4d} | E = {e0:.15e} | L2 = {0.0:.6e} | max u = {umax0:.6f}")

    ev_ms_total, wall_s_total = 0.0, 0.0
    n_chunks = N_STEPS // DIAG_EVERY
    for c in range(n_chunks):
        cuda.synchronize()
        ctx.cart.Barrier()
        ev_start = cuda.event(timing=True)
        ev_stop = cuda.event(timing=True)
        ev_start.record()
        t0 = time.perf_counter()
        for _ in range(DIAG_EVERY):
            step(ctx)
        ev_stop.record()
        ev_stop.synchronize()
        t1 = time.perf_counter()
        ev_ms_total += cuda.event_elapsed_time(ev_start, ev_stop)
        wall_s_total += (t1 - t0)

        s = (c + 1) * DIAG_EVERY
        e, r, um = diagnostics(ctx)
        trace.append({"step": s, "energy": e, "l2_residual": r, "u_max": um})
        root_print(ctx, f"  [{mode:>10}] step {s:4d} | E = {e:.15e} | L2 = {r:.6e} | max u = {um:.6f}")

    final = ctx.fields[ctx.cur].copy_to_host()
    ev_ms = global_max(ctx, ev_ms_total)            # slowest rank defines time-to-solution
    wall_ms = global_max(ctx, wall_s_total * 1e3)
    return final, trace, ev_ms, wall_ms


def ptx_audit(kernel):
    asm = kernel.inspect_asm()
    if isinstance(asm, dict):
        ptx = "\n".join(str(v) for v in asm.values())
    else:
        ptx = str(asm)
    counts = {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "ld.shared.v2.f64": ptx.count("ld.shared.v2.f64"),
        "st.shared (any)": len(re.findall(r"\bst\.shared\.", ptx)),
        "ld.shared (any)": len(re.findall(r"\bld\.shared\.", ptx)),
        "bar.sync": ptx.count("bar.sync"),
        "barrier.sync": ptx.count("barrier.sync"),
        "ld.global.f64 (any)": len(re.findall(r"\bld\.global(\.nc)?\.f64", ptx)),
        "st.global.f64": ptx.count("st.global.f64"),
        "fma.rn.f64": ptx.count("fma.rn.f64"),
        "ptx_lines": ptx.count("\n") + 1,
    }
    passed = (counts["st.shared (any)"] > 0 and counts["ld.shared (any)"] > 0
              and (counts["bar.sync"] + counts["barrier.sync"]) > 0)
    return counts, passed, ptx


def _file_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def run_solver():
    github_username = os.environ.get("GITHUB_USERNAME", "")
    ctx = init_distributed_context(github_username)

    root_print(ctx, "=" * 78)
    root_print(ctx, " Distributed MPI + Numba-CUDA anisotropic reaction-diffusion solver")
    root_print(ctx, f" Student: {STUDENT_NAME} | ID: {STUDENT_ID} | GitHub: {ctx.github_username}")
    root_print(ctx, "=" * 78)
    root_print(ctx, f"[Task 1] MPI ranks = {ctx.size} | Cartesian dims = {PROC_DIMS} | periods = {PERIODS}")
    root_print(ctx, f"[Task 1] MPI library: {MPI.Get_library_version().splitlines()[0].strip()}")
    root_print(ctx, f"[Task 1] mpi4py {sys.modules['mpi4py'].__version__} | numba {numba.__version__} | numpy {np.__version__}")
    root_print(ctx, f"[Task 1] GPU: {ctx.gpu_name} | CC {ctx.cc[0]}.{ctx.cc[1]} | UUID {ctx.gpu_uuid}")
    root_print(ctx, f"[Task 1] GPU attributes: {ctx.dev_attrs}")
    root_print(ctx, f"[Task 1] SHA-256('{ctx.github_username}' + GPU UUID) = {ctx.seed_digest}")
    root_print(ctx, f"[Task 1] INSTANCE_SEED (64-bit) = {ctx.instance_seed}  (0x{ctx.instance_seed:016x})")

    def nb(v):
        return "PROC_NULL" if v == MPI.PROC_NULL else str(v)
    ordered_print(ctx, (f"[Task 1] rank {ctx.rank} coords={ctx.coords} host={ctx.hostname} "
                        f"up={nb(ctx.nbr_up)} down={nb(ctx.nbr_down)} "
                        f"left={nb(ctx.nbr_left)} right={nb(ctx.nbr_right)} | cuda device 0"))

    ctx.has_up = int(ctx.nbr_up != MPI.PROC_NULL)
    ctx.has_down = int(ctx.nbr_down != MPI.PROC_NULL)
    ctx.has_left = int(ctx.nbr_left != MPI.PROC_NULL)
    ctx.has_right = int(ctx.nbr_right != MPI.PROC_NULL)

    # ---------------- Task 2 ----------------
    u0, kappa, src, init_meta = initialize_subdomains(ctx)
    assert u0.shape == LOCAL_SHAPE and kappa.shape == LOCAL_SHAPE and src.shape == LOCAL_SHAPE
    ctx.u0_host = u0
    k_max = global_max(ctx, float(kappa.max()))
    k_min = global_min(ctx, float(kappa.min()))
    dt_max = DX * DX / (4.0 * k_max)
    ctx.dt = CFL_SAFETY * dt_max
    ctx.inv_dx2 = 1.0 / (DX * DX)
    ctx.inv_dy2 = 1.0 / (DY * DY)
    root_print(ctx, f"[Task 2] local padded array shape = {LOCAL_SHAPE} (owned {NY}x{NX}) | global {NY_GLOBAL}x{NX_GLOBAL}")
    root_print(ctx, f"[Task 2] kappa range (global Allreduce) = [{k_min:.6f}, {k_max:.6f}]")
    root_print(ctx, f"[Task 2] dx = dy = 1/513 = {DX:.10e} | dt_max = dx^2/(4 kappa_max) = {dt_max:.10e}")
    root_print(ctx, f"[Task 2] dt = {CFL_SAFETY} * dt_max = {ctx.dt:.10e} | dt/dt_max = {ctx.dt / dt_max:.3f}")
    root_print(ctx, f"[Task 2] {len(init_meta['pulses'])} seeded Gaussian pulses:")
    for p in init_meta["pulses"]:
        root_print(ctx, f"           centre=({p['cx']:.4f},{p['cy']:.4f}) amp={p['amp']:.4f} sigma={p['sigma']:.4f}")
    s = init_meta["source"]
    root_print(ctx, f"[Task 2] source S: centre=({s['cx']:.4f},{s['cy']:.4f}) amp={s['amp']} sigma={s['sigma']}")
    ordered_print(ctx, (f"[Task 2] rank {ctx.rank} x in [{init_meta['local_x_range'][0]:.5f},"
                        f"{init_meta['local_x_range'][1]:.5f}] y in [{init_meta['local_y_range'][0]:.5f},"
                        f"{init_meta['local_y_range'][1]:.5f}] | local sum(u0) = {u0[1:-1, 1:-1].sum():.6f}"))

    # ---------------- device allocation (rank-local only) ----------------
    ctx.fields = [cuda.to_device(u0), cuda.to_device(u0)]
    ctx.kappa_d = cuda.to_device(kappa)
    ctx.src_d = cuda.to_device(src)
    ctx.stream_boundary = cuda.stream()
    ctx.stream_interior = cuda.stream()
    ctx.bufs = {
        "d_send": cuda.device_array(HALO_LEN, dtype=np.float64),
        "d_recv": cuda.device_array(HALO_LEN, dtype=np.float64),
        "h_send": cuda.pinned_array(HALO_LEN, dtype=np.float64),
        "h_recv": cuda.pinned_array(HALO_LEN, dtype=np.float64),
    }
    ctx.bufs["h_send"][:] = 0.0
    ctx.bufs["h_recv"][:] = 0.0
    ctx.cur = 0
    root_print(ctx, f"[Task 3] pinned host halo buffers: 2 x {HALO_LEN} float64 = {2 * HALO_LEN * 8} bytes per rank")
    root_print(ctx, f"[Task 3] halo payload per step per rank: {HALO_LEN * 8} B D2H + {HALO_LEN * 8} B H2D")

    # ---------------- JIT warm-up (excluded from timing) ----------------
    t_jit = time.perf_counter()
    for _ in range(2):
        step_serialized(ctx)
    for _ in range(2):
        step_overlapped(ctx)
    cuda.synchronize()
    ctx.cart.Barrier()
    root_print(ctx, f"[Task 4] JIT warm-up done in {time.perf_counter() - t_jit:.2f} s "
                    f"(tile {TILE}x{TILE} threads, shared tile {TILE + 2}x{TILE + 2} float64)")

    # ---------------- Task 5: benchmark both modes ----------------
    root_print(ctx, f"[Task 5] Running {N_STEPS} steps - SERIALIZED (single stream, no overlap)")
    fin_ser, trace_ser, ser_ms, ser_wall = run_mode(ctx, "serialized")
    root_print(ctx, f"[Task 5] Running {N_STEPS} steps - OVERLAPPED (stream_boundary + stream_interior)")
    fin_ovl, trace_ovl, ovl_ms, ovl_wall = run_mode(ctx, "overlapped")

    local_diff = float(np.max(np.abs(fin_ser[1:-1, 1:-1] - fin_ovl[1:-1, 1:-1])))
    parity_max_diff = global_max(ctx, local_diff)
    parity_ok = parity_max_diff < PARITY_TOL
    energies = [t["energy"] for t in trace_ovl]
    monotonic = all(energies[k + 1] <= energies[k] for k in range(len(energies) - 1))
    speedup = ser_ms / ovl_ms if ovl_ms > 0 else float("nan")

    root_print(ctx, "-" * 78)
    root_print(ctx, f"[Task 5] serialized_ms (CUDA events, max over ranks) = {ser_ms:.3f} ms "
                    f"({ser_ms / N_STEPS:.4f} ms/step) | wall = {ser_wall:.3f} ms")
    root_print(ctx, f"[Task 5] overlapped_ms (CUDA events, max over ranks) = {ovl_ms:.3f} ms "
                    f"({ovl_ms / N_STEPS:.4f} ms/step) | wall = {ovl_wall:.3f} ms")
    root_print(ctx, f"[Task 5] overlap speedup serialized/overlapped = {speedup:.3f}x")
    root_print(ctx, f"[Task 5] parity max |u_ser - u_ovl| = {parity_max_diff:.3e} "
                    f"(< {PARITY_TOL:.0e}: {parity_ok})")
    root_print(ctx, f"[Task 5] energy monotonically non-increasing over steps 0..{N_STEPS}: {monotonic}")
    for k in range(1, len(energies)):
        root_print(ctx, f"           dE[{trace_ovl[k - 1]['step']:3d}->{trace_ovl[k]['step']:3d}] = "
                        f"{energies[k] - energies[k - 1]:+.6e}")

    # ---------------- PTX audit ----------------
    counts, ptx_ok, ptx_text = ptx_audit(tiled_stencil_kernel)
    root_print(ctx, "[Task 5] PTX audit of tiled_stencil_kernel.inspect_asm():")
    for k, v in counts.items():
        root_print(ctx, f"           {k:<22s} : {v}")
    root_print(ctx, f"[Task 5] ptx_audit_passed = {ptx_ok}")

    # ---------------- manifest (rank 0) ----------------
    if ctx.rank == 0:
        with open("tiled_stencil_kernel.ptx", "w") as f:
            f.write(ptx_text)
        topo = []
        for r in range(ctx.size):
            rc = ctx.cart.Get_coords(r)
            topo.append({"rank": r, "coords": [int(rc[0]), int(rc[1])]})
        manifest = {
            "student_name": STUDENT_NAME,
            "student_id": STUDENT_ID,
            "GITHUB_USERNAME": ctx.github_username,
            "github_username": ctx.github_username,
            "repository": f"mpi-cuda-stencil-{ctx.github_username}",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_name": ctx.gpu_name,
            "gpu_uuid": ctx.gpu_uuid,
            "compute_capability": f"{ctx.cc[0]}.{ctx.cc[1]}",
            "gpu_attributes": ctx.dev_attrs,
            "seed_sha256": ctx.seed_digest,
            "INSTANCE_SEED": str(ctx.instance_seed),
            "instance_seed_hex": f"0x{ctx.instance_seed:016x}",
            "mpi_ranks": ctx.size,
            "cart_dims": PROC_DIMS,
            "periods": PERIODS,
            "rank_topology": topo,
            "mpi_library": MPI.Get_library_version().splitlines()[0].strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "numba": numba.__version__, "mpi4py": sys.modules["mpi4py"].__version__},
            "global_grid": [NY_GLOBAL, NX_GLOBAL],
            "local_padded_shape": list(LOCAL_SHAPE),
            "threadblock": [TILE, TILE],
            "shared_tile": [TILE + 2, TILE + 2],
            "dx": DX, "dy": DY,
            "kappa_min": k_min, "kappa_max": k_max,
            "dt": ctx.dt, "dt_max_cfl": dt_max, "cfl_fraction": ctx.dt / dt_max,
            "lambda": LAMBDA,
            "n_steps": N_STEPS,
            "initial_conditions": init_meta["pulses"],
            "source": init_meta["source"],
            "energy_trace_serialized": trace_ser,
            "energy_trace_overlapped": trace_ovl,
            "energy_monotonic_nonincreasing": monotonic,
            "serialized_ms": ser_ms,
            "overlapped_ms": ovl_ms,
            "serialized_wall_ms": ser_wall,
            "overlapped_wall_ms": ovl_wall,
            "overlap_speedup": speedup,
            "parity_max_diff": parity_max_diff,
            "parity_passed": parity_ok,
            "ptx_counts": counts,
            "ptx_audit_passed": ptx_ok,
            "halo_bytes_per_step_per_rank": {"d2h": HALO_LEN * 8, "h2d": HALO_LEN * 8},
            "forbidden_collectives_on_fields": False,
            "solver_sha256": _file_sha256(os.path.abspath(__file__)),
            "tasks_completed": {
                "task1_topology_seed": True,
                "task2_initialization": True,
                "task3_halo_exchange": True,
                "task4_tiled_stencil_overlap": True,
                "task5_reductions_ptx_benchmark": bool(parity_ok and ptx_ok),
            },
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("-" * 78, flush=True)
        print("[Task 5] submission_manifest.json written", flush=True)
        if parity_ok and ptx_ok:
            print("ALL CHECKS PASSED", flush=True)
        else:
            print("SOME CHECKS FAILED - see above", flush=True)

    ctx.cart.Barrier()
    return 0


if __name__ == "__main__":
    sys.exit(run_solver())
