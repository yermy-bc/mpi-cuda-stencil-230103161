# Cache Line & False Sharing Mechanics — Analysis (Q1–Q5)

**Student:** Ainat Yermurat  **ID:** 230103161  **Git:** `yermy-bc`
**Instructor:** Sufyan bin Uzayr
**Artifacts:** `false_sharing_lab.py`, `cache_lab_manifest.json` (`"verification_passed": true`)

## Test system & run configuration

| item | value (from my run / manifest) |
|---|---|
| CPU | 11th Gen Intel Core i5-11300H @ 3.10 GHz (Tiger Lake, 4 cores / 8 threads) |
| OS / runtime | Windows 11 (10.0.26300), Python 3.15.0, NumPy 2.5.5, Numba 0.68.1 |
| logical cores | 8 |
| L1D coherency line size | 64 B → 8 × `uint64` per line |
| `cpu_sig` | `AMD64::Intel64 Family 6 Model 140 Stepping 1, GenuineIntel::8` |
| SHA-256(`yermy-bc` + cpu_sig) | `290535f245bd1407cdaeacfbe370527ed3bc548216274c1badc723b162900669` |
| `INSTANCE_SEED` | 2955828045106975751 (`0x290535f245bd1407`) |
| thread affinity | logical CPUs **0, 2, 4, 6** via `SetThreadAffinityMask`. Windows numbers SMT siblings in adjacent pairs (0/1, 2/3, …), so these are 4 *different physical cores* and the threads do not share an L1. |
| workload | 4 threads × 60,000,000 read-modify-write updates, 3 runs per layout, median reported |
| context switches | `resource.getrusage` does not exist on Windows (POSIX-only module), so the manifest records `null` for voluntary/involuntary switches |

Measured results (median of 3 runs; the runs were very stable, e.g. stride 1: 203.1 / 203.0 / 202.8 ms):

| layout | stride | slot byte offsets | cache lines used | elapsed (ns) | ns / update (per thread) | vs stride 8 |
|---|---|---|---|---|---|---|
| A Contiguous | 1 (8 B) | 0, 8, 16, 24 | 0, 0, 0, 0 | **202,952,800** | **3.383** | **1.91×** |
| B Half-line | 4 (32 B) | 0, 32, 64, 96 | 0, 0, 1, 1 | **226,644,300** | **3.777** | **2.13×** |
| C Line-padded | 8 (64 B) | 0, 64, 128, 192 | 0, 1, 2, 3 | **106,390,000** | **1.773** | 1.00× |
| D Prefetch-safe | 16 (128 B) | 0, 128, 256, 384 | 0, 2, 4, 6 | **106,042,300** | **1.767** | 0.997× |
| Register-privatized | 1 (single final store) | 0, 8, 16, 24 | 0, 0, 0, 0 | **35,352,100** | **0.589** | 0.33× |

Parity: every layout and the privatized baseline produced bit-identical final counters
`[1140000000, 2700000000, 3420000000, 4980000000]` (= 60,000,000 × per-thread increment
19, 45, 57, 83, all derived from `INSTANCE_SEED`). All other buffer cells remained 0.

---

## Q1 — MESI state transitions under `stride = 1`

With stride 1, `buffer[0]` (offset 0 B) and `buffer[1]` (offset 8 B) lie in the **same 64-byte line L**
(my run: cache lines `[0, 0, 0, 0]`). Let Core 0 (logical CPU 0) store to `buffer[0]`, and let Core 1
(logical CPU 2, a different physical core) immediately store to `buffer[1]`:

| step | event | Core 0 copy of L | Core 1 copy of L |
|---|---|---|---|
| 0 | cold start | I | I |
| 1 | Core 0 executes `load buffer[0]` (the kernel reads before it writes) | **E** (sole reader; filled from L3/DRAM) | I |
| 2 | Core 0 retires the store `buffer[0] = v` | E → **M** (silent upgrade, no bus traffic) | I |
| 3 | Core 1 executes `load buffer[1]` | snooped: M → **S** (dirty data supplied or written back) | I → **S** |
| 4 | Core 1's store `buffer[1] = w` reaches the head of its store buffer. It needs ownership, so it issues an **RFO / invalidate** for L | S → **I** | S → (waits) → **M** |
| 5 | Core 0's next iteration: `load buffer[0]` misses (line is I) and its store needs an RFO | I → S → M after its own RFO | M → S → **I** |
| … | the two cores keep stealing the line from each other ("ping-pong") | M ↔ I | I ↔ M |

When both cores only write (no intervening load), step 3 is skipped. The RFO in step 4 then takes the line
from **M in Core 0 directly to I**: Core 0 must forward its dirty copy (cache-to-cache transfer through
the L3/ring on Tiger Lake), and Core 1 receives it in M. The logical data never conflict, because each core
writes a different 8-byte word. But coherence is tracked **per 64-byte line**, so each write by one core
invalidates the whole line in the other core.

**Why the RFO stalls Core 1's store buffer.** A store retires into the core's store buffer immediately. It
can only *commit* to L1 when the line is held in M (or E) state, because only the owner may modify it. While
the RFO is outstanding (tens of ns: snoop Core 0, wait for its dirty data and for the invalidation
acknowledgement), the store at the head of the buffer cannot drain. Stores must commit in program order
(x86-TSO), so every younger store queues behind it. Core 1 can still run ahead for a while: its own loads of
`buffer[1]` are satisfied by **store-to-load forwarding** from the buffered store. Once the store buffer is
full, however, the allocation stage stalls and no new store can issue. The core is then stalled on
coherence, not on computation. The same happens to Core 0 when the line is taken back. In my run the
contention adds **1.610 ns per update** (3.383 − 1.773 ns). A cross-core line transfer costs several tens of
ns, so this means each ownership period drains a burst of roughly dozens of buffered stores before the line
is stolen again. This batching explains why the slowdown is ~2× rather than ~20×.

---

## Q2 — Quantitative slowdown & coherency overhead

| regime | elapsed (ns) | ns / update per thread (= elapsed / 60 M) | ns / update aggregate (= elapsed / 240 M) | total throughput |
|---|---|---|---|---|
| stride 1 (4-way false sharing) | 202,952,800 | **3.383** | 0.846 | 1.18 G updates/s |
| stride 4 (2-way false sharing) | 226,644,300 | **3.777** | 0.944 | 1.06 G updates/s |
| stride 8 (one line per thread) | 106,390,000 | **1.773** | 0.443 | 2.26 G updates/s |
| stride 16 (128 B apart) | 106,042,300 | **1.767** | 0.442 | 2.26 G updates/s |
| register-privatized | 35,352,100 | **0.589** | 0.147 | 6.79 G updates/s |

**Slowdown stride 1 vs stride 8 = 202,952,800 / 106,390,000 = 1.91×.** The four threads touch completely
independent counters. The only difference is that in Layout A they share one physical line.

The three regimes separate cleanly:

* **Register-privatized, 0.589 ns ≈ 2.4 cycles at ~4 GHz.** This is the pure ALU dependency chain
  `acc = acc + inc + (acc >> 63)` (shift, then add) kept in a register. There are no memory operations in the
  loop (LLVM shows **0** `load i64` in this kernel).
* **Stride 8 / 16, 1.77 ns ≈ 7 cycles.** Each core owns its line in M state forever, so every update is an L1
  hit. The loop-carried dependency now runs *through memory*: store → store-to-load forwarding (~5 cycles) →
  add (1 cycle). This is the **L1 store-forwarding latency**, and it costs 1.773 − 0.589 = **1.18 ns per
  update** (3.01× the register version).
* **Stride 1, 3.38 ns.** The same work, plus the **cross-core coherence latency** from RFO ping-pong:
  **+1.61 ns per update** on top of store forwarding, i.e. 5.74× slower than the register baseline.

Context-switch counts are `null` because `getrusage` is unavailable on Windows. All four worker threads were
pinned to their own physical core, and the run-to-run spread was below 1 %. This rules out scheduler noise as
the cause of the 1.91× gap.

**Stride 4 slower than stride 1 (2.13× vs 1.91×).** With stride 4, threads 0/1 share line 0 and threads 2/3
share line 1. Each line has only 2 contenders, yet the result is *worse* than 4 contenders on one line. Lines
0 and 1 form one even/odd 128-byte pair, so the adjacent-line prefetcher (Q3) can pull the buddy line on every
coherence miss. That in turn triggers extra snoops and invalidations against the *other* pair of cores, so all
4 cores still interfere, now across two lines. Under stride 1 a single line migrates, and each owner drains a
longer burst of coalesced stores per ownership period. The key point is that halving the sharing degree did
**not** halve the cost: 32-byte padding is useless, and only full-line separation (stride ≥ 8) removes the
penalty.

---

## Q3 — L2 spatial prefetcher & 128-byte pairing

Measured: **stride 8 = 106,390,000 ns (1.773 ns/update)** vs **stride 16 = 106,042,300 ns (1.767 ns/update)**.
The ratio is 0.997, a 0.3 % difference that lies within run-to-run noise (stride 16 runs: 106.0, 106.0,
108.3 ms). On my i5-11300H, 64-byte padding already removes **all** measurable false sharing for this workload.

**How the pairing works.** Intel cores (and many ARM cores) have an *L2 adjacent-cache-line (spatial)
prefetcher*. When a line misses into L2, it also requests the "buddy" line that completes the aligned
**128-byte pair**: line 2k is paired with 2k+1. Physical address bit 6 selects even/odd, and bits ≥ 7 select
the pair. The L2 streamer can likewise fetch neighbouring lines. In stride-8 Layout C the threads own lines 0,
1, 2, 3, so threads 0/1 sit in one 128-byte pair and threads 2/3 in another. In stride-16 Layout D (lines 0, 2,
4, 6) every thread owns an entire 128-byte pair.

**When 64-byte padding still suffers secondary invalidations.** The prefetcher only acts on **L2 misses**. If
Core 0 misses on line 0, it may prefetch line 1, which Core 1 currently holds in M. The prefetch is a read, so
it snoops Core 1 and downgrades the line M → S. Core 1's next store then needs an upgrade/RFO even though no
other thread ever wrote that line. This "secondary" invalidation occurs when:
* the lines actually miss repeatedly: true sharing or eviction pressure on the slot's line, other data in the
  same line, many threads per pair, or a working set larger than L1/L2;
* the hardware prefetchers are enabled, which is the default (BIOS setting "Adjacent Cache Line Prefetch");
* or the data are written with a pattern that keeps re-triggering misses, as in my stride-4 layout (Q2).

In my stride-8 run each core's line stays resident in its own L1 in M state for all 60 M iterations, so there
are no misses that could trigger a buddy prefetch. Hence stride 8 = stride 16. The 128-byte layout is the
robust choice for real code; this is why Intel guidance and C++'s `hardware_destructive_interference_size` on
some platforms use 128 B. For this tight loop on Tiger Lake, however, 64 B was sufficient.

---

## Q4 — Compiler register promotion vs real memory stores (LLVM IR)

LLVM audit of `worker_strided_accumulate` from my run
(signature `(Array(uint64, 1, 'C', False, aligned=True), uint64, int64, uint64)`):
**`store i64` = 10, `load i64` = 9, `phi` = 6, `br i1` = 5, `PyObject` references = 0, NRT calls = 0**, 88
lines in the native function body. The kernel compiled in pure nopython mode with no Python object fallback.

**Why a compiler wants to hoist `buffer[slot]`.** In the naive loop `for i in range(n): buffer[slot] += inc`,
the address `&buffer[slot]` is loop-invariant and no other store in the loop can alias it. LLVM's **LICM with
scalar promotion** (store-to-register promotion) is then legal. Under the language memory model another
thread's writes do not have to be observed without synchronisation, so the optimiser may load once before the
loop, keep the value in a register (`phi`), and store once after the loop. It may even fold the whole loop to
`buffer[slot] += n*inc`. The benchmark would then measure nothing: no stores and no coherence traffic. This is
exactly what my **control** kernel does on purpose. Its IR has **0 `load i64`** in the loop, and its
**2 `store i64`** are the single final `buffer[slot] = acc` plus the return-value slot.

**How my data dependency prevents promotion.** The kernel computes
`idx = slot + (v >> 63); v = buffer[idx] + inc; buffer[idx] = v`. At runtime `v < 2^63` always holds, so
`idx == slot`, but the compiler cannot prove this. The address of iteration k depends on the value loaded in
iteration k−1, so the address is **not loop-invariant** and LICM/promotion is impossible. My IR excerpt shows
LLVM **unrolling the loop 8×**, yet each copy keeps the full chain:

```llvm
%v.2.085 = phi i64 [ 0, %B168.lr.ph.new ], [ %.187.7, %B168 ]
%.156    = lshr i64 %v.2.085, 63          ; v >> 63  (opaque "zero")
%.185    = load i64, ptr %.184, align 8   ; real L1 load of buffer[idx]
%.187    = add i64 %.185, %.22            ; + inc (inc hoisted once as %.22)
store i64 %.187, ptr %.184, align 8       ; real L1 store
%.156.1  = lshr i64 %.187, 63             ; next address depends on the stored value
%.185.1  = load i64, ptr %.184.1, align 8
...                                        ; x8 per unrolled body
```

The unrolled body has 8 `load i64` + 8 `store i64`; the remainder (epilogue) loop adds 1 + 1, and the result
return adds 1 store. Together that gives the 9 loads / 10 stores counted. Note that LLVM *did* hoist the truly
invariant computation `inc = (seed & 0xFF) | 1` into `%.22`. It optimised everything it legally could, except
the memory operations, which shows the dependency works exactly as intended. Every one of the 60 M iterations
issues an `i64` store to `buffer[slot]`, and that store is what generates the RFO traffic measured in Q2. This
achieves the effect of a C `volatile` access without Numba having a `volatile` qualifier.

---

## Q5 — CPython GIL masking vs `nogil=True` execution

I ran the identical stride-1 read-modify-write loop as pure Python bytecode in `threading.Thread`s
(2,000,000 updates per thread; 60 M would take minutes):

| configuration | elapsed (ns) | aggregate ns / update |
|---|---|---|
| 1 thread, stride 1 | 181,059,000 | 90.53 |
| 4 threads, stride 1 | 670,082,900 | 83.76 |
| 4 threads, stride 8 | 679,054,700 | 84.88 |

* **No parallel speed-up.** 4 threads doing 4× the work took **3.70×** longer (670.1 / 181.1 ms). Each thread
  therefore needed ~335 ns of wall time per update instead of 90.5 ns. Throughput did not scale; the threads
  simply took turns.
* **No false-sharing signature.** Stride 1 vs stride 8 under the GIL: 670.1 vs 679.1 ms, ratio **0.987**,
  which is no difference at all. Compare the native `nogil` kernels, where the same layout change gave
  **1.91×**.
* **The cost is ~50× higher than the hardware effect.** 83.8–90.5 ns per update in bytecode vs 1.77–3.38 ns
  natively. The interpreter overhead per update (fetching and decoding bytecode, allocating Python `int`s,
  reference counting, list `__getitem__`/`__setitem__`) is ~25–50× larger than the entire coherence penalty
  (1.61 ns) it would have to reveal.

**Why.** The GIL lets only **one thread execute bytecode at any instant**. The others are blocked in the GIL
mutex/condition variable and are released at the switch interval (`sys.getswitchinterval()` = 5 ms) or on
blocking calls. The four "concurrent" threads therefore never store to the shared line *simultaneously*. A
5 ms time slice amounts to tens of thousands of updates by a single core while the others sleep. The line
stays in M in that one core, and an ownership transfer happens only once per GIL hand-off (a few hundred
times per second) instead of millions of times. With more threads, the run gets **no faster and can get
slower**, because every hand-off adds lock contention cost: OS wake-ups and context switches, convoying on the
GIL, the dropped-request handshake, and cache migration of the interpreter's own hot data (thread states,
frame objects, refcounts of shared objects). The `list` object, its refcount and the boxed `int`s are
themselves shared, so the threads also truly share interpreter data structures. In my run that overhead
roughly cancels the small gain from cache warmth: per-update cost went from 90.5 ns to 83.8 ns, and zero work
ran in parallel. It is **lock contention that serialises the threads, not hardware cache-line contention**.

With `@njit(nogil=True)` the compiled kernel releases the GIL for its entire 60 M-iteration loop. The four OS
threads then really run in parallel on cores 0, 2, 4, 6, issuing native `store i64` instructions back-to-back.
This is the only situation in which MESI RFO ping-pong becomes the bottleneck and can be measured from
Python: 202.95 ms (stride 1) vs 106.39 ms (stride 8).
