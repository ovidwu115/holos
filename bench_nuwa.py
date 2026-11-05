#!/usr/bin/env python3
"""Scale benchmark for the Nuwa substrate (paper §4.1.3 analogue).
Measures: bulk seed creation, bytes/agent on disk, resident memory, point-lookup latency vs
population (O(1) check), brute-force vector retrieval vs population (O(N), reported honestly),
and JIT instantiation latency (profile fetch + S-MMR).   Usage: python bench_nuwa.py [N]"""
import os, random, sqlite3, sys, time, zlib
import numpy as np
import holos
from holos import Nuwa, MockLLM, ROLES, QUALS, TASK_TOPIC, TIERS, TIER_PRICE, embed, DIM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
DB = "/tmp/nuwa_bench.db"
POPS = [p for p in (10, 100, 1_000, 10_000, 100_000, 1_000_000) if p <= N]

def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1024

def pct(a, q): return float(np.percentile(a, q))

if os.path.exists(DB): os.remove(DB)
rng = random.Random(0); nrng = np.random.default_rng(0)
nuwa = Nuwa(MockLLM(0)); nuwa.db.close()
nuwa.db = sqlite3.connect(DB, check_same_thread=False)
nuwa.db.execute("PRAGMA journal_mode=OFF"); nuwa.db.execute("PRAGMA synchronous=OFF")
nuwa.db.execute("CREATE TABLE seeds(id INTEGER PRIMARY KEY, role, qual, task, profile, ability REAL, price REAL)")

# unique phenotype texts -> cached embeddings (variants share a base vector + small noise)
tasks, flavors = list(TASK_TOPIC), ["methodical", "fast", "thorough", "pragmatic", "curious"]
combos = [(r, q, t, f) for r in ROLES for q in QUALS for t in tasks for f in flavors]
texts = [f"{r} ({q}) specialized in {t}. Working style: {f}." for r, q, t, f in combos]
base = np.stack([embed(x) for x in texts]).astype(np.float32)

t0 = time.time(); ids = []; vecs = []; made = 0; batch = []
while made < N:
    i = rng.randrange(len(combos)); r, q, t, f = combos[i]
    aid = zlib.crc32(f"{r}|{q}|{t}|{rng.randrange(10**9)}".encode())
    batch.append((aid, r, q, t, texts[i], rng.uniform(0.3, 0.95), TIER_PRICE[TIERS[aid % 3]], i))
    if len(batch) == 50_000 or made + len(batch) >= N:
        cur = nuwa.db.executemany("INSERT OR IGNORE INTO seeds VALUES(?,?,?,?,?,?,?)", [b[:7] for b in batch])
        # collisions are rare (32-bit space); rowcount tells how many inserted
        made_now = nuwa.db.execute("SELECT COUNT(*) FROM seeds").fetchone()[0] - made
        ids.extend(b[0] for b in batch); vecs.append(np.array([b[7] for b in batch], np.int32)); made += made_now
        batch = []
        print(f"  seeds {made:>9,}  {time.time()-t0:6.1f}s", flush=True)
nuwa.db.commit()
creation_s = time.time() - t0
rows = nuwa.db.execute("SELECT id FROM seeds ORDER BY id").fetchall()
nuwa.ids = np.array([r[0] for r in rows], np.int64)
# vector matrix (float32): base embedding of the agent's phenotype + variant noise
pt = dict(zip(ids, np.concatenate(vecs)))
ptr = np.array([pt[i] for i in nuwa.ids], np.int32)
mat = np.empty((len(ptr), DIM), np.float32)
for s in range(0, len(ptr), 100_000):
    m = base[ptr[s:s+100_000]] + 0.03 * nrng.standard_normal((len(ptr[s:s+100_000]), DIM)).astype(np.float32)
    mat[s:s+100_000] = m / np.linalg.norm(m, axis=1, keepdims=True)
nuwa.mat = mat
disk = os.path.getsize(DB)
print(f"\nN={len(nuwa.ids):,}  create {creation_s:.1f}s ({len(nuwa.ids)/creation_s:,.0f} seeds/s, no LLM)")
print(f"disk: {disk/1e6:.1f} MB  ({disk/len(nuwa.ids):.0f} B/agent; 10M est {disk/len(nuwa.ids)*1e7/1e9:.2f} GB)")
print(f"RSS: {rss_mb():.0f} MB total (vector index {mat.nbytes/1e6:.0f} MB = {mat.nbytes/len(mat):.0f} B/agent)")

print("\npopulation | point lookup mean/P99 (ms) | top-10 retrieval mean/P99 (ms)")
q = embed("data analysis of financial figures").astype(np.float32)
for p in POPS:
    sample = nuwa.ids[:p] if p == len(nuwa.ids) else nuwa.ids[rng.sample(range(len(nuwa.ids)), p)]
    keys = [int(sample[rng.randrange(p)]) for _ in range(500)]
    lat = []
    for k in keys:
        t = time.perf_counter(); nuwa.db.execute("SELECT * FROM seeds WHERE id=?", (k,)).fetchone(); lat.append((time.perf_counter() - t) * 1e3)
    sub = holos.Nuwa.__new__(holos.Nuwa); sub.mat, sub.ids = mat[:p], nuwa.ids[:p]
    rl = []
    for _ in range(20 if p >= 100_000 else 100):
        t = time.perf_counter(); sub.retrieve(q, 10); rl.append((time.perf_counter() - t) * 1e3)
    print(f"{p:>10,} | {np.mean(lat):7.3f} / {pct(lat,99):7.3f}        | {np.mean(rl):8.2f} / {pct(rl,99):8.2f}")

il = []
for k in nuwa.ids[nrng.integers(0, len(nuwa.ids), 200)]:
    t = time.perf_counter(); nuwa.instantiate(int(k)); il.append((time.perf_counter() - t) * 1e3)
print(f"\nJIT instantiate (profile fetch + S-MMR K=5, 40-tool pool): mean {np.mean(il):.2f} ms, P99 {pct(il,99):.2f} ms")
print(f"final RSS {rss_mb():.0f} MB")
