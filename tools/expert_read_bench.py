#!/usr/bin/env python3
"""Time 0xBakeer's OWN expert reader, not a re-implementation of it.

My first attempt at this measured a harness that issued an expert's chunks serially inside one
thread, and concluded `read_chunk_mb=4` costs 41%. `engine/experts.py:219` submits every chunk but
the first to a second pool, so that harness did not describe this engine. This one imports
`ExpertStore` and calls `_read_leased`, so whatever it reports is what the engine does.

Sweeps DSV41_READ_CHUNK_MB x concurrent experts, with DSV41_IO_THREADS fixed per run. Reads only
layer shards that are actually on disk. Nothing is written and no arena is allocated: the sink is a
no-op, so this measures the read path alone (staging lease + O_DIRECT + chunk scheduling), which is
the term the engine spends 2.68 GB/s on against a 6.8 GB/s device.
"""
import argparse, json, os, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--model-dir", default="/opt/llm/models/dsv41-shards")
ap.add_argument("--chunks", default="2,4,8,32")
ap.add_argument("--conc", default="1,2,4")
ap.add_argument("--io-threads", default="12")
ap.add_argument("--read-threads", type=int, default=12)
ap.add_argument("--n", type=int, default=48, help="expert reads per cell")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

sys.path.insert(0, a.repo)
index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
wmap = index["weight_map"]

# which layers do we have a shard for?
have = {f for f in os.listdir(a.model_dir) if f.endswith(".safetensors")}
layers = sorted({int(k.split("layers.")[1].split(".")[0])
                 for k, v in wmap.items() if ".ffn.experts." in k and v in have})
if not layers:
    sys.exit("no layer shard with experts is present")
print(f"  layers on disk: {layers}")


class StubArena:
    slots = 64          # only has to exceed transient_slots; no memory is allocated
    bytes_per_slot = 0


def cell(chunk_mb, conc, io_threads, n):
    os.environ["DSV41_READ_CHUNK_MB"] = str(chunk_mb)
    os.environ["DSV41_IO_THREADS"] = str(io_threads)
    import importlib
    import engine.experts as EX
    importlib.reload(EX)
    r = EX.ExpertStore(a.model_dir, index, StubArena(), 40,
                        transient_slots=8, read_threads=a.read_threads)
    rnd = random.Random(1234)
    jobs = [(rnd.choice(layers), rnd.randrange(384)) for _ in range(n)]
    for l, e in jobs[:conc]:                       # warm the fds
        r._read_leased(l, e, None, lambda v: None)
    r.stats["bytes_read"] = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as pool:
        list(pool.map(lambda le: r._read_leased(le[0], le[1], None, lambda v: None), jobs))
    dt = time.perf_counter() - t0
    gbs = r.stats["bytes_read"] / dt / 1e9
    for sh in list(getattr(r, "_shards", {}).values()):
        try: os.close(sh.fd)
        except Exception: pass
    return gbs


print(f"  {a.n} expert reads per cell, {a.reps} reps, read_threads={a.read_threads}\n")
for io_threads in [int(x) for x in a.io_threads.split(",")]:
    print(f"  DSV41_IO_THREADS={io_threads}")
    print(f"  {'chunk MB':>9} " + "".join(f"{'conc '+c:>14}" for c in a.conc.split(",")))
    for cm in [float(x) for x in a.chunks.split(",")]:
        row = []
        for conc in [int(x) for x in a.conc.split(",")]:
            v = [cell(cm, conc, io_threads, a.n) for _ in range(a.reps)]
            row.append(f"{statistics.median(v):8.2f} GB/s")
        print(f"  {cm:>9g} " + "".join(f"{c:>14}" for c in row))
    print()
print("== ALL DONE ==")
