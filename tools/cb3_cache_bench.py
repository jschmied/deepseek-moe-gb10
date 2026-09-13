#!/usr/bin/env python3
"""FP4-checkpoint miss vs native-CB3-cache miss, through the engine's own ExpertStore.

Both arms call `ExpertStore._load_into_slot`, so both pay the staging lease, the copy stream and the
arena write. The differences are exactly the ones the format is meant to remove: 18,800,640 B in two
runs 585 MB apart plus `fp4_to_cb3_v2` on the GPU, against 13,774,848 B in one aligned extent plus a
3-bit scale expansion.
"""
import argparse, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--model-dir", default="/opt/llm/models/dsv41-shards")
ap.add_argument("--cache", required=True)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--slots", type=int, default=64)
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--conc", default="1,2,4")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "tools"))
import torch
import cb3_moe as C3
from engine.codebook_sim import CodebookSim

index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
dev = torch.device("cuda")


def run(use_cache, conc):
    if use_cache:
        os.environ["DSV41_CB3_CACHE"] = a.cache
    else:
        os.environ.pop("DSV41_CB3_CACHE", None)
    import importlib, engine.experts as EX
    importlib.reload(EX)
    arena = C3.CB3ArenaV2(a.slots, dev); arena.sim = CodebookSim(3, dev)
    st = EX.ExpertStore(a.model_dir, index, arena, 40, transient_slots=8, read_threads=12)
    jobs = [((a.layer, e % 384), e % a.slots) for e in range(a.n)]
    for k, s in jobs[:conc]:
        st._load_into_slot(k, s)
    st.stats["bytes_read"] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as pool:
        list(pool.map(lambda ks: st._load_into_slot(*ks), jobs))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    nb = st.stats["bytes_read"]
    if getattr(st, "cb3_cache", None):
        st.cb3_cache.close()
    del st, arena
    torch.cuda.empty_cache()
    return dt, nb


print(f"  {a.n} expert loads per cell, layer {a.layer}, {a.slots}-slot CB3 arena, median of {a.reps}")
print(f"  {'arm':<22} {'conc':>4} {'MB/load':>8} {'ms/load':>8} {'GB/s':>7} {'total ms':>9}")
res = {}
for use_cache, tag in ((False, "FP4 checkpoint"), (True, "native CB3 cache")):
    for conc in [int(x) for x in a.conc.split(",")]:
        v = [run(use_cache, conc) for _ in range(a.reps)]
        dts = sorted(x[0] for x in v); dt = dts[len(dts) // 2]
        nb = v[0][1]
        res[(tag, conc)] = dt
        print(f"  {tag:<22} {conc:>4} {nb/a.n/1e6:8.2f} {dt/a.n*1e3:8.3f} {nb/dt/1e9:7.2f} {dt*1e3:9.1f}")
print()
for conc in [int(x) for x in a.conc.split(",")]:
    f, c = res[("FP4 checkpoint", conc)], res[("native CB3 cache", conc)]
    print(f"  conc {conc}: cache is {f/c:.3f}x the FP4 path  ({(1-c/f)*100:+.1f}% wall)")
print("== ALL DONE ==")
