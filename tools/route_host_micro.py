"""How much of route_s is actually Python? Replay resolve()'s HOST work on real recorded
routes, with no GPU and no D2H copy, and time it."""
import collections, json, time
import numpy as np

log = [json.loads(l) for l in open("/home/jschmied/ds41-queue/logs/route-11k.jsonl") if l.strip()]
dec = [c for c in log if c["pf"] == 0][:400]
print(f"{len(dec)} recorded decode resolve() calls, "
      f"{np.mean([len(c['uniq']) for c in dec]):.1f} unique experts each")

N_EXPERTS, N_SLOTS = 384, 5465
lru = collections.OrderedDict((i, i % N_SLOTS) for i in range(N_SLOTS))
tmap = {}

def host_work(layer, ex):
    """Everything resolve() does between the .numpy() and the pool.map -- no CUDA."""
    uniq = np.unique(ex)
    slot_of, to_load, used = {}, [], set()
    for e in uniq.tolist():
        key = (layer, e)
        s = lru.get(key)
        if s is None:
            s = tmap.get(key)
        else:
            lru.move_to_end(key)
        if s is not None:
            slot_of[e] = s; used.add(s)
    for e in uniq.tolist():
        if e in slot_of: continue
        s = (len(used) + e) % N_SLOTS
        slot_of[e] = s; used.add(s); to_load.append(((layer, e), s))
    lut = np.full(N_EXPERTS, -1, dtype=np.int32)
    for e, s in slot_of.items():
        lut[e] = s
    return lut[ex.astype(np.intp)]

blocks = [(c["L"], np.array(c["uniq"], dtype=np.int32)[None, :].repeat(6, 0)) for c in dec]
for _ in range(3):                                   # warm
    for L, ex in blocks[:40]: host_work(L, ex)
t0 = time.perf_counter()
for L, ex in blocks: host_work(L, ex)
dt = (time.perf_counter() - t0) / len(blocks)
print(f"\n  pure host bookkeeping: {dt*1000:.3f} ms per resolve() call")
print(f"  measured route_s     : 12.7 ms per layer (§19, 508 ms/step over 40 layers)")
print(f"  => Python is {100*dt*1000/12.7:.1f} % of route_s; the rest is NOT Python")
