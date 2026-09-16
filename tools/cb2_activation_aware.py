#!/usr/bin/env python3
"""Does the CB2 deficit come from CALIBRATION or from the REPRESENTATION?

Our CB2 is not a 2-bit quantizer. engine/codebook_sim.py picks, per matrix row, the best 4 of the
16 fixed E2M1 grid levels {0, +-.5, +-1, +-1.5, +-2, +-3, +-4, +-6}, scored by

    hist[level] = sum over weights of scale_g^2 * [code == level]
    cost(subset) = sum_level hist[level] * min_{m in subset} (v_level - v_m)^2

That objective is PURE WEIGHT ERROR -- it never sees an activation. So the measured -2.16 pp
prices this particular scalar requantizer, not two bits.

The cheapest possible improvement keeps the format, the packing and the kernel, and changes one
line: weight each input COLUMN's error by its measured activation second moment, because the error
that matters in y = W x is sum_j (W - Wq)[i,j] * x_j, not sum_j (W - Wq)[i,j]^2.

    wgt = scale2.repeat_interleave(32, dim=1) * act[None, :]

This script measures whether that helps, on OUTPUT error against real activations, before anyone
spends a day on paired NLL or a week on a trellis format.

ONE PASS, because of what is on disk. The lean checkpoint has no routed-expert safetensors -- the
CB3 cache replaced them -- so the only source for routed expert weights is the loaded arena. CB3
codes ARE a subset of the FP4 grid, so dequantizing a CB3 slot returns exact FP4 codes for the 8
levels that slot kept.

SCOPE, stated because it changes what the number means: this measures CB3 -> CB2 (4 of the 8 levels
already chosen), not FP4 -> CB2 (4 of 16). The ABSOLUTE errors are therefore not comparable to the
-2.16 pp result. The A/B is still valid -- both variants requantize the identical source and differ
only in the objective -- which is the question being asked.
"""
import argparse, json, os, sys

import torch

V1 = os.path.expanduser("~/git/deepseek-v41-flash-spark")
sys.path[:0] = [V1, os.path.join(V1, "tools")]

ap = argparse.ArgumentParser()
ap.add_argument("--acts", default=os.path.expanduser("~/ds41-queue/logs/cb2-acts.pt"))
ap.add_argument("--layers", default="0,10,20,30,39")
ap.add_argument("--experts", type=int, default=4)
ap.add_argument("--tokens", type=int, default=2048)
a = ap.parse_args()
LAYERS = [int(x) for x in a.layers.split(",")]


def run():
    import v41_ref as R
    from engine.v41_engine import V41Engine
    from engine.codebook_sim import CodebookSim
    import statistics as st

    # ACTIVATION CACHE. Capturing needs a 65 s engine load and ~24 GB; the comparison itself needs
    # neither. Cache so a variant sweep costs seconds instead of a load each.
    cache = os.path.expanduser("~/ds41-queue/logs/cb2-acts.pt")
    if os.path.exists(cache) and os.environ.get("RECAPTURE") != "1":
        blob = torch.load(cache)
        acc, n, samples = blob["acc"], blob["n"], {k: v.cuda() for k, v in blob["samples"].items()}
        print(f"  loaded cached activations for layers {sorted(samples)} ({blob['tokens']} tokens)")
        return compare(acc, n, samples)

    eng = V41Engine(os.path.expanduser("~/dsv41-lean"), max_seq=8192,
                    arena_gb=float(os.environ.get("ARENA_GB", 40)), spec=True, expert_format="cb3")
    m = eng.model
    # moe_fn's FIRST argument is the FFN input y -- the same vector every routed expert's w1 and w3
    # see. Hooking it is exact and needs no new plumbing; Model.block runs layers in order during
    # prefill, so a cycling counter identifies the layer.
    acc, n, samples, cur = {}, {}, {}, [0]
    orig = m.moe_fn

    def spy(y, slots, wts, arena, limit, *rest, **kw):
        L = cur[0] % eng.args.n_layers
        cur[0] += 1
        f = y.detach().float().reshape(-1, y.shape[-1])
        acc[L] = acc.get(L, 0) + (f * f).sum(0)
        n[L] = n.get(L, 0) + f.shape[0]
        # ACCUMULATE across prompts. Requiring 64 rows from a single prefill left every layer with
        # no sample at all on a 4-prompt corpus of ~30 tokens each -- the run completed and
        # measured nothing.
        if L in LAYERS:
            have = samples.get(L)
            if have is None or have.shape[0] < 128:
                samples[L] = f.clone() if have is None else torch.cat([have, f.clone()])[:128]
        return orig(y, slots, wts, arena, limit, *rest, **kw)

    m.moe_fn = spy
    corpus = [
        "Explain how an NVMe SSD controller schedules writes and why the flash translation layer "
        "matters for tail latency under a mixed read/write workload.",
        "def merge_intervals(intervals):\n    intervals.sort()\n    out = []\n    for s, e in intervals:",
        "Die Wettervorhersage fuer die kommende Woche zeigt einen deutlichen Temperaturrueckgang, "
        "begleitet von anhaltenden Niederschlaegen im Alpenvorland.",
        "In a mixture-of-experts transformer the router assigns each token to a small subset of "
        "feed-forward experts, which makes the memory traffic depend on the routing distribution.",
    ]
    tot = 0
    for text in corpus:
        ids = eng.tokenizer.encode(text, add_special_tokens=False)
        m.c.rollback(0)                      # begin_prompt does NOT rewind c.len; forward asserts it
        m.begin_prompt()
        m.forward(torch.tensor(ids, dtype=torch.long, device="cuda"), 0, prefill=True,
                  need_logits=False)
        tot += len(ids)
        if tot >= a.tokens:
            break
    m.moe_fn = orig
    print(f"  captured {tot} tokens; layers with samples: {sorted(samples)}", flush=True)
    torch.save({"acc": {k: v.cpu() for k, v in acc.items()}, "n": n,
                "samples": {k: v.cpu() for k, v in samples.items()}, "tokens": tot}, cache)
    return compare(acc, n, samples)


def compare(acc, n, samples):
    import v41_ref as R
    from engine.codebook_sim import CodebookSim
    import statistics as st

    sim2 = CodebookSim(2, "cuda")
    sim3 = CodebookSim(3, "cuda")
    # THE REAL FP4 SOURCE, fetched by byte range from the backup checkpoint (tools/fetch_fp4_experts.py).
    # Requantizing from the CB3 arena instead would be 4-of-8 rather than 4-of-16 and would not be
    # comparable to the -2.16 pp result at all.
    src = torch.load(os.path.expanduser("~/dsv41-fp4-partial/experts.pt"))
    print(f"\n  {'layer':>5} {'expert':>6} {'mat':>3} {'CB3':>9} {'CB2 weight':>11} "
          f"{'CB2 act':>9} {'CB2 free':>9}")
    agg = {"cb3": [], "w": [], "act": [], "free": []}
    for L in LAYERS:
        if L not in samples:
            continue
        x = samples[L].cuda() if not samples[L].is_cuda else samples[L]
        aw = (acc[L] / max(1, n[L]))
        for e in range(a.experts):
            for mat in ("w1", "w3"):
                kw = f"layers.{L}.ffn.experts.{e}.{mat}.weight"
                ks = f"layers.{L}.ffn.experts.{e}.{mat}.scale"
                if kw not in src:
                    continue
                w, s = src[kw].cuda(), src[ks].cuda()
                ref = R.dequant_fp4_packed(w, s)
                y0 = x @ ref.T.float()
                errs = []
                for tag, sim, extra in (("cb3", sim3, None), ("w", sim2, None),
                                        ("act", sim2, aw), ("free", None, aw)):
                    if sim is None:
                        wq = requant_free(w, s, extra)
                    else:
                        q = sim.requant_packed(w, s) if extra is None else requant_act(sim, w, s, extra)
                        wq = R.dequant_fp4_packed(q, s)
                    err = ((x @ (ref - wq).T.float()).norm() / y0.norm()).item()
                    agg[tag].append(err); errs.append(err)
                print(f"  {L:>5} {e:>6} {mat:>3} {errs[0]:9.5f} {errs[1]:11.5f} "
                      f"{errs[2]:9.5f} {errs[3]:9.5f}")
    if agg["w"]:
        mw, ma, m3 = st.mean(agg['w']), st.mean(agg['act']), st.mean(agg['cb3'])
        print(f"\n  mean output error   CB3 {m3:.5f}   CB2 weight {mw:.5f}   CB2 act {ma:.5f}")
        print(f"  activation weighting removes {(mw - ma) / mw * 100:.1f} % of CB2's output error")
        mf = st.mean(agg['free'])
        print(f"  CB2 with FREE levels {mf:.5f}")
        if mw > m3:
            print(f"  activation weighting closes {(mw - ma) / (mw - m3) * 100:5.1f} % of the CB2 -> CB3 gap")
            print(f"  free levels        close  {(mw - mf) / (mw - m3) * 100:5.1f} % of the CB2 -> CB3 gap")
            if mf < m3:
                print(f"  free-level 2-bit BEATS 3-bit-on-grid by {(m3 - mf) / m3 * 100:.1f} %")


FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@torch.no_grad()
def pack_fp4(wf: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Recover packed FP4 codes from a dequantized CB3 slot.

    Exact, not approximate: CB3 stores a per-row subset of the E2M1 grid, so every dequantized
    value is (grid level) x (group scale) and divides back to a grid point. The assert guards that
    -- if it ever fires, the slot did not come from a grid-subset format and this whole comparison
    is measuring something else.
    """
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)
    v = (wf.float() / scale).unsqueeze(-1)
    vals = FP4_VALS.to(wf.device)[None, None, :]
    d = (v - vals).abs()
    code = d.argmin(dim=-1)
    assert d.min(dim=-1).values.max() < 1e-3, "dequantized values are not on the FP4 grid"
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)


@torch.no_grad()
def requant_free(w, s, actw=None, iters: int = 12):
    """What 2 bits could do if the 4 levels were NOT locked to the FP4 grid.

    Same 2 bits per weight and the same per-row codebook, but the four values are chosen freely by
    Lloyd-Max on the row's own distribution instead of being picked from
    {0, +-.5, +-1, +-1.5, +-2, +-3, +-4, +-6}. Not representable by today's packing -- the kernel
    rebuilds an FP4 nibble from the index -- so this is a BOUND on what changing the format buys,
    measured before anyone writes a kernel for it.

    Returns dequantized weights directly, since the result has no FP4 code to pack into.
    """
    N, K2 = w.shape
    K = K2 * 2
    lo = (w & 0x0F).long(); hi = ((w >> 4) & 0x0F).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)
    vals = FP4_VALS.to(w.device)[codes] * scale                      # [N, K] real weights
    wgt = torch.ones_like(vals) if actw is None else actw[None, :K].to(w.device).expand(N, K)
    # init on quantiles of each row, then alternate assign / recompute
    q = torch.tensor([0.125, 0.375, 0.625, 0.875], device=w.device)
    cb = torch.quantile(vals.float(), q, dim=1).T.contiguous()       # [N, 4]
    for _ in range(iters):
        d = (vals.unsqueeze(-1) - cb.unsqueeze(1)).abs()             # [N, K, 4]
        a = d.argmin(dim=-1)
        oh = torch.nn.functional.one_hot(a, 4).to(vals.dtype) * wgt.unsqueeze(-1)
        num = (oh * vals.unsqueeze(-1)).sum(1)
        den = oh.sum(1).clamp_min(1e-9)
        cb = num / den
    d = (vals.unsqueeze(-1) - cb.unsqueeze(1)).abs()
    return torch.gather(cb, 1, d.argmin(dim=-1))


@torch.no_grad()
def requant_act(sim, w, s, actw):
    """CodebookSim.requant_packed with ONE change: each column's error is weighted by E[x_j^2]."""
    N, K2 = w.shape
    K = K2 * 2
    lo = (w & 0x0F).long()
    hi = ((w >> 4) & 0x0F).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    scale2 = torch.exp2(2.0 * (s.float() - 127.0))
    wgt = scale2.repeat_interleave(32, dim=1) * actw[None, :K].to(w.device)
    hist = torch.zeros(N, 16, device=w.device, dtype=torch.float32)
    hist.scatter_add_(1, codes, wgt)
    best = (hist @ sim.cost.T).argmin(dim=1)
    new_codes = sim.near[best][torch.arange(N, device=w.device)[:, None], codes]
    return (new_codes[:, 0::2] | (new_codes[:, 1::2] << 4)).to(torch.uint8)


run()
