"""One long request, with the box sampled underneath it: where does the time actually go?"""
import json, subprocess, sys, threading, time, urllib.request, os
BASE="http://127.0.0.1:8001/v1/chat/completions"
TARGET=int(sys.argv[1]) if len(sys.argv)>1 else 8000
OUT=int(sys.argv[2]) if len(sys.argv)>2 else 256

# a long, real prompt: the engine's own NOTES, truncated to about TARGET tokens
src=open(os.path.expanduser("~/git/deepseek-v41-flash-spark/NOTES.md")).read()
words=src.split()
prompt=" ".join(words[:int(TARGET*0.75)])
q=("Read the following engineering notes and then answer in detail: what are the three most "
   "important performance bottlenecks described, and what would you measure next?\n\n"+prompt)

stop=threading.Event(); samples=[]
def dev_read_bytes():
    for l in open("/proc/diskstats"):
        f=l.split()
        if f[2]=="nvme0n1": return int(f[5])*512, int(f[9])*512   # sectors read, sectors written
    return 0,0
def sampler():
    r0,w0=dev_read_bytes(); t0=time.time()
    while not stop.is_set():
        time.sleep(0.5)
        try:
            g=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,utilization.memory",
                              "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3).stdout.strip()
            gu=int(g.split(",")[0])
        except Exception: gu=-1
        r,w=dev_read_bytes(); t=time.time()
        samples.append((t-t0, gu, r-r0, w-w0)); r0,w0,t0=r,w,t
th=threading.Thread(target=sampler,daemon=True); th.start()

req=urllib.request.Request(BASE, method="POST", headers={"Content-Type":"application/json"},
    data=json.dumps({"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":q}],
                     "temperature":0.6,"max_tokens":OUT,"stream":True}).encode())
t_start=time.time(); ttft=None; ntok=0; stats=None
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line=raw.decode().strip()
        if not line.startswith("data: "): continue
        body=line[6:]
        if body=="[DONE]": break
        o=json.loads(body)
        if o.get("x_engine_stats"): stats=o["x_engine_stats"]
        ch=o.get("choices") or [{}]
        d=ch[0].get("delta") or {}
        if d.get("content"):
            ntok+=1
            if ttft is None: ttft=time.time()-t_start
total=time.time()-t_start
stop.set(); time.sleep(0.7)

pre=[s for s in samples if s[0]<=0] # unused
tprefill=ttft or 0
sp=[s for s in samples]
def phase(lo,hi):
    el=0; gsum=0; rd=0; wr=0; n=0
    t=0
    for dt,gu,r,w in sp:
        t+=dt
        if lo<=t<hi:
            el+=dt; gsum+=gu*dt if gu>=0 else 0; rd+=r; wr+=w; n+=1
    return el, (gsum/el if el else 0), rd, wr, n
p_el,p_gpu,p_rd,p_wr,_=phase(0,tprefill)
d_el,d_gpu,d_rd,d_wr,_=phase(tprefill,1e9)
print(f"\n  prompt ~{TARGET} tok, generated {ntok} tokens")
print(f"  TTFT {tprefill:8.2f} s     decode {total-tprefill:8.2f} s     total {total:8.2f} s")
print(f"  decode rate {ntok/max(total-tprefill,1e-9):6.2f} tok/s")
print(f"\n  {'phase':<8} {'s':>8} {'GPU busy':>9} {'NVMe read':>11} {'read GB/s':>10} {'GB/token':>9}")
print(f"  {'prefill':<8} {p_el:8.2f} {p_gpu:8.1f}% {p_rd/1e9:10.1f} GB {p_rd/1e9/max(p_el,1e-9):10.2f} {'-':>9}")
print(f"  {'decode':<8} {d_el:8.2f} {d_gpu:8.1f}% {d_rd/1e9:10.1f} GB {d_rd/1e9/max(d_el,1e-9):10.2f} "
      f"{d_rd/1e9/max(ntok,1):9.3f}")
if stats: print(f"\n  engine: {json.dumps(stats, indent=None)[:600]}")
print("== ALL DONE ==")
