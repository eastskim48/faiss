import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import h5py

try:
    import faiss
except ImportError as e:
    raise SystemExit(
        "faiss import failed. Install faiss-gpu (conda/pip) for your CUDA version.\n"
        f"Error: {e}"
    )

# Optional: GPU props for SM count (nice for logs)
def get_sm_count() -> int:
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        return int(props["multiProcessorCount"])
    except Exception:
        return -1

@dataclass
class RunConfig:
    index_type: str          # "ivfflat" or "ivfpq"
    nlist: int
    nprobe: int
    k: int
    temp_mem_mb: int         # affects queryTileSize indirectly
    use_float16: bool
    pq_m: int
    pq_nbits: int

@dataclass
class RunResult:
    qps: float
    avg_ms: float
    recall_at_k: float

def load_ann_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads ANN-Benchmarks HDF5.
    Expected keys:
      - 'train' : base vectors (float32)
      - 'test'  : query vectors (float32)
      - 'neighbors' : ground-truth neighbor ids for each query (int)
        (usually shape [nq, max_k])
    """
    with h5py.File(path, "r") as f:
        Xb = np.array(f["train"], dtype=np.float32)
        Xq = np.array(f["test"], dtype=np.float32)

        if "neighbors" in f:
            gt = np.array(f["neighbors"], dtype=np.int64)
        else:
            # Some variants name it differently
            raise KeyError("HDF5 missing 'neighbors' key (ground truth).")

    return Xb, Xq, gt

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    """
    Standard ANN recall@k: fraction of queries whose top-1 GT is in retrieved set,
    OR fraction of GT-k items recovered; here we use "set recall" against gt[:,:k].
    """
    gt_k = gt[:, :k]
    # Build sets via vectorized compare (fast enough for typical nq up to 10k)
    # hit if any retrieved id matches any gt id
    hits = 0
    for qi in range(I.shape[0]):
        retrieved = set(I[qi, :k].tolist())
        truth = set(gt_k[qi].tolist())
        if len(retrieved.intersection(truth)) > 0:
            hits += 1
    return hits / I.shape[0]

def build_gpu_index(Xb: np.ndarray, cfg: RunConfig):
    d = Xb.shape[1]
    res = faiss.StandardGpuResources()

    # Important: this influences queryTileSize (temp memory per query)
    res.setTempMemory(cfg.temp_mem_mb * 1024 * 1024)
    co = faiss.GpuClonerOptions()
    co.useFloat16 = cfg.use_float16

    if cfg.index_type == "ivfflat":
        cpu_quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, cfg.nlist, faiss.METRIC_L2)
        cpu_index.train(Xb)
        cpu_index.add(Xb)

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index

    elif cfg.index_type == "ivfpq":
        cpu_quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(
            cpu_quantizer, d, cfg.nlist, cfg.pq_m, cfg.pq_nbits, faiss.METRIC_L2
        )
        cpu_index.train(Xb)
        cpu_index.add(Xb)

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index

    else:
        raise ValueError(f"unknown index_type: {cfg.index_type}")

def run_search(gpu_index, Xq, gt, cfg, batch_size: int,
               warmup: int = 3, iters: int = 10) -> RunResult:
    try:
        gpu_index.nprobe = cfg.nprobe
    except Exception:
        pass

    nq = Xq.shape[0]
    bs = nq if (batch_size is None or batch_size <= 0 or batch_size >= nq) else batch_size

    def one_pass():
        # returns last (D, I) so recall can be computed on full output
        I_all = np.empty((nq, cfg.k), dtype=np.int64)
        # D_all 굳이 안 쓰면 생략 가능
        for s in range(0, nq, bs):
            e = min(s + bs, nq)
            D, I = gpu_index.search(Xq[s:e], cfg.k)
            I_all[s:e] = np.asarray(I, dtype=np.int64)
        return I_all

    # Warmup
    for _ in range(warmup):
        _ = one_pass()

    # Timed runs
    t0 = time.perf_counter()
    I_all = None
    for _ in range(iters):
        I_all = one_pass()
    t1 = time.perf_counter()

    per_iter = (t1 - t0) / iters
    avg_ms = per_iter * 1000.0
    qps = nq / per_iter

    r = recall_at_k(I_all, gt, cfg.k)
    return RunResult(qps=qps, avg_ms=avg_ms, recall_at_k=r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="./sift-128-euclidean.hdf5", help="ANN-Benchmarks .hdf5 path (e.g., sift-128-euclidean.hdf5)")
    ap.add_argument("--index", choices=["ivfflat", "ivfpq"], default="ivfpq")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--pq_m", type=int, default=16)
    ap.add_argument("--pq_nbits", type=int, default=8)
    ap.add_argument("--use_float16", action="store_true")

    ap.add_argument("--k_list", type=str, default="10,32,64,128,256")
    ap.add_argument("--nprobe_list", type=str, default="128,256,512")
    ap.add_argument("--temp_mem_mb_list", type=str, default="2048")  # make tile small/large by restricting temp

    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)

    ap.add_argument("--batch_size", type=int, default=None,
                    help="queries per search() call. <=0 means use full Xq.")
    ap.add_argument("--repeat_q", type=int, default=1,
                    help="repeat queries to increase nq (stress throughput/cache).")

    args = ap.parse_args()

    Xb, Xq, gt = load_ann_hdf5(args.hdf5)

    if args.repeat_q > 1:
        Xq = np.repeat(Xq, args.repeat_q, axis=0)
        gt = np.repeat(gt, args.repeat_q, axis=0)
    print(f"[data] Xb={Xb.shape} Xq={Xq.shape} gt={gt.shape}")
    print(f"[gpu] SM count={get_sm_count()} (if -1, cupy not available)")

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    nprobe_list = [int(x) for x in args.nprobe_list.split(",") if x.strip()]
    temp_list = [int(x) for x in args.temp_mem_mb_list.split(",") if x.strip()]

    # Build once per temp memory setting (because tile depends on temp)
    all_results: Dict[str, RunResult] = {}

    for temp_mb in temp_list:
        cfg0 = RunConfig(
            index_type=args.index,
            nlist=args.nlist,
            nprobe=nprobe_list[0],
            k=k_list[0],
            temp_mem_mb=temp_mb,
            use_float16=args.use_float16,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
        )

        print(f"\n[build] index={cfg0.index_type} nlist={cfg0.nlist} temp_mem_mb={temp_mb}")
        gpu_index = build_gpu_index(Xb, cfg0)

        for k in k_list:
            for nprobe in nprobe_list:
                cfg = RunConfig(**{**cfg0.__dict__, "k": k, "nprobe": nprobe})
                res = run_search(
                    gpu_index, Xq, gt, cfg,
                    warmup=args.warmup, iters=args.iters, batch_size=args.batch_size
                )
                key = f"temp={temp_mb}MB k={k} nprobe={nprobe}"
                all_results[key] = res
                print(f"[run] {key:28s}  QPS={res.qps:10.1f}  avg={res.avg_ms:7.3f}ms  recall@{k}={res.recall_at_k:.4f}")

    print("\n[done] Tip: run this script once with baseline FAISS-GPU, and once with patched FAISS-GPU, then diff logs.")

if __name__ == "__main__":
    main()