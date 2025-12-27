# Evaluate Faiss-autotune

## Requirements

To run the evaluation, install `h5py`:

```bash
pip install h5py
```

## Download dataset

```bash
./download_data.sh
```
If you want to evaluate with more data, download hdf5 file from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file). 

## Evaluation
###  Example
```bash
python eval.py \
  --hdf5 sift-128-euclidean.hdf5 \
  --index ivfpq \
  --nlist 1024 \
  --pq_m 16 \
  --pq_nbits 8 \
  --nprobe_list 128,256 \
  --k_list 10,100 \
  --iters 10
```

### Command-line Arguments

#### Dataset & Index

- `--hdf5`  
  ANN-Benchmarks `.hdf5` dataset path  
  (e.g., `sift-128-euclidean.hdf5`)

- `--index`  
  IVF index type  
  - `ivfflat`
  - `ivfpq`  
  (default: `ivfpq`)

- `--nlist`  
  Number of IVF clusters  
  (default: `1024`)

---

#### Product Quantization (IVFPQ)

- `--pq_m`  
  Number of PQ sub-vectors  
  (default: `16`)

- `--pq_nbits`  
  Bits per PQ code  
  (default: `8`)

- `--use_float16`  
  Enable FP16 on GPU

---

#### Search Parameters

- `--k_list`  
  Comma-separated top-k values  
  (default: `10,32,64,128,256`)

- `--nprobe_list`  
  Comma-separated `nprobe` values  
  (default: `128,256,512`)

- `--temp_mem_mb_list`  
  Temporary GPU memory limits (MB)  
  (default: `2048`)

---

#### Benchmark Control

- `--iters`  
  Number of iterations  
  (default: `10`)

- `--warmup`  
  Number of warm-up iterations  
  (default: `3`)

---

#### Query Batching

- `--batch_size`  
  Queries per `search()` call  
  (`<=0` means use all queries)

- `--repeat_q`  
  Repeat query set for throughput stress  
  (default: `1`)
