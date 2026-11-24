# Edward Su (蘇胤瑞)

**Assistant Researcher** at Taiwan's Central Weather Administration. I deploy, verify and speed up AI weather models
(NVIDIA FourCastNet v3, CorrDiff, Earth-2 Atlas, ECMWF AIFS) on HPC GPU clusters, from offline installation to
Nsight profiling, multi-GPU scaling and probabilistic verification.

[littleblacker29@gmail.com](mailto:littleblacker29@gmail.com) ·
[LinkedIn](https://www.linkedin.com/in/%E8%83%A4%E7%91%9E-%E8%98%87-00a893369/)

## Featured projects

### [Earth2Studio Taiwan](https://github.com/yinruei/earth2studio-taiwan): NVIDIA Earth-2 on an air-gapped A100 cluster

FourCastNet v3 global forecasts downscaled over Taiwan by CorrDiff (~2 km), deployed fully offline and
benchmarked with Typhoon Kong-rey (2024).

- Ran the whole stack without internet access, root or a modern system compiler; documented 13 deployment pitfalls
  and their fixes.
- Nsight Systems showed CorrDiff is memory-bound (68% of GPU time in element-wise, GroupNorm and SiLU kernels);
  `torch.compile` kernel fusion made it **1.71x faster** and closer to FP32.
- BF16 made FCN3 **1.37x faster** with the typhoon centre on the same grid point at every lead, but only 15% end to
  end, which is why the next target was CorrDiff.
- A 16-member ensemble on 4 A100s: **3.40x** in the compute phase with bitwise-identical outputs. Moving the Python
  environment off the parallel file system cut start-up from **280 s to 32 s**.
- PhysicsNeMo AFNO trained with PyTorch DDP and NCCL over NVLink at **0.98 / 0.93** weak-scaling efficiency
  (FP32 / BF16) on 4 GPUs.

### [Weather GPU Benchmark](https://github.com/yinruei/weather-gpu-benchmark): what GPUs really speed up in forecast verification

Bias, MAE and RMSE with NumPy, PyTorch CPU and CUDA, on synthetic data and real NOAA GFS and ECMWF IFS ENS forecasts.

- At 10 million grid points on an RTX 3050, CUDA computes the metrics **14.8x** faster than NumPy, but only
  **2.2x** once input transfer is included.
- Keeping inputs on the GPU removes most of that transfer cost: **6.4x** faster than re-uploading every time.

## Work

**Assistant Researcher**, Central Weather Administration (CWA), Taiwan · Nov 2020 – present

- **Operational AI downscaling:** built a daily pipeline that downscales 50-member, 45-day FCN3 global ensemble
  forecasts over Taiwan with CorrDiff on two A100s, in about 22 minutes per run.
- **Probabilistic verification of AI vs NWP ensembles:** after CorrDiff downscaling, compared FCN3 with the ECMWF
  extended-range ensemble against the 2 km TReAD reanalysis: weekly precipitation Brier scores with reliability
  decomposition, and 10 m wind CRPS, rank histograms, spread-skill and ROC.
- **Extended-range monsoon onset:** verified 50-member AIFS-ENS and FCN3 ensembles against ERA5 for South China Sea
  summer monsoon and Meiyu onset, presented as a poster at AOGS 2026. Cropping GRIB files with wgrib2 before
  decoding made data loading about 30x faster.
- **NVIDIA Earth-2 Atlas:** deployed the CRPS ensemble model offline on the GPU cluster and ran a 10-day, 8-member
  forecast at 3.9 s per member-step on A100.

**Full-Stack Engineer**, 雲灣資訊, Taiwan · Sep 2018 – Jul 2020

## Skills

| Area | Tools |
|---|---|
| GPU performance | Nsight Systems, NVTX, mixed precision (BF16 / FP16 / TF32), `torch.compile` / Triton |
| Multi-GPU | PyTorch DDP, NCCL, NVLink, ensemble scaling |
| AI weather models | Earth2Studio, PhysicsNeMo, FourCastNet v3, CorrDiff, Earth-2 Atlas, AIFS, Pangu-Weather |
| Verification | CRPS, Brier score and decomposition, reliability, ROC, rank histograms, spread-skill, track error |
| HPC | Air-gapped deployment, Lustre, PJM / Slurm batch jobs, offline pip and conda toolchains, CUDA extension builds |
| Languages and data | Python, PyTorch, NumPy, xarray, Zarr, NetCDF, GRIB, Bash |

## Education

**M.S.**, Graduate Institute of Atmospheric Physics, National Central University, Taiwan · 2015 – 2017
