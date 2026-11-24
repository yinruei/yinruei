# Yin-Ruei (Edward) Su · 蘇胤瑞

**Assistant Researcher** at IISI, working at Taiwan's Central Weather Administration. I deploy, verify and speed up AI weather models
(NVIDIA FourCastNet v3, CorrDiff, Earth-2 Atlas, ECMWF AIFS) on HPC GPU clusters, from offline installation to
Nsight profiling, multi-GPU scaling and probabilistic verification.

[littleblacker29@gmail.com](mailto:littleblacker29@gmail.com) ·
[LinkedIn](https://www.linkedin.com/in/yin-ruei-su-00a893369/)

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

**Assistant Researcher**, International Integrated Systems (IISI), on assignment to Taiwan's Central Weather
Administration (CWA) · Nov 2020 – present

Develop and run the global verification system from 2020; machine-learning weather prediction since 2023.

- **Taipei-1 compute project:** assess the computing environment and compute needs for running ECMWF AIFS-ENS in
  operations, in the project "AIFS-ENS performance optimization in an operational environment".
- **AI downscaling, being introduced into operations:** built a daily pipeline that downscales 50-member, 45-day
  FCN3 global ensemble forecasts over Taiwan with CorrDiff on two A100s, in about 22 minutes per run, and am
  verifying its output before operational use. I also handle CorrDiff's data preprocessing.
- **AI typhoon ensemble forecasting:** set up the environment for an MLWP ensemble typhoon model and its preliminary
  verification (track, intensity, moisture); developed a typhoon vortex initialization method.
- **Probabilistic verification of AI vs NWP ensembles:** after CorrDiff downscaling, compared FCN3 with the ECMWF
  extended-range ensemble against the 2 km TReAD reanalysis: weekly precipitation Brier scores with reliability
  decomposition, and 10 m wind CRPS, rank histograms, spread-skill and ROC. Also compare typhoon tracks across
  NWP and AI models.
- **Extended-range AI ensembles:** ran ECMWF AIFS-ENS and NVIDIA FCN3 in-house as 50-member, 45-day ensembles and
  verified them against ERA5. AIFS-ENS was more skilful in the first two weeks, but its errors grew quickly after a
  crossover around day 16, while FCN3 stayed bounded and, in the 2025 case, caught the end of the Meiyu circulation
  over Taiwan 30 days ahead. First-author poster at AOGS 2026. Cropping GRIB files with wgrib2 before decoding made
  data loading about 30x faster.
- **NVIDIA Earth-2 Atlas:** deployed the CRPS ensemble model offline on the GPU cluster and ran a 10-day, 8-member
  forecast at 3.9 s per member-step on A100.
- **GPU hackathon with NVIDIA mentors:** at the
  [2023 NCHC × NVIDIA OpenACC Hackathon](https://www.nchc.org.tw/Message/MessageView?id=3872), our team rebuilt the
  core of CWA's global ensemble verification system in JAX on GPUs. The existing workflow took over 12 hours per
  cycle; the proof of concept ran up to **44x faster**. I co-developed the GPU code and presented the team's results.
- **Operational systems:** maintain the global verification system (GVER) and other operational systems (GMCP, GNPD,
  cherry-blossom prediction); built long-term charts of forecast scores and ensemble predictability.
- **Projects and communication:** assist a green-energy forecasting project, including RWRF shortwave-radiation bias
  correction with a U-Net for solar power, by compiling monthly and quarterly results and briefing management; give
  public science talks on climate.

**Full-Stack Engineer**, Cloudybay Information (雲灣資訊), Taiwan · Sep 2018 – Jul 2020

- **Full-stack web development (Python, Django, Vue):** built interactive front-end pages and the back-end APIs that
  serve them, from CRUD services to integrations with other systems.
- **Back end and database:** implemented server-side algorithms and tuned database performance.
- Built the website for CWA's weather analysis seminar.

## Publications

- **Su, Y.-R.**, et al. (2026). Evaluation of extended-range ensemble forecasts from probabilistic machine learning
  weather prediction models. Poster AS27-A002, AOGS 2026 Annual Meeting.

## Skills

| Area | Tools |
|---|---|
| GPU performance | Nsight Systems, NVTX, mixed precision (BF16 / FP16 / TF32), `torch.compile` / Triton |
| Multi-GPU | PyTorch DDP, NCCL, NVLink, ensemble scaling |
| AI weather models | Earth2Studio, PhysicsNeMo, FourCastNet v3, CorrDiff, Earth-2 Atlas, AIFS, Pangu-Weather |
| Verification | CRPS, Brier score and decomposition, reliability, ROC, rank histograms, spread-skill, track error |
| HPC | Linux, air-gapped deployment, Lustre, PJM / Slurm batch jobs, offline pip and conda toolchains, CUDA extension builds |
| Languages and data | Python, PyTorch, JAX, NumPy, xarray, Zarr, NetCDF, GRIB, Bash |
| Web development | Django, Vue, back-end APIs, CRUD services, database tuning |

## Education

**M.S.**, Graduate Institute of Atmospheric Physics, National Central University, Taiwan · 2015 – 2017

**B.S.**, Department of Atmospheric Sciences, Chinese Culture University, Taiwan · 2011 – 2015
