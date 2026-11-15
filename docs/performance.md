# Memory and speed

RAM and time of each step, by scan size. Both grow with the number of voxels: half the voxels, about half the memory and time. A scan of 1,024 × 1,024 × 1,024 has about 1,000 million voxels.

## By step

An 8-bit scan of about 1,000 million voxels:

| Step | RAM | Time | More threads |
|---|---|---|---|
| Seed generation | 4 GB + 3 GB per thread | 2 min (4 threads) | Faster, more RAM |
| Adaptive seed, erosion | 8 GB | 6 min | No effect |
| Adaptive seed, thresholds | 9 GB | 10 min | No effect |
| Sweep adaptive seed | 15 GB, 20 GB with 8 threads | 1 min (8 threads) | Faster, more RAM |
| Growth | 9 GB | 4 min (8 threads) | Faster |

More than 8 threads made no step faster. The default (half the CPU cores) is fine unless RAM is short.

## 16-bit and noisy scans

They need more: a 16-bit image is twice as large, and a step with more than 65,535 regions (noise often gives that) needs 32-bit labels. A 16-bit, noisy scan of 970 million voxels, 4 threads:

| Step | RAM | Time |
|---|---|---|
| Seed generation | 29 GB | 2 min |
| Adaptive seed | 10–12 GB | 6–11 min |
| Sweep adaptive seed | 26 GB | 1.5 min |
| Growth | 10 GB | 3 min |

## Large scans

An 8-bit scan of 3,900 million voxels, 8 threads:

| Step | RAM | Time |
|---|---|---|
| Sweep adaptive seed, `memory_limit_gb: 32` | 31 GB | 10 min |
| Growth | 35 GB | 8 min |

## When RAM is short

The run becomes very slow or stops with a memory error. Then:

- Use fewer threads (`num_threads`, or Threads in napari): this helps seed generation and sweep.
- Sweep: set `memory_limit_gb` (slower, the same result); see [Large images](config_sweep_adaptive_seed.md#large-images).
- Crop the scan to the part you need.

## GPU

Not included here. For nnInteractive or SAM, see their docs: [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2).

## How this was measured

Intel Xeon W-2295 (18 cores), 128 GB RAM, Windows 11, SPROUT 1.1.1. Scans: the aardvark skull of the paper (8-bit), a spectacled bear (16-bit) and an ant (8-bit), with the settings of the paper (Supplementary Table 8): 3 erosions or 6 thresholds for adaptive seed, 6 steps for sweep, 40 dilations for growth. RAM is the peak of the run. More thresholds, erosions or dilations take longer.
