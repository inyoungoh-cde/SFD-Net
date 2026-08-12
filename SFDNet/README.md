# SFD-Net — Sharp Feature Detection Network

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Venue-ECCV%202026-1f6feb"></a>
  <a href="https://inyoungoh-cde.github.io/SFD-Net/"><img src="https://img.shields.io/badge/Project%20Page-live-2ea44f"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-cite-555"></a>
</p>

Training and inference code for **SFD-Net: Sharp Feature Detection Network
Based on Local Geometric Features** (ECCV 2026). Given a raw point cloud — no
mesh, no connectivity, no supplied normals — the network labels every point as
lying on a sharp feature or not.

Detection runs in two stages. First, the **Local Geometric Descriptor (LGD)**
turns each point into three values summarising how the surface-normal field
varies around it at three nested scales. Then those three channels are
concatenated with the raw coordinates and fed to an enhanced PointNet++
backbone trained with a class-imbalance loss. The descriptor is computed once,
offline, by the standalone extractor in [`LGD/`](../LGD) — this
directory contains only the network.

Sharp points are rare, typically 4–13% of a cloud, so the split matters: a
model that answered "nothing is sharp" everywhere would still score about 95%
raw accuracy. All numbers below are reported the way the paper reports them.

## Quick start

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Then build the CUDA extension (see [below](#cuda-extension) for why it needs
its own step), and run inference against a trained checkpoint:

```bash
.venv/bin/python test.py --log_dir noise_none --data_dir dataset/ABC/noise_none \
    --normal --model SFDNet_cuda --gpu 0
```

## Installation

| | |
|---|---|
| Python | 3.11 |
| PyTorch | 2.4.1, CUDA 12.4 build |
| GPU | required — the default backbone uses a CUDA extension |
| CUDA toolkit | required at install time only, to compile that extension |

`requirements.txt` pins the exact versions the published results were produced
with, and installs with no extra flags under either `uv` or `pip`. It names the
PyTorch wheel by URL instead of adding `--extra-index-url`, because that index
also mirrors `numpy`, `tqdm` and `setuptools` at versions older than the pins
here, which trips resolvers that restrict a package to the first index carrying
it. On Windows, or a different Python version, substitute the matching wheel
from the [cu124 index](https://download.pytorch.org/whl/cu124/torch/) —
`cp311` is Python 3.11.

### CUDA extension

The default model, `SFDNet_cuda`, calls into `pointnet2_ops` for furthest-point
sampling, ball query and grouping. That package compiles against the PyTorch
installed above, so it cannot be resolved from `requirements.txt` in the same
pass — install it second, with build isolation off:

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib

# Its setup.py hard-codes an architecture list that starts at sm_37, which
# CUDA 12 removed; leave it as is and nvcc fails. Make the list overridable:
sed -i 's/^os\.environ\["TORCH_CUDA_ARCH_LIST"\].*/os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")/' setup.py

# 8.0 = A100. Use 8.6 for A40/3090, 8.9 for L40S/4090, 9.0 for H100.
TORCH_CUDA_ARCH_LIST="8.0" uv pip install --no-build-isolation .
```

The build takes roughly a minute and needs `nvcc` on `PATH`.

**Prefer to skip it?** `models/SFDNet.py` is the same architecture backed by
pure-PyTorch operators in `models/pointnet2_utils.py`, with no extension to
compile. Pass `--model SFDNet` instead. It is slower, and the released
checkpoints were trained with the CUDA path.

## Data

### Where to put it

`--data_dir` points at **one condition** — a flat directory of point clouds
plus the split lists that index them. Nothing above that path is hard-coded, so
the tree below is a convention, not a requirement:

```
dataset/
└── ABC/
    ├── noise_none/                 <- --data_dir dataset/ABC/noise_none
    │   ├── 0000.txt
    │   ├── 0009.txt
    │   ├── ...                     (1000 clouds)
    │   └── train_test_split/
    │       ├── shuffled_train_file_list.json
    │       ├── shuffled_val_file_list.json
    │       └── shuffled_test_file_list.json
    ├── noise_small/                same 1000 models, σ = 0.12%
    ├── noise_medium/               same 1000 models, σ = 0.6%
    └── noise_large/                same 1000 models, σ = 1.2%
```

The four conditions hold the same models under increasing perturbation —
density variation plus Gaussian noise following the PCPNet protocol, at 0.12%,
0.6% and 1.2% of the bounding-box diagonal — so only the point data differs.
Each is trained and evaluated separately, which is why every command pairs a
`--data_dir` with a matching `--log_dir`.

### File format

Each `.txt` holds one point per line, seven whitespace-separated columns, up to
10,000 points per model:

```
x y z  w0 w1 w2  L
```

`w0 w1 w2` are the LGD channels at the small, medium and large scale — exactly
the `f0 f1 f2` columns written by [`LGD/`](../LGD), which also
carries the label column straight through. `L` is 0 for a non-sharp point and 1
for a sharp one.

Each split list is a flat JSON array of model ids, without directory prefix and
without the `.txt` suffix, resolved against the condition directory:

```json
["4193", "6968", "2202"]
```

The ABC split used in the paper is 400 train / 90 validation / 510 test.

### Bringing your own clouds

Write each cloud as `x y z L`, run the LGD extractor over the folder, and the
tool emits `x y z f0 f1 f2 L` — already the layout above. Then drop a
`train_test_split/` next to the results with the three JSON lists. Coordinates
are centred and scaled to the unit sphere at load time, and the LGD channels are
used as written, since the descriptor is already invariant to similarity
transforms — so no further normalisation is needed.

## Training

```bash
python train.py --log_dir my_run --data_dir dataset/ABC/noise_none --normal --gpu 0
```

`--normal` switches on the extra input channels and `--add_channel 3` says how
many there are — both are required for LGD input, and omitting `--normal` trains
on coordinates alone.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `SFDNet_cuda` | model module in `models/` |
| `--log_dir` | timestamp | run directory under `log/sem_seg/` |
| `--data_dir` | `dataset/ABC/noise_none` | dataset directory for one condition |
| `--normal` | off | use the extra input channels |
| `--add_channel` | `3` | number of extra channels (LGD) |
| `--npoint` | `500` | points per training sample |
| `--batch_size` | `16` | samples per step |
| `--epoch` | `200` | epochs |
| `--learning_rate` | `0.001` | initial LR |
| `--ftl_alpha` / `--ftl_beta` / `--ftl_gamma` | `0.8` / `0.2` / `1.5` | Focal Tversky parameters |

Each run writes to `log/sem_seg/<log_dir>/`:

```
checkpoints/best_f1_model.pth     best macro F1 on the validation split
checkpoints/best_acc_model.pth    best macro accuracy
checkpoints/best_fpr_model.pth    lowest false-positive rate
checkpoints/best_loss_model.pth   lowest validation loss
checkpoints/model.pth             periodic snapshot
logs/<model>.txt                  training log
<model>.py, pointnet2_utils*.py   architecture archived at launch time
```

The architecture is archived into the run directory so that evaluation always
rebuilds the network the checkpoint was trained with, even if `models/` moves on
afterwards.

## Inference

```bash
python test.py --log_dir noise_none --data_dir dataset/ABC/noise_none \
    --normal --model SFDNet_cuda --gpu 0 --save_pc
```

Evaluation runs on whole clouds. Each cloud is randomly permuted with a fixed
seed, split into chunks of `--npoint` points, and pushed through the network in
mini-batches of `--batch_size` chunks; predictions are scattered back to the
original point indices, so every point is predicted exactly once. The chunk size
matters — the set-abstraction radii and sampling counts are tuned for the
density the model was trained at, so leave `--npoint` at the training value.

| Flag | Default | Meaning |
|---|---|---|
| `--log_dir` | required | run directory under `log/sem_seg/` |
| `--eval_metric` | `f1` | which checkpoint to load: `f1`, `acc`, `loss`, `fpr` |
| `--save_pc` | off | write per-point predictions |
| `--seed` | `42` | seed for the deterministic chunking |
| `--batch_size` | `64` | chunks per forward |

With `--save_pc`, each cloud gets a file under `predicted_results/` named after
its source, five columns per line and index-aligned with the input:

```
x y z  pred  L
```

Metrics land in `log/sem_seg/<log_dir>/eval.txt`.

### Reading the metric

`Average F1` — the figure reported in the paper — is the **arithmetic mean of
the two per-class F1 scores**, not the F1 of the sharp class. The two are far
apart, so it is worth being explicit:

```
Class 0 (non-sharp)  F1 = 0.9176
Class 1 (sharp)      F1 = 0.3078
Average              F1 = 0.6127     <- the reported number
```

The same convention applies to the accuracy, precision, recall and FPR printed
on the `Average Metrics` line.

## Results

ABC test split, 510 models, `best_f1_model.pth` for each noise condition.
Reproduced values come from running the commands above on the
[released checkpoints](https://github.com/inyoungoh-cde/SFD-Net/releases/tag/sfdnet-v1.0)
— unzip `SFDNet_checkpoints.zip` in this directory and the four run folders
land under `log/sem_seg/` ready for `test.py`. The residual gap is the random
chunk permutation.

| Condition | Paper F1 | Reproduced F1 | Paper FPR | Reproduced FPR |
|---|---:|---:|---:|---:|
| no noise | 61.33 | **61.27** | 36.93 | **37.10** |
| 0.12% | 60.42 | **60.32** | 39.76 | **39.99** |
| 0.6% | 58.62 | **58.53** | 40.88 | **41.08** |
| 1.2% | 57.05 | **57.12** | 41.94 | **41.79** |

All four land within 0.1 percentage points of the published table.

## How it works

The backbone is PointNet++ with multi-scale grouping — four set-abstraction
levels sampling 1024 / 256 / 64 / 16 centroids, then four feature-propagation
levels back to full resolution — with two transformer encoders inserted into the
path: one on the 96-channel features after the first set-abstraction level
(2 layers, 6 heads) and one on the 256-channel features after the first
feature-propagation level (2 layers, 8 heads). The head is a 1-D convolution,
dropout, and a two-way classifier.

Because sharp points are a small minority, plain cross-entropy collapses onto
the majority class. Training instead minimises a weighted sum of a Focal Tversky
loss and a Dice loss (`w = 0.7` on the former), with the Tversky α/β pair
controlling how false positives and false negatives trade against each other.

For how the LGD channels themselves are computed — nested k-NN scales, PCA
normals kept as unoriented directions, second moments of normal differences,
and the per-scale isotropy index — see the
[LGD README](../LGD/README.md).

## Directory layout

```
train.py                 training loop, validation, checkpoint selection
test.py                  whole-cloud chunked evaluation
provider.py              point-cloud augmentation helpers
data_utils/dataloader.py dataset; whole-cloud or resampled, one class
models/SFDNet_cuda.py    the model — CUDA operators (default)
models/SFDNet.py         the same model — pure-PyTorch operators
models/pointnet2_utils_cuda.py   set abstraction / feature propagation, CUDA
models/pointnet2_utils.py        the same, pure PyTorch
requirements.txt         pinned runtime dependencies
../LGD/                  standalone LGD extractor (C++), with its own README
```

`log/` is created on first run and holds checkpoints, logs and predictions.

## Third-party code

| Project | Author | License |
|---|---|---|
| [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | yanx27 | MIT |
| [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) | E. Wijmans | Unlicense |

The set-abstraction and feature-propagation modules, the augmentation helpers
and the training scaffolding derive from the first; the `pointnet2_ops` CUDA
extension is installed from the second. The underlying method is PointNet++
(Qi et al., NeurIPS 2017).

## Citation

> Placeholder — to be replaced with the official ECCV 2026 citation once released.

```bibtex
@inproceedings{oh2026sfdnet,
  title     = {SFD-Net: Sharp Feature Detection Network Based on Local Geometric Features},
  author    = {Oh, Inyoung and Ko, Kwang Hee},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). The upstream projects listed above
keep their own licenses, and [`LGD/`](../LGD) carries its own.
