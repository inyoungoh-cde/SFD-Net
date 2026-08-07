# LGD — Local Geometric Descriptor

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Venue-ECCV%202026-1f6feb"></a>
  <a href="https://inyoungoh-cde.github.io/SFD-Net/"><img src="https://img.shields.io/badge/Project%20Page-live-2ea44f"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-cite-555"></a>
</p>

Standalone C++ implementation of the **Local Geometric Descriptor (LGD)** from
*SFD-Net: Sharp Feature Detection Network Based on Local Geometric Features*
(ECCV 2026). LGD turns each point of a cloud into a compact three-value cue
describing how the surface-normal field varies around it at three nested
scales — low on planar patches, high near creases and corners.

In SFD-Net the descriptor is precomputed with this tool, concatenated with the
raw coordinates, and fed to a point-cloud backbone. Because it is
**architecture-agnostic**, it can be prepended unmodified to any network that
accepts per-point channels; in the paper it improves three independent
backbones in a plug-and-play way. This directory contains only the descriptor
extractor — see the [repository root](../) for the network, training code and
results.

The tool is deliberately dependency-light: nothing beyond the C++ standard
library and OpenMP is required at runtime, and it builds on both Windows and
Linux.

## Quick start

**Windows, no build required.** Download `LGD_portable.zip` from
[Releases](https://github.com/inyoungoh-cde/SFD-Net/releases), extract it, put
your `.txt` files in `data\`, and double-click `LGD.exe`.

**From source** — see [BUILD.md](BUILD.md):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/LGD data output 40
```

On Windows you can instead open `LGD.sln` (Visual Studio 2017) and press
Ctrl+F5.

## Usage

```
LGD [input_dir] [output_dir] [k]
```

| Argument | Default | Meaning |
|---|---|---|
| `input_dir` | `data` | folder containing the input `.txt` files |
| `output_dir` | `output_YYMMDD` | folder for the results (created if needed) |
| `k` | `40` | k-NN neighborhood size |

Run with no arguments for an interactive prompt where Enter accepts each
default. Argument order is flexible: a purely numeric argument is read as `k`,
the first non-numeric one as the input folder and the second as the output
folder. Relative paths resolve against the project root; absolute paths are
used as given.

## Input and output format

Input: one point per line, whitespace-separated, at least three columns.

```
x y z [extra columns ...] L
```

The first three columns are the coordinates. If there are more, the **last**
column is treated as a label and carried through to the output. All lines in a
file must have the same number of columns.

Output: one file per input file, same name, one line per point:

```
x y z f0 f1 f2 L
```

`f0`, `f1`, `f2` are the descriptor channels computed at the small, medium and
large scale. `L` is present only if the input had a label column.

## How it works

Neighbors are retrieved with a kd-tree (nanoflann). Since the query point is
returned among its own neighbors, the search uses `k+1` so that `k` real
neighbors remain. Three nested neighborhoods of roughly `k/2`, `k` and `2k`
points are then processed identically — about 20 / 40 / 80 with the default
`k = 40`:

1. **Multi-scale PCA normals.** For each scale, PCA over the neighborhood; the
   eigenvector of the smallest eigenvalue is the unit normal. Only directions
   are kept — no orientation propagation — which makes the descriptor invariant
   to normal sign flips.
2. **Second moments of normal differences.** Weighted outer products of the
   normal differences `Δn Δnᵀ` between the center point and its neighbors are
   accumulated per scale. Building the descriptor from `Δn Δnᵀ` makes it
   invariant to rigid motion and global scale with no explicit alignment.
3. **Per-scale isotropy index.** Each scale's tensor is collapsed to a single
   value, `1 - λ_max / Σλ`, and the three are concatenated into `φ ∈ ℝ³`.

The scales are complementary: the smallest gives acuity on narrow creases, the
middle balances localization and stability, and the largest suppresses variance
from noise and undersampling.

Files are processed in parallel with OpenMP, and so is the per-point loop
within each file.

## Requirements

* A C++17 compiler, or Visual Studio 2017 (which uses the pre-standard
  `std::experimental::filesystem`; handled automatically).
* OpenMP — bundled with MSVC, and with GCC on Linux.

Eigen and nanoflann are header-only and already vendored in
`third_party_includes/`, so there is nothing to install.

## Directory layout

```
main.cpp                 CLI, file discovery and parsing
runner.cpp / runner.h    per-file parallel driver
algorithms.h             the method itself (estimator class)
FileData.h               per-file data container
platform.h               Windows/Linux shims
CMakeLists.txt           cross-platform build
LGD.sln / LGD.vcxproj    Visual Studio 2017 build
make_package.bat         builds the redistributable Windows package
package/                 files copied into that package
data/                    sample input
```

`x64/`, `build/`, `.vs/` and `deploy/` are generated and excluded by
`.gitignore`.

## Third-party code

| Library | Author | License |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | B. Jacob, G. Guennebaud et al. | MPL2 |
| [nanoflann](https://github.com/jlblancoc/nanoflann) | J. L. Blanco-Claraco (from FLANN by M. Muja, D. Lowe) | BSD |

Both are redistributed unmodified with their license headers intact.

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

Released under the [MIT License](LICENSE). The vendored third-party libraries
keep their own licenses, listed above.
