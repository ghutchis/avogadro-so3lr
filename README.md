# avogadro-so3lr

Energy models for Avogadro 2 using the SO3LR machine-learned potentials
(SO3krates with learned long-range electrostatics and dispersion), run through
the upstream [so3lr](https://github.com/general-molecular-simulations/so3lr)
ASE calculator on JAX.

## Models

| Name | Upstream model | Elements | Charge / spin | Unit cells | Weights license |
| --- | --- | --- | --- | --- | --- |
| SO3LR-Tauri-M | `so3lr-tauri-m-2606` | H to At (Z = 1-85) | yes / yes | yes | CC BY-NC 4.0 (non-commercial) |
| SO3LR | `so3lr` (original, JACS 2025) | H, C, N, O, F, P, S, Cl | yes / yes | no | MIT |

Weights ship inside the `so3lr` package and are selected by name, so there is
nothing to download. The Tauri model is pinned to a dated release so a new
upstream release never changes your results silently.

The original SO3LR was trained on closed-shell biomolecular chemistry; treat
open-shell results from it with care. Total charge and spin multiplicity are
passed from Avogadro to the model.

### Long-range settings

- Gas phase: `lr_cutoff = 100` Angstrom, as upstream suggests.
- Periodic (SO3LR-Tauri-M only): PME electrostatics with a 10 Angstrom
  real-space cutoff, 2 Angstrom smearing and 1.55 Angstrom mesh, the upstream
  command line's periodic defaults. Dispersion is summed in real space to 10
  Angstrom (the ASE calculator has no k-space dispersion). K-space is untested
  upstream for the original SO3LR, so it is not offered for unit cells.

Calculations use float64 on the CPU (upstream "mixed" precision on a GPU).

## Installation notes

`so3lr` is not on PyPI, so it is installed from a pinned upstream commit, along
with its own git-hosted dependencies (glp, jax-md, jax-pme, jraph). This needs
`git` on the machine at install time. Python 3.12 or 3.13 is required.
Platforms: osx-arm64 and linux-64 lock; only osx-arm64 has been run. Windows
is untested and not listed.

JAX runs on the CPU. For an NVIDIA GPU on Linux, install the upstream CUDA
extra into the plugin environment (`pixi run pip install "jax[cuda12]"`);
this is untested here.

First use is slow: importing JAX and compiling the model takes roughly 7-11
seconds per process. `pixi run setup` (run by Avogadro after installing)
compiles the models once and fills a persistent JAX cache in
`~/.cache/avogadro-so3lr/jax`. Moving atoms does not recompile.

## Development

```bash
pixi install
pixi run setup
pixi run test
```

Avogadro starts the model as
`pixi run avogadro-so3lr SO3LR-Tauri-M --protocol binary-v1`. The entry point
duplicates file descriptor 1 for the protocol and redirects fd 1 to stderr
before importing JAX, so library output can never corrupt a frame.

## Licenses

- This plugin: BSD-3-Clause.
- SO3LR code: MIT (parts Apache-2.0).
- Original SO3LR weights (`so3lr`): MIT.
- SO3LR Tauri weights (`so3lr-tauri-*`): CC BY-NC 4.0, non-commercial use only.

## Citation

- Kabylda et al., "Molecular Simulations with a Pretrained Neural Network and
  Universal Pairwise Force Fields", J. Am. Chem. Soc. 2025,
  https://doi.org/10.1021/jacs.5c09558 (SO3LR).
- SO3LR Tauri preprint, https://doi.org/10.48550/arXiv.2611.06237.
