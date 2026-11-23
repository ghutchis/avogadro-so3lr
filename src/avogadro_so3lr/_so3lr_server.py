#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-so3lr/

"""Shared EnergyServer loop and calculator factory for the SO3LR models.

Avogadro works in kJ/mol and Angstrom and wants the gradient dE/dx, not the
force. SO3LR (through ASE) reports eV and eV/Angstrom.
"""

import json
import os
import sys
from pathlib import Path

# Third-party imports go to stderr while they run: a stray print or warning on
# stdout corrupts the binary stream.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

import numpy as np
from ase import Atoms
from ase.cell import Cell

sys.stdout = _real_stdout
del _real_stdout

from avogadro.energy import EnergyServer

_EV_TO_KJ_MOL = 96.4853321

# identifier -> upstream bundled model name. The Tauri entry is pinned to a
# dated release so a new upstream release never changes results silently.
MODELS = {
    "SO3LR-Tauri-M": {"name": "so3lr-tauri-m-2606", "periodic": True},
    "SO3LR": {"name": "so3lr", "periodic": False},
}

# Gas phase: upstream suggests a 100 Angstrom long-range cutoff.
GAS_PHASE_LR_CUTOFF = 100.0
# Periodic: the upstream command line's defaults (10 Angstrom real-space
# cutoff, PME electrostatics, 2 Angstrom smearing, 1.55 Angstrom mesh).
PERIODIC_KWARGS = dict(
    lr_cutoff=10.0,
    kspace_electrostatics="pme",
    kspace_smearing=2.0,
    kspace_spacing=1.55,
)


def _enable_jax_cache():
    """Persist compiled XLA programs so later starts skip most of the JIT."""
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache:
        root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        cache = str(Path(root) / "avogadro-so3lr" / "jax")
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except Exception as error:  # noqa: BLE001
        print(f"avogadro-so3lr: no JAX compilation cache ({error})",
              file=sys.stderr)


def make_calculator(identifier, periodic=False):
    """Build the upstream ASE calculator for one of our identifiers."""
    _enable_jax_cache()
    from so3lr import So3lrCalculator

    import jax

    name = MODELS[identifier]["name"]
    # float64 throughout on CPU: upstream's default "mixed" mode keeps a
    # float32 model, whose rounding noise (~1e-4 eV/A) hampers optimizers and
    # finite-difference checks. On a GPU, where float64 is slow, keep mixed.
    precision = "mixed" if jax.default_backend() == "gpu" else "float64"
    if periodic:
        return So3lrCalculator(model=name, precision=precision,
                               **PERIODIC_KWARGS)
    return So3lrCalculator(model=name, precision=precision,
                           lr_cutoff=GAS_PHASE_LR_CUTOFF)


def build_atoms(bootstrap, identifier):
    """Return (atoms, keep, num_atoms) from the bootstrap JSON."""
    mol_cjson = bootstrap["cjson"]
    numbers = np.array(mol_cjson["atoms"]["elements"]["number"])
    coords = np.array(
        mol_cjson["atoms"]["coords"]["3d"], dtype=float
    ).reshape(-1, 3)
    # num_atoms is the full count in every frame, periodic images included.
    num_atoms = len(numbers)
    keep = np.ones(num_atoms, dtype=bool)

    periodic = "unitCell" in mol_cjson and MODELS[identifier]["periodic"]
    if periodic:
        fractional = mol_cjson["atoms"]["coords"].get("3dFractional", [])
        if fractional:
            frac = np.array(fractional, dtype=float).reshape(-1, 3)
            keep = ~np.any(np.abs(frac - 1.0) < 1e-4, axis=1)
        lattice = np.array(
            mol_cjson["unitCell"]["cellVectors"], dtype=float
        ).reshape(3, 3)
        atoms = Atoms(numbers[keep], coords[keep], cell=Cell(lattice),
                      pbc=True)
    else:
        atoms = Atoms(numbers, coords, pbc=False)

    # Top-level bootstrap keys: charge, and spin as a multiplicity (2S+1).
    atoms.info["charge"] = float(bootstrap.get("charge", 0))
    atoms.info["multiplicity"] = int(bootstrap.get("spin", 1))
    atoms.calc = make_calculator(identifier, periodic)
    return atoms, keep, num_atoms


def _evaluate(atoms, keep, num_atoms, coords, want_gradient):
    atoms.set_positions(coords[keep])
    energy = float(atoms.get_potential_energy()) * _EV_TO_KJ_MOL
    if not want_gradient:
        return energy, None
    gradient = np.zeros((num_atoms, 3), dtype=np.float64)
    gradient[keep] = atoms.get_forces() * -_EV_TO_KJ_MOL
    return energy, gradient


def run(identifier, proto_out=None):
    if proto_out is None:
        proto_out = sys.stdout.buffer
    # sys.stdin.buffer, never sys.stdin: the text wrapper swallows frames.
    bootstrap = json.loads(sys.stdin.buffer.readline())
    atoms, keep, num_atoms = build_atoms(bootstrap, identifier)

    with EnergyServer(sys.stdin.buffer, proto_out, num_atoms) as server:
        for request in server.requests():
            try:
                if request.wants_energy_and_gradient:
                    energy, gradient = _evaluate(
                        atoms, keep, num_atoms, request.coords, True)
                    request.send_energy_and_gradient(energy, gradient)
                elif request.wants_gradient:
                    _, gradient = _evaluate(
                        atoms, keep, num_atoms, request.coords, True)
                    request.send_gradient(gradient)
                else:
                    energy, _ = _evaluate(
                        atoms, keep, num_atoms, request.coords, False)
                    request.send_energy(energy)
            except Exception as error:  # noqa: BLE001
                request.send_error(str(error))
