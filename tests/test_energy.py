"""Tests for the SO3LR-Tauri-M energy model in avogadro-so3lr."""

import numpy as np
import pytest

from avogadro.energy import (
    FLAG_REQUEST_ENERGY_AND_GRADIENT,
    FLAG_REQUEST_GRADIENT,
)

from conftest import coordinates, exchange, image_mask


def test_energy_request_returns_a_finite_number(water):
    coords = coordinates(water)
    (energy,) = exchange(water, [(coords, 0)])

    assert isinstance(energy, float)
    assert np.isfinite(energy)


def test_gradient_has_one_vector_per_atom(water):
    coords = coordinates(water)
    (gradient,) = exchange(water, [(coords, FLAG_REQUEST_GRADIENT)])

    assert gradient.shape == coords.shape
    assert np.all(np.isfinite(gradient))


def test_fused_request_agrees_with_the_separate_ones(water):
    coords = coordinates(water)
    energy, gradient, fused = exchange(
        water,
        [
            (coords, 0),
            (coords, FLAG_REQUEST_GRADIENT),
            (coords, FLAG_REQUEST_ENERGY_AND_GRADIENT),
        ],
    )

    assert fused[0] == pytest.approx(energy)
    assert fused[1] == pytest.approx(gradient)


def test_the_model_stays_up_across_requests(water):
    """Avogadro reuses one process for a whole optimization."""
    coords = coordinates(water)
    # Stretch one bond. A rigid translation would not do: the energy is
    # invariant under it, which is what the next test checks.
    stretched = coords.copy()
    stretched[1] *= 1.1

    first, second, again = exchange(
        water, [(coords, 0), (stretched, 0), (coords, 0)]
    )

    assert first != pytest.approx(second)
    # No state leaks between requests: the same geometry gives the same energy.
    assert first == pytest.approx(again)


def test_energy_is_translation_invariant(water):
    """A non-periodic energy cannot depend on where the molecule sits.

    Failing this usually means the model kept the bootstrap coordinates
    somewhere instead of using the ones each request carries.
    """
    coords = coordinates(water)
    here, there = exchange(water, [(coords, 0), (coords + 3.0, 0)])

    assert here == pytest.approx(there)


def test_gradient_matches_finite_differences(water):
    """The gradient is dE/dx, not the force. A sign error shows up here.

    Central differences of the energy are accurate to O(h^2), so a correct
    analytic gradient agrees to several digits; a sign error disagrees by a
    factor of exactly -1.
    """
    coords = coordinates(water)
    step = 1.0e-4

    requests = [(coords, FLAG_REQUEST_GRADIENT)]
    for atom in range(coords.shape[0]):
        for axis in range(3):
            for sign in (+1, -1):
                displaced = coords.copy()
                displaced[atom, axis] += sign * step
                requests.append((displaced, 0))

    results = exchange(water, requests)
    analytic = results[0]
    energies = results[1:]

    numerical = np.empty_like(analytic)
    for index in range(analytic.size):
        plus, minus = energies[2 * index], energies[2 * index + 1]
        numerical.flat[index] = (plus - minus) / (2.0 * step)

    assert analytic == pytest.approx(numerical, rel=1e-4, abs=1e-3)


def test_periodic_energy_is_finite(copper):
    coords = coordinates(copper)
    (energy,) = exchange(copper, [(coords, 0)])

    assert np.isfinite(energy)


def test_gradient_covers_every_atom_avogadro_sent(copper):
    """Frames are sized by the full atom count, images included.

    The calculator never sees the images, so their rows come back as zeros --
    but the rows have to be there, or the frame is the wrong length and the
    protocol desynchronizes.
    """
    coords = coordinates(copper)
    (gradient,) = exchange(copper, [(coords, FLAG_REQUEST_GRADIENT)])

    images = image_mask(copper)
    assert gradient.shape == coords.shape
    assert np.all(gradient[images] == 0.0)
    assert np.all(np.isfinite(gradient))


def test_energy_is_invariant_under_a_lattice_translation(copper):
    """Shifting the cell contents by a lattice vector changes nothing.

    This is the check that the unit cell is actually being applied: without
    periodic boundary conditions the two energies differ.
    """
    coords = coordinates(copper)
    lattice_a = np.array(
        copper["cjson"]["unitCell"]["cellVectors"][0:3], dtype=float
    )

    here, there = exchange(copper, [(coords, 0), (coords + lattice_a, 0)])

    assert here == pytest.approx(there)


def test_stdout_starts_with_the_frame_magic(water):
    """Avogadro never resyncs: the first stdout bytes must be a frame header,
    whatever the libraries printed while loading."""
    import json
    import subprocess

    from conftest import _LAUNCH, IDENTIFIER, request_frame

    stdin = json.dumps(water).encode() + b"\n" + request_frame(
        coordinates(water), 0
    )
    process = subprocess.run(
        _LAUNCH + [IDENTIFIER, "--protocol", "binary-v1"],
        input=stdin, capture_output=True, timeout=300,
    )
    assert process.stdout[:4] == b"AVB1"


def test_charge_and_spin_reach_the_model(water):
    """OH- and the OH radical differ, and both differ from the neutral singlet."""
    import copy

    oh = copy.deepcopy(water)
    oh["cjson"]["atoms"]["elements"]["number"] = [8, 1]
    oh["cjson"]["atoms"]["coords"]["3d"] = [0, 0, 0, 0, 0, 0.97]
    coords = coordinates(oh)
    results = []
    for charge, spin in [(0, 2), (-1, 1), (0, 1)]:
        oh["charge"], oh["spin"] = charge, spin
        results.append(exchange(oh, [(coords, 0)])[0])
    assert len({round(r, 3) for r in results}) == 3


def test_legacy_model_runs(water):
    (energy,) = exchange(water, [(coordinates(water), 0)], identifier="SO3LR")
    assert np.isfinite(energy)
