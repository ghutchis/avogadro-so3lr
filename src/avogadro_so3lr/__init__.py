#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-so3lr/

"""Entry point for the avogadro-so3lr plugin.

Avogadro calls this as:
    avogadro-so3lr <identifier> --protocol binary-v1 [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin as one compact line, followed by
binary coordinate frames for as long as the model stays loaded.
"""

import argparse
import os


def setup():
    """Compile both models once so the JAX cache is warm for first use.

    The weights ship inside the so3lr package, so nothing is downloaded.
    """
    from ase.build import molecule

    from ._so3lr_server import MODELS, make_calculator

    for identifier in MODELS:
        print(f"Preparing {identifier}...", flush=True)
        atoms = molecule("H2O")
        atoms.info["charge"] = 0
        atoms.calc = make_calculator(identifier)
        atoms.get_forces()
    print("SO3LR setup complete.")


def main():
    parser = argparse.ArgumentParser("avogadro-so3lr")
    # Avogadro always passes the feature identifier as the first argument.
    parser.add_argument("feature")
    # Every plugin must accept these without crashing, even if it ignores them.
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    # Sent for any energy model declaring protocol = "binary-v1".
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    args = parser.parse_args()

    # Handlers are imported inside each branch so that a slow dependency is
    # only loaded by the feature that actually needs it.
    # Protect the protocol stream at the file-descriptor level, before jax or
    # so3lr is imported: keep a private copy of the real stdout for frames,
    # then point fd 1 at stderr so anything printed by Python, C or XLA code
    # (which bypasses sys.stdout) can never corrupt a frame.
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)

    from ._so3lr_server import MODELS, run

    if args.feature not in MODELS:
        raise SystemExit(f"Unknown model: {args.feature}")
    run(args.feature, proto_out)
