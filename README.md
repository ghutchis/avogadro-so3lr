# avogadro-so3lr

Energy models using the SO3LR machine-learned potentials

## Models

| Kind | Name | Notes |
| --- | --- | --- |
| Energy | SO3LR-Tauri-M | Calculate SO3LR-Tauri-M energies and gradients |

`SO3LR-Tauri-M` appears in Avogadro's force field list, and is offered for
molecules built from elements 1-85, with a unit cell, ions allowed, radicals allowed.

## Development

```bash
pixi install
pixi run setup
pixi run test
```

Avogadro starts the model once, as
`pixi run avogadro-so3lr SO3LR-Tauri-M --protocol binary-v1`, sends
one line of bootstrap JSON, and then exchanges binary frames with it for as
long as the model stays selected. The loop ends when Avogadro closes stdin.

That single long-running process is the thing to keep in mind while editing
`src/avogadro_so3lr/so3lr_tauri_m.py`:

- **Nothing may print to stdout.** stdout is the protocol. A `print()` left in
  for debugging, or a dependency that greets you on import, breaks the stream.
  Write to `sys.stderr` instead; `--debug` is there for exactly this.
- **Return the gradient, not the force.** Avogadro wants dE/dx in
  kJ/mol/Angstrom. ASE and most ML potentials give forces, which are its
  negative.
- **Units are kJ/mol and Angstrom.** Hartree, eV and kcal/mol all need
  converting.
- **Answer every request.** On failure send `request.send_error(...)` rather
  than raising, so the process survives to answer the next one.

`pixi run test` drives the real protocol against the real entry point in a
subprocess. It checks the analytic gradient against finite differences of the
energy, which is what catches a sign or unit error before Avogadro does, and it
asserts that stdout carries protocol frames and nothing else, which catches the
stray print. Note that a `print()` can pass a casual manual test and still be a
bug: Python block-buffers text stdout into a pipe, so the output often appears
harmlessly at exit right up until some flush lands it between two frames.

### Replacing the model

Only the top half of `src/avogadro_so3lr/so3lr_tauri_m.py` is meant to
change:

1. `_prepare()` — build the calculator and anything else that survives across
   requests.
2. `_energy()` and `_energy_and_gradient()` — the calculation itself.
3. Update `support.elements` and the other `support.*` keys in
   `pyproject.toml` to match what the new model actually handles. Avogadro
   treats them as promises and will offer the model wherever they say it works.

### Adding a model

1. Add a `[[tool.avogadro.energy-models]]` block to `pyproject.toml` with a new
   `identifier`.
2. Add a matching `case` to `src/avogadro_so3lr/__init__.py`.
3. Write it in its own module, importing it inside that `case` so its
   dependencies load only when it is used.

## Installing

In Avogadro: **Extensions → Plugin Manager → Install from directory**, and
choose this folder.

## Suggestions

Please file suggestions and bugs at
<https://github.com/ghutchis/avogadro-so3lr/issues>.
