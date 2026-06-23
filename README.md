# eqx-cam

**Copy-and-mutate for [Equinox](https://github.com/patrick-kidger/equinox) modules.**

*An Equinox port of `copy_and_mutate` from [jax_dataclasses](https://github.com/brentyi/jax_dataclasses) by Brent Yi. See [Attribution](#attribution).*

Equinox modules are frozen dataclasses. Editing a deeply nested field normally
means threading replacement values back out through `eqx.tree_at` (or nested
`dataclasses.replace`) calls. `copy_and_mutate` can often be more convenient.
It just yields a temporarily mutable *copy* so you can just change whatever
you need and get back your `eqx.Module`.

```python
import jax
import equinox as eqx
from eqx_cam import copy_and_mutate

model = eqx.nn.MLP(2, 2, 8, 2, key=jax.random.key(0))

with copy_and_mutate(model) as m:
    m.layers[0].weight = m.layers[0].weight.at[0, 0].set(1.0)

# `model` is unchanged; `m` is a normal frozen Equinox module.
```

## Attribution

This package is an independent reimplementation of the `copy_and_mutate`
context manager from **[jax_dataclasses](https://github.com/brentyi/jax_dataclasses)**
by **Brent Yi**, adapted to target `equinox.Module` rather than
`jax_dataclasses` pytree dataclasses. The idea of temporarily unfreezing a pytree
came directly from that project.

## Development

```bash
uv sync           # install with dev dependencies
uv run pytest     # run the test suite
uv run ruff check # lint
```
