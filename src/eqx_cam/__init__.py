"""eqx-cam: copy-and-mutate for Equinox modules.

Equinox modules are frozen dataclasses. Editing a deeply nested field normally
means threading replacement values back out through `eqx.tree_at` (or nested
`dataclasses.replace`) calls. `copy_and_mutate` can often be more convenient.
It just yields a temporarily mutable *copy* so you can just change whatever
you need and get back your `eqx.Module`.

The API and approach are directly inspired by ``jax_dataclasses.copy_and_mutate``
by Brent Yi (https://github.com/brentyi/jax_dataclasses, MIT licensed). This is
a reimplementation targeting :class:`equinox.Module` rather than
``jax_dataclasses`` pytree dataclasses. See the README for full attribution.
"""

from importlib import metadata

from eqx_cam.cam import copy_and_mutate

try:
    __version__ = metadata.version("eqx-cam")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.0.0+unknown"

__all__ = ["copy_and_mutate", "__version__"]
