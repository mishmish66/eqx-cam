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

import contextlib
import dataclasses
from collections.abc import Generator
from importlib import metadata
from typing import Any, TypeVar

import equinox as eqx
import jax

try:
    __version__ = metadata.version("eqx-cam")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.0.0+unknown"

__all__ = ["copy_and_mutate", "__version__"]

M = TypeVar("M", bound=eqx.Module)

# Maps each frozen Equinox module class to a temporary mutable subclass, and
# back again, so the swap is cheap and reversible.
_mutable_cache: dict[type, type] = {}
_frozen_of: dict[type, type] = {}


def _mutable_cls(cls: type) -> type:
    """A subclass of `cls` whose instances permit attribute assignment."""
    if cls not in _mutable_cache:
        mut = type(
            f"_Mutable{cls.__name__}",
            (cls,),
            {"__setattr__": lambda self, k, v: object.__setattr__(self, k, v)},
        )
        _mutable_cache[cls] = mut
        _frozen_of[mut] = cls
    return _mutable_cache[cls]


def _children(obj: Any):
    leaves, treedef = jax.tree.flatten(obj, is_leaf=lambda x: x is not obj)
    if treedef.num_leaves == 1 and leaves[0] is obj:
        return None, None
    return leaves, treedef


def _set_mutable(obj: Any) -> None:
    if isinstance(obj, eqx.Module):
        for f in dataclasses.fields(obj):
            _set_mutable(object.__getattribute__(obj, f.name))
        object.__setattr__(obj, "__class__", _mutable_cls(type(obj)))
        return
    children, _ = _children(obj)
    if children is None:
        return
    for c in children:
        _set_mutable(c)


def _set_frozen(obj: Any) -> None:
    if type(obj) in _frozen_of:
        for f in dataclasses.fields(obj):
            _set_frozen(object.__getattribute__(obj, f.name))
        object.__setattr__(obj, "__class__", _frozen_of[type(obj)])
        return
    children, _ = _children(obj)
    if children is None:
        return
    for c in children:
        _set_frozen(c)


def _leaf_meta(x: Any) -> tuple[Any, Any]:
    """The ``(shape, dtype)`` of a leaf, or ``(None, None)`` if it has none."""
    return getattr(x, "shape", None), getattr(x, "dtype", None)


def _validate(original: Any, mutated: Any) -> None:
    """Raise if `mutated` differs from `original` in structure, shape, or dtype."""
    orig_items, orig_treedef = jax.tree.flatten_with_path(original)
    new_items, new_treedef = jax.tree.flatten_with_path(mutated)
    if orig_treedef != new_treedef:
        msg = (
            "copy_and_mutate(..., validate=True): the pytree structure changed "
            + "inside the block. Pass validate=False to allow this.\n"
            + f"  before: {orig_treedef}\n"
            + f"  after:  {new_treedef}"
        )
        raise ValueError(msg)

    for (path, o), (_, n) in zip(orig_items, new_items, strict=True):
        if _leaf_meta(o) != _leaf_meta(n):
            where = jax.tree_util.keystr(path) or "<root>"
            (os_, od), (ns, nd) = _leaf_meta(o), _leaf_meta(n)
            msg = (
                f"copy_and_mutate(..., validate=True): leaf '{where}' changed "
                + "shape/dtype inside the block. Pass validate=False to allow this.\n"
                + f"  before: shape={os_}, dtype={od}\n"
                + f"  after:  shape={ns}, dtype={nd}"
            )
            raise ValueError(msg)


@contextlib.contextmanager
def copy_and_mutate(module: M, *, validate: bool = True) -> Generator[M]:
    """Yield a mutable copy of ``module``, which freezes when the block exits.

    Within the ``with`` block, the yielded copy and every Equinox module nested
    inside it (including modules held in tuples, lists, and dicts) accept plain
    attribute assignment. The original ``module`` is never mutated, and
    untouched array leaves are shared with it rather than copied. On exit the
    copy and its children are restored to their real (frozen) classes, even if
    the block raises.

    If ``validate`` is true (the default), the copy is checked against the
    original after the block: its pytree structure, and every array leaf's shape
    and dtype, must be unchanged. This catches accidental structural edits during
    "model surgery". Pass ``validate=False`` to permit shape- or dtype-changing
    assignments (e.g. growing a buffer). Validation is skipped if the block
    raises, so it never masks the original error.

    Example::

        with copy_and_mutate(model) as m:
            m.layer.weight = m.layer.weight.at[0].set(1.0)
        # `m` is a normal frozen Equinox module here, `model` is unchanged
    """
    new = jax.tree.map(lambda x: x, module)
    _set_mutable(new)
    try:
        yield new
    finally:
        _set_frozen(new)
    if validate:
        _validate(module, new)
