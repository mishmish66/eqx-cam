"""Implementation of :func:`copy_and_mutate`.

The public entry point is re-exported from the package root; see ``__init__.py``
for the user-facing overview and attribution.
"""

import contextlib
import dataclasses
from collections.abc import Generator
from typing import Any, TypeVar

import equinox as eqx
import jax

# To support init in __init__, Equinox allows write to modules with the id
# in _currently_initialising. We abuse this and use it to temporarily allow writes
try:
    from equinox._module._module import _currently_initialising as _eqx_writable
except ImportError as e:
    raise ImportError(
        "eqx-cam relies on equinox._module._module._currently_initialising to "
        "make modules temporarily writable. This private symbol was not found, "
        "which usually means the installed Equinox version is incompatible "
        "(tested against equinox 0.13.x)."
    ) from e

__all__ = ["copy_and_mutate"]

M = TypeVar("M", bound=eqx.Module)


def _children(obj: Any):
    leaves, treedef = jax.tree.flatten(obj, is_leaf=lambda x: x is not obj)
    if treedef.num_leaves == 1 and leaves[0] is obj:
        return None, None
    return leaves, treedef


def _set_mutable(obj: Any) -> None:
    if isinstance(obj, eqx.Module):
        for f in dataclasses.fields(obj):
            _set_mutable(object.__getattribute__(obj, f.name))
        _eqx_writable.add(obj)
        return
    children, _ = _children(obj)
    if children is None:
        return
    for c in children:
        _set_mutable(c)


def _set_frozen(obj: Any) -> None:
    if isinstance(obj, eqx.Module):
        for f in dataclasses.fields(obj):
            _set_frozen(object.__getattribute__(obj, f.name))
        if obj in _eqx_writable:
            _eqx_writable.remove(obj)
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

    Within the ``with`` block, the yielded copy and any nested modules are mutable
    allowing attribute assignment. The original ``module`` is never mutated.
    On exit the copy and its children are frozen again.

    If ``validate`` is true (the default), the new module is checked against the
    original. This catches accidental structural edits. Pass ``validate=False``
    to permit shape- or dtype- changing assignments (e.g. growing a buffer).

    Example::

        with copy_and_mutate(model) as m:
            m.layer.weight = m.layer.weight.at[0].set(1.0)
        # `m` is a normal frozen Equinox module here, `model` is unchanged
    """

    new = jax.tree.map(lambda x: x, module)
    try:
        _set_mutable(new)
        yield new
    finally:
        _set_frozen(new)
    if validate:
        _validate(module, new)
