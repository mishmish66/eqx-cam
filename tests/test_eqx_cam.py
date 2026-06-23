"""Behavioural tests for :func:`eqx_cam.copy_and_mutate`."""

import dataclasses

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr

from eqx_cam import copy_and_mutate


class _Buf(eqx.Module):
    x: jax.Array
    y: jax.Array
    cursor: int

    def __init__(self, n: int):
        self.x = jnp.zeros(n)
        self.y = jnp.ones(n)
        self.cursor = 0


class _Inner(eqx.Module):
    w: jax.Array

    def __init__(self, n: int):
        self.w = jnp.ones(n)


class _Outer(eqx.Module):
    inner: _Inner
    cursor: int

    def __init__(self, n: int):
        self.inner = _Inner(n)
        self.cursor = 0


class _Acc(eqx.Module):
    """A carry-friendly module: all-array leaves, no static fields."""

    buf: jax.Array
    total: jax.Array

    def __init__(self, n: int):
        self.buf = jnp.zeros(n)
        self.total = jnp.zeros(())


class _MLP(eqx.Module):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(self, key, in_dim: int = 784, hidden: int = 32, out: int = 10):
        k1, k2 = jr.split(key)
        self.w1 = jr.normal(k1, (in_dim, hidden)) / jnp.sqrt(in_dim)
        self.b1 = jnp.zeros(hidden)
        self.w2 = jr.normal(k2, (hidden, out)) / jnp.sqrt(hidden)
        self.b2 = jnp.zeros(out)

    def __call__(self, x):
        h = jax.nn.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


def _is_frozen(mod, attr) -> bool:
    """True if assigning `attr` on `mod` raises (i.e. it is a frozen Module)."""
    try:
        setattr(mod, attr, getattr(mod, attr))
        return False
    except Exception:
        return True


def test_flat_mutation():
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        nb.x = nb.x.at[0].set(5.0)
        nb.cursor = nb.cursor + 1
    assert type(nb) is _Buf
    assert float(nb.x[0]) == 5.0
    assert int(nb.cursor) == 1


def test_original_untouched():
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        nb.x = nb.x.at[0].set(5.0)
        nb.cursor = 9
    assert nb is not b
    assert float(b.x[0]) == 0.0
    assert int(b.cursor) == 0


def test_untouched_leaf_is_shared():
    # a field we never write should not be needlessly copied
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        nb.cursor = 1
    assert nb.y is b.y


def test_custom_init_module():
    # dataclasses.replace cannot rebuild a custom-__init__ module; cam must.
    b = _Buf(2)
    raised = False
    try:
        dataclasses.replace(b, cursor=9)  # type: ignore[type-var]
    except Exception:
        raised = True
    assert raised, "expected dataclasses.replace to fail on a custom __init__"
    with copy_and_mutate(b) as nb:
        nb.cursor = 9
    assert int(nb.cursor) == 9


def test_shape_changing_assignment():
    b = _Buf(4)
    with copy_and_mutate(b, validate=False) as nb:
        nb.x = jnp.concatenate([nb.x, jnp.zeros_like(nb.x)])
    assert nb.x.shape == (8,)
    assert b.x.shape == (4,)


def test_validate_rejects_shape_change():
    b = _Buf(4)
    raised = False
    try:
        with copy_and_mutate(b) as nb:  # validate=True by default
            nb.x = jnp.concatenate([nb.x, jnp.zeros_like(nb.x)])
    except ValueError:
        raised = True
    assert raised, "expected default validation to reject a shape change"


def test_validate_rejects_dtype_change():
    b = _Buf(3)
    raised = False
    try:
        with copy_and_mutate(b) as nb:
            nb.x = nb.x.astype(jnp.int32)
    except ValueError:
        raised = True
    assert raised, "expected default validation to reject a dtype change"


def test_validate_rejects_structure_change():
    o = _Outer(3)
    raised = False
    try:
        with copy_and_mutate(o) as no:
            no.inner = (no.inner.w, no.inner.w)  # type: ignore[assignment]  # swap a Module for a tuple
    except ValueError:
        raised = True
    assert raised, "expected default validation to reject a structure change"


def test_validate_allows_inplace_edit():
    # shape/dtype-preserving edits pass validation and leave the module frozen
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        nb.x = nb.x.at[0].set(5.0)
        nb.cursor = nb.cursor + 1
    assert float(nb.x[0]) == 5.0 and int(nb.cursor) == 1
    assert _is_frozen(nb, "cursor")


def test_validate_does_not_mask_block_exception():
    # if the block raises, the original error propagates (not a ValidationError),
    # even though a shape change also occurred
    b = _Buf(3)
    try:
        with copy_and_mutate(b) as nb:
            nb.x = jnp.zeros(99)  # would fail validation
            raise RuntimeError("boom")
    except RuntimeError as e:
        assert str(e) == "boom"
    except ValueError as e:
        raise AssertionError("validation masked the original exception") from e


def test_nested_module():
    o = _Outer(3)
    orig_w = o.inner.w
    with copy_and_mutate(o) as no:
        no.inner.w = no.inner.w.at[0].set(7.0)
        no.cursor = 2
    assert type(no) is _Outer and type(no.inner) is _Inner
    assert float(no.inner.w[0]) == 7.0 and int(no.cursor) == 2
    # original sub-module untouched and not aliased
    assert float(orig_w[0]) == 1.0
    assert no.inner is not o.inner


def test_modules_inside_containers():
    class _Holder(eqx.Module):
        seq: eqx.nn.Sequential
        d: dict

        def __init__(self, seq, lin):
            self.seq = seq
            self.d = {"lin": lin}

    seq = eqx.nn.Sequential(
        [eqx.nn.Linear(3, 3, key=jr.key(0)), eqx.nn.Lambda(jax.nn.relu)]
    )
    h = _Holder(seq, eqx.nn.Linear(2, 2, key=jr.key(1)))
    w0 = h.seq.layers[0].weight
    b0 = h.d["lin"].bias
    with copy_and_mutate(h) as nh:
        nh.seq.layers[0].weight = nh.seq.layers[0].weight.at[0, 0].set(9.0)
        nh.d["lin"].bias = nh.d["lin"].bias.at[0].set(3.0)
    # tuple-nested and dict-nested Modules were mutated, real classes restored
    assert type(nh.seq) is eqx.nn.Sequential
    assert type(nh.seq.layers[0]) is eqx.nn.Linear
    assert float(nh.seq.layers[0].weight[0, 0]) == 9.0
    assert float(nh.d["lin"].bias[0]) == 3.0
    # originals untouched, deep layers not aliased
    assert float(w0[0, 0]) != 9.0
    assert float(b0[0]) != 3.0
    assert nh.seq.layers[0] is not h.seq.layers[0]


def test_refrozen_after_block():
    b = _Buf(2)
    with copy_and_mutate(b) as nb:
        nb.cursor = 1
    assert _is_frozen(nb, "cursor")
    o = _Outer(2)
    with copy_and_mutate(o) as no:
        no.cursor = 1
    # nested modules are frozen again too
    assert _is_frozen(no.inner, "w")


def test_pytree_roundtrip():
    o = _Outer(3)
    with copy_and_mutate(o) as no:
        no.inner.w = no.inner.w.at[1].set(4.0)
    leaves, treedef = jax.tree.flatten(no)
    rt = jax.tree.unflatten(treedef, leaves)
    assert type(rt) is _Outer and type(rt.inner) is _Inner
    assert float(rt.inner.w[1]) == 4.0


def test_no_mutation():
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        pass
    assert type(nb) is _Buf and nb is not b
    assert bool(jnp.all(nb.x == b.x)) and int(nb.cursor) == int(b.cursor)


def test_under_jit():
    @eqx.filter_jit
    def store(b, i, v):
        with copy_and_mutate(b) as nb:
            nb.x = nb.x.at[i].set(v)
            nb.cursor = nb.cursor + 1
        return nb

    nb = store(_Buf(4), jnp.array(2), jnp.array(8.0))
    assert type(nb) is _Buf
    assert float(nb.x[2]) == 8.0 and int(nb.cursor) == 1


def test_under_vmap():
    def store(b, i, v):
        with copy_and_mutate(b) as nb:
            nb.x = nb.x.at[i].set(v)
            nb.cursor = nb.cursor + 1
        return nb

    bs = jax.tree.map(lambda *x: jnp.stack(x), *[_Buf(4) for _ in range(3)])
    bs = eqx.filter_vmap(store)(bs, jnp.array([0, 1, 2]), jnp.array([1.0, 2.0, 3.0]))
    assert type(bs) is _Buf
    assert bs.x.shape == (3, 4)
    assert [float(bs.x[k, k]) for k in range(3)] == [1.0, 2.0, 3.0]
    assert bool(jnp.all(jnp.asarray(bs.cursor) == 1))


def test_exception_safety():
    b = _Buf(3)
    try:
        with copy_and_mutate(b) as nb:
            nb.cursor = 5
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # original survives an aborted mutation, and the class cache isn't corrupted
    assert int(b.cursor) == 0
    with copy_and_mutate(b) as nb2:
        nb2.cursor = 1
    assert int(nb2.cursor) == 1 and type(nb2) is _Buf
    assert _is_frozen(nb2, "cursor")


def test_under_scan():
    # carry a module through lax.scan, mutating its buffer each step
    def step(carry, xs):
        i, v = xs
        with copy_and_mutate(carry) as nc:
            nc.buf = nc.buf.at[i].set(v)
            nc.total = nc.total + v
        return nc, nc.total

    acc = _Acc(4)
    idxs = jnp.arange(4)
    vals = jnp.array([10.0, 20.0, 30.0, 40.0])
    final, totals = jax.lax.scan(step, acc, (idxs, vals))

    assert type(final) is _Acc
    assert [float(x) for x in final.buf] == [10.0, 20.0, 30.0, 40.0]
    assert float(final.total) == 100.0
    # the per-step outputs were collected as the running total
    assert [float(t) for t in totals] == [10.0, 30.0, 60.0, 100.0]
    assert _is_frozen(final, "total")


def test_under_cond():
    def inc(b):
        with copy_and_mutate(b) as nb:
            nb.x = nb.x + 1.0
        return nb

    def dec(b):
        with copy_and_mutate(b) as nb:
            nb.x = nb.x - 1.0
        return nb

    b = _Buf(3)  # x starts at zeros
    up = jax.lax.cond(True, inc, dec, b)
    down = jax.lax.cond(False, inc, dec, b)

    assert type(up) is _Buf and type(down) is _Buf
    assert float(up.x[0]) == 1.0
    assert float(down.x[0]) == -1.0
    # both branches mutate but leave a frozen module behind
    assert _is_frozen(up, "x") and _is_frozen(down, "x")


def test_under_switch():
    def make(delta):
        def branch(b):
            with copy_and_mutate(b) as nb:
                nb.x = nb.x + delta
            return nb

        return branch

    branches = [make(1.0), make(2.0), make(3.0)]
    b = _Buf(3)
    for idx in range(3):
        out = jax.lax.switch(idx, branches, b)
        assert type(out) is _Buf
        assert float(out.x[0]) == float(idx + 1)


def test_select_inside_block():
    # branchless selection (jnp.where) on a mutable copy inside the block
    b = _Buf(3)
    with copy_and_mutate(b) as nb:
        nb.x = jnp.where(jnp.array(True), nb.x + 5.0, nb.x - 5.0)
    assert float(nb.x[0]) == 5.0
    with copy_and_mutate(b) as nb2:
        nb2.x = jnp.where(jnp.array(False), nb2.x + 5.0, nb2.x - 5.0)
    assert float(nb2.x[0]) == -5.0


def test_cond_under_vmap():
    # a per-example branch inside vmap: jax lowers cond to a select, and each
    # lane must still come back as a properly frozen module
    def f(b, pred):
        def inc(b):
            with copy_and_mutate(b) as nb:
                nb.x = nb.x + 1.0
            return nb

        def dec(b):
            with copy_and_mutate(b) as nb:
                nb.x = nb.x - 1.0
            return nb

        return jax.lax.cond(pred, inc, dec, b)

    bs = jax.tree.map(lambda *x: jnp.stack(x), *[_Buf(3) for _ in range(3)])
    preds = jnp.array([True, False, True])
    out = eqx.filter_vmap(f)(bs, preds)

    assert type(out) is _Buf
    assert out.x.shape == (3, 3)
    assert [float(out.x[k, 0]) for k in range(3)] == [1.0, -1.0, 1.0]
    assert _is_frozen(out, "x")


def test_mnist_training_steps():
    # a few steps of SGD on synthetic, MNIST-shaped data, where the parameter
    # update is expressed as a copy_and_mutate edit of the model
    n, in_dim, n_cls = 128, 784, 10
    key = jr.key(0)
    dkey, wkey, mkey = jr.split(key, 3)
    X = jr.normal(dkey, (n, in_dim))
    W_true = jr.normal(wkey, (in_dim, n_cls))
    Y = jnp.argmax(X @ W_true, axis=1)  # a learnable labelling

    model = _MLP(mkey, in_dim=in_dim, hidden=32, out=n_cls)

    def loss_fn(m, xb, yb):
        logits = jax.vmap(m)(xb)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(logp[jnp.arange(yb.shape[0]), yb])

    @eqx.filter_jit
    def train_step(m, xb, yb, lr):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, xb, yb)
        # shape/dtype-preserving update -> default validation passes
        with copy_and_mutate(m) as nm:
            nm.w1 = nm.w1 - lr * grads.w1
            nm.b1 = nm.b1 - lr * grads.b1
            nm.w2 = nm.w2 - lr * grads.w2
            nm.b2 = nm.b2 - lr * grads.b2
        return nm, loss

    losses = []
    for _ in range(30):
        model, loss = train_step(model, X, Y, 0.3)
        losses.append(float(loss))

    assert type(model) is _MLP
    assert all(jnp.isfinite(jnp.array(losses)))  # training stayed numerically sane
    assert losses[-1] < losses[0]  # and the loss actually went down
    assert _is_frozen(model, "w1")
