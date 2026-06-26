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


# A 3-deep nest used by the nested-construction tests below.
class _L1(eqx.Module):
    a: jax.Array

    def __init__(self, n: int):
        self.a = jnp.zeros(n)


class _L2(eqx.Module):
    l1: _L1
    b: jax.Array

    def __init__(self, n: int):
        self.l1 = _L1(n)
        self.b = jnp.ones(n)


class _L3(eqx.Module):
    l2: _L2
    c: jax.Array

    def __init__(self, n: int):
        self.l2 = _L2(n)
        self.c = jnp.full((n,), 2.0)


def _is_frozen(mod, attr) -> bool:
    """True if assigning `attr` on `mod` raises (i.e. it is a frozen Module)."""
    try:
        setattr(mod, attr, getattr(mod, attr))
        return False
    except Exception:
        return True


def _writable_count() -> int:
    """How many instances are currently in Equinox's writability set."""
    from eqx_cam.cam import _eqx_writable

    return len(_eqx_writable._dict)


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


def test_structure_stable_while_mutable():
    # the mutable copy must keep the original's pytree structure *inside* the
    # block (it is the same class, not a mutable subclass)
    mlp = eqx.nn.MLP(2, 2, 8, 2, key=jr.key(0))
    with copy_and_mutate(mlp) as m:
        assert type(m) is type(mlp)
        assert jax.tree.structure(m) == jax.tree.structure(mlp)
        m.layers[0].weight = m.layers[0].weight.at[0, 0].set(1.0)
        # still matches after an edit
        assert jax.tree.structure(m) == jax.tree.structure(mlp)


def test_optax_compatible_in_block():
    # the original bug: a mutable MLP must tree-map against pytrees built from
    # the *original* MLP (gradients / updates / optimizer state) while still
    # inside the with-block. A mutable-subclass copy would fail this.
    mlp = eqx.nn.MLP(2, 2, 8, 2, key=jr.key(0))
    grads = jax.tree.map(lambda x: jnp.ones_like(x) if eqx.is_array(x) else x, mlp)
    with copy_and_mutate(mlp) as m:
        m.layers[0].weight = m.layers[0].weight.at[0, 0].set(1.0)
        # optax-style: tree_map over (updates_from_original, mutable_params)
        merged = jax.tree.map(
            lambda _g, p: p,
            eqx.filter(grads, eqx.is_array),
            eqx.filter(m, eqx.is_array),
        )
        assert jax.tree.structure(merged) == jax.tree.structure(
            eqx.filter(mlp, eqx.is_array)
        )
        # eqx.apply_updates (sgd-style) also works in-block
        updates = jax.tree.map(lambda x: -0.1 * x if eqx.is_array(x) else None, grads)
        stepped = eqx.apply_updates(m, updates)
        assert type(stepped) is type(mlp)


def test_no_leaked_writability():
    # after the block nothing stays writable (and the writability set is unwound)
    from eqx_cam.cam import _eqx_writable

    mlp = eqx.nn.MLP(2, 2, 8, 2, key=jr.key(0))
    before = len(_eqx_writable._dict)
    with copy_and_mutate(mlp) as m:
        m.layers[0].weight = m.layers[0].weight.at[0, 0].set(1.0)
    assert len(_eqx_writable._dict) == before  # nothing left behind
    assert _is_frozen(m, "depth")
    assert _is_frozen(m.layers[0], "weight")


def test_nested_copy_and_mutate_same_tree():
    # an inner copy_and_mutate over the outer mutable copy must not freeze the
    # outer one early: each call only ever touches its own fresh copy's instances
    o = _Outer(3)
    with copy_and_mutate(o) as a:
        a.cursor = 1
        with copy_and_mutate(a) as b:
            b.cursor = 2
        assert int(b.cursor) == 2 and _is_frozen(b, "cursor")
        # `a` is still writable after the inner block exits
        a.inner.w = a.inner.w.at[0].set(9.0)
    assert int(a.cursor) == 1 and float(a.inner.w[0]) == 9.0
    assert _is_frozen(a, "cursor") and _is_frozen(a.inner, "w")


def test_inside_init():
    # using copy_and_mutate during another module's __init__ (when that module
    # is itself in Equinox's writability set) must not corrupt the set: the
    # enclosing module still freezes correctly once its __init__ returns
    from eqx_cam.cam import _eqx_writable

    before = len(_eqx_writable._dict)

    class Wraps(eqx.Module):
        inner: _Buf
        tag: int

        def __init__(self, src):
            with copy_and_mutate(src) as m:
                m.cursor = 7
            self.inner = m
            self.tag = 1

    w = Wraps(_Buf(3))
    assert int(w.inner.cursor) == 7
    assert type(w) is Wraps and type(w.inner) is _Buf
    # the enclosing module and its child are frozen after construction
    assert _is_frozen(w, "tag") and _is_frozen(w.inner, "cursor")
    assert len(_eqx_writable._dict) == before  # set fully unwound


# --------------------------------------------------------------------------- #
# Weirder nested constructions: __init__ × copy_and_mutate at several depths,  #
# with combinations that should and should not error.                         #
# --------------------------------------------------------------------------- #


def test_deep_simultaneous_edits():
    # edit leaves at all three depths in one block; everything re-freezes
    base = _writable_count()
    o = _L3(3)
    orig_a = o.l2.l1.a
    with copy_and_mutate(o) as m:
        m.c = m.c + 1.0
        m.l2.b = m.l2.b * 2.0
        m.l2.l1.a = m.l2.l1.a.at[0].set(9.0)
    assert float(m.c[0]) == 3.0
    assert float(m.l2.b[0]) == 2.0
    assert float(m.l2.l1.a[0]) == 9.0
    # original untouched and not aliased at any depth
    assert float(orig_a[0]) == 0.0 and m.l2.l1.a is not o.l2.l1.a
    assert _is_frozen(m, "c") and _is_frozen(m.l2, "b") and _is_frozen(m.l2.l1, "a")
    assert _writable_count() == base


def test_triple_nested_blocks():
    # three nested copy_and_mutate over the same (progressively copied) tree;
    # each inner exit must leave the enclosing copy writable, all re-freeze
    base = _writable_count()
    o = _L3(3)
    with copy_and_mutate(o) as a:
        a.c = a.c + 1.0
        with copy_and_mutate(a) as b:
            b.l2.b = b.l2.b + 1.0
            with copy_and_mutate(b) as c:
                c.l2.l1.a = c.l2.l1.a + 1.0
            assert _is_frozen(c, "c")
            # `b` still writable after the innermost block closed
            b.l2.l1.a = b.l2.l1.a + 5.0
        assert _is_frozen(b, "c")
        # `a` still writable after the middle block closed
        a.l2.b = a.l2.b + 10.0
    assert float(a.c[0]) == 3.0 and float(a.l2.b[0]) == 11.0
    assert _is_frozen(a, "c") and _is_frozen(a.l2.l1, "a")
    assert _writable_count() == base


def test_cam_in_init_assigns_module_field():
    # __init__ uses copy_and_mutate and stores the *module* it produced
    base = _writable_count()

    class Inner(eqx.Module):
        buf: _L1

        def __init__(self, src: _L1):
            with copy_and_mutate(src) as m:
                m.a = m.a.at[0].set(10.0)
            self.buf = m  # a frozen _L1, produced by cam during __init__

    class Outer(eqx.Module):
        inner: Inner
        w: jax.Array

        def __init__(self, src: _L1):
            self.inner = Inner(src)  # nested __init__ also runs cam
            with copy_and_mutate(src) as m2:
                m2.a = m2.a + 1.0
            self.w = m2.a

    out = Outer(_L1(3))
    assert float(out.inner.buf.a[0]) == 10.0
    assert float(out.w[0]) == 1.0
    assert type(out.inner.buf) is _L1
    assert _is_frozen(out, "w") and _is_frozen(out.inner.buf, "a")
    assert _writable_count() == base


def test_construct_module_inside_block():
    # constructing a fresh Equinox module inside the block interleaves Equinox's
    # own add/remove with ours; the fresh module must end frozen and our copy
    # must stay writable, with the set fully unwound afterwards
    base = _writable_count()
    o = _L2(3)
    with copy_and_mutate(o) as m:
        fresh = _L1(3)
        assert _is_frozen(fresh, "a")  # Equinox froze it on construction
        m.l1.a = m.l1.a + fresh.a + 1.0
        m.b = m.b * 2.0  # our copy is still writable afterwards
    assert float(m.l1.a[0]) == 1.0 and float(m.b[0]) == 2.0
    assert _is_frozen(m.l1, "a") and _is_frozen(m, "b")
    assert _writable_count() == base


def test_aliased_submodule_is_de_aliased():
    # the same submodule instance held in two fields becomes two independent
    # copies; editing one must not touch the other, and freezing visits the
    # (now-duplicated) node without a double-remove error
    base = _writable_count()

    class Dup(eqx.Module):
        x: _L1
        y: _L1

        def __init__(self, shared: _L1):
            self.x = shared
            self.y = shared

    shared = _L1(3)
    d = Dup(shared)
    assert d.x is d.y  # aliased in the original
    with copy_and_mutate(d) as nd:
        assert nd.x is not nd.y  # de-aliased in the copy
        nd.x.a = nd.x.a.at[0].set(5.0)
    assert float(nd.x.a[0]) == 5.0 and float(nd.y.a[0]) == 0.0
    assert _is_frozen(nd.x, "a") and _is_frozen(nd.y, "a")
    assert _writable_count() == base


def test_exception_in_nested_block_unwinds_all_levels():
    # an exception raised in the innermost of nested blocks must unwind the
    # writability set at every level and leave the original untouched
    base = _writable_count()
    o = _L3(3)
    captured = {}
    try:
        with copy_and_mutate(o) as a:
            captured["a"] = a
            with copy_and_mutate(a) as b:
                captured["b"] = b
                b.l2.l1.a = b.l2.l1.a + 1.0
                raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _writable_count() == base  # both levels unwound
    assert _is_frozen(captured["a"], "c") and _is_frozen(captured["b"], "c")
    assert float(o.l2.l1.a[0]) == 0.0  # original survives


def test_deep_validation_errors():
    # shape, dtype, and structure changes deep in the tree all error by default,
    # and the leaf-level messages point at the offending path
    o = _L3(3)

    raised = ""
    try:
        with copy_and_mutate(o) as m:
            m.l2.l1.a = jnp.zeros(9)  # deep shape change
    except ValueError as e:
        raised = str(e)
    assert ".l2.l1.a" in raised

    raised = ""
    try:
        with copy_and_mutate(o) as m:
            m.l2.l1.a = m.l2.l1.a.astype(jnp.int32)  # deep dtype change
    except ValueError as e:
        raised = str(e)
    assert ".l2.l1.a" in raised

    structural = False
    try:
        with copy_and_mutate(o) as m:
            m.l2.l1 = jnp.zeros(3)  # type: ignore[assignment]  # module -> array, deep
    except ValueError:
        structural = True
    assert structural


def test_deep_shape_change_allowed_with_flag():
    # the same deep shape change is fine when validation is disabled
    base = _writable_count()
    o = _L3(3)
    with copy_and_mutate(o, validate=False) as m:
        m.l2.l1.a = jnp.zeros(9)
    assert m.l2.l1.a.shape == (9,)
    assert o.l2.l1.a.shape == (3,)  # original untouched
    assert _is_frozen(m.l2.l1, "a")
    assert _writable_count() == base


def test_dict_structure_change_errors_but_value_edit_ok():
    # mutating a dict-valued field: changing keys is a structure change (errors),
    # editing values in place with matching keys/shapes is fine
    class Hold(eqx.Module):
        d: dict

        def __init__(self):
            self.d = {"a": jnp.zeros(2)}

    h = Hold()
    raised = False
    try:
        with copy_and_mutate(h) as nh:
            nh.d = {"a": jnp.zeros(2), "b": jnp.zeros(2)}  # added key
    except ValueError:
        raised = True
    assert raised

    with copy_and_mutate(h) as nh:
        nh.d["a"] = nh.d["a"].at[0].set(3.0)  # same keys/shape, in place
    assert float(nh.d["a"][0]) == 3.0
    assert float(h.d["a"][0]) == 0.0  # original untouched


# --------------------------------------------------------------------------- #
# Field kinds and semantic contracts: static fields, invariant checks,        #
# converters, scalar promotion, inheritance, and degenerate modules.          #
# --------------------------------------------------------------------------- #


def test_static_field_change_errors():
    # a static field lives in the treedef, so reassigning it is a structure
    # change and is rejected by default validation
    class WithStatic(eqx.Module):
        x: jax.Array
        name: str = eqx.field(static=True)

        def __init__(self, n):
            self.x = jnp.zeros(n)
            self.name = "a"

    ws = WithStatic(3)
    raised = False
    try:
        with copy_and_mutate(ws) as m:
            m.name = "b"
    except ValueError:
        raised = True
    assert raised
    # but a static field can be changed with validation off
    with copy_and_mutate(ws, validate=False) as m:
        m.name = "b"
    assert m.name == "b" and ws.name == "a"


def test_check_init_is_bypassed():
    # like eqx.tree_at, copy_and_mutate edits without reconstructing, so
    # __check_init__ invariants are NOT re-enforced on exit
    class Positive(eqx.Module):
        a: jax.Array

        def __init__(self, v):
            self.a = jnp.asarray(v)

        def __check_init__(self):
            if float(self.a) <= 0:
                raise ValueError("a must be positive")

    p = Positive(5.0)
    with copy_and_mutate(p, validate=False) as m:
        m.a = jnp.asarray(-1.0)  # would violate __check_init__ if it were re-run
    assert float(m.a) == -1.0  # no exception: the check was bypassed


def test_converter_is_bypassed():
    # field converters run during __init__, not on copy_and_mutate assignment
    class Conv(eqx.Module):
        a: jax.Array = eqx.field(converter=lambda v: 2 * jnp.asarray(v))

        def __init__(self, v):
            self.a = v

    cv = Conv(5.0)
    assert float(cv.a) == 10.0  # converter doubled at construction
    with copy_and_mutate(cv) as m:
        m.a = jnp.asarray(3.0)
    assert float(m.a) == 3.0  # NOT 6.0 — converter was not reapplied


def test_scalar_promotion_to_array_is_validated():
    # promoting a python-int field to a jax array changes the leaf's shape/dtype
    # (None -> ()), so default validation rejects it; validate=False allows it
    class Step(eqx.Module):
        x: jax.Array
        step: int

        def __init__(self):
            self.x = jnp.zeros(2)
            self.step = 0

    s = Step()
    raised = False
    try:
        with copy_and_mutate(s) as m:
            m.step = jnp.asarray(1)  # type: ignore[assignment]
    except ValueError:
        raised = True
    assert raised
    with copy_and_mutate(s, validate=False) as m:
        m.step = jnp.asarray(1)  # type: ignore[assignment]
    assert int(m.step) == 1


def test_inheritance():
    # a Module subclass with extra fields: inherited and new fields are both
    # mutable, and the result is the subclass
    class Base(eqx.Module):
        a: jax.Array

        def __init__(self, n):
            self.a = jnp.zeros(n)

    class Sub(Base):
        b: jax.Array

        def __init__(self, n):
            self.a = jnp.zeros(n)
            self.b = jnp.ones(n)

    with copy_and_mutate(Sub(3)) as m:
        m.a = m.a.at[0].set(1.0)
        m.b = m.b.at[0].set(2.0)
    assert type(m) is Sub
    assert float(m.a[0]) == 1.0 and float(m.b[0]) == 2.0
    assert _is_frozen(m, "a") and _is_frozen(m, "b")


def test_empty_and_all_static_modules():
    # degenerate modules (no fields / only static fields) round-trip cleanly
    base = _writable_count()

    class Empty(eqx.Module):
        pass

    class AllStatic(eqx.Module):
        name: str = eqx.field(static=True)

        def __init__(self):
            self.name = "hi"

    with copy_and_mutate(Empty()) as e:
        pass
    with copy_and_mutate(AllStatic()) as s:
        pass
    assert type(e) is Empty and type(s) is AllStatic and s.name == "hi"
    assert _writable_count() == base


def test_chained_sequential_blocks():
    # the frozen result of one block is a valid input to the next
    base = _writable_count()
    b = _Buf(3)
    with copy_and_mutate(b) as a:
        a.x = a.x.at[0].set(1.0)
    with copy_and_mutate(a) as c:
        c.x = c.x.at[1].set(2.0)
        c.cursor = c.cursor + 1
    assert float(c.x[0]) == 1.0 and float(c.x[1]) == 2.0 and int(c.cursor) == 1
    assert float(b.x[0]) == 0.0  # the very first original is still untouched
    assert _is_frozen(c, "cursor")
    assert _writable_count() == base
