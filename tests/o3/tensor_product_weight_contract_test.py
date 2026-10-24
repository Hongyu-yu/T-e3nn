import pytest
import torch

from e3nn import o3


@pytest.mark.parametrize("internal,shared", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("weighted", [(False, True), (True, False, True), (False, True, False, True, False)])
def test_mixed_weight_views_alias_packed_weights(internal, shared, weighted):
    tp = o3.TensorProduct(
        "2x0ee",
        "2x0ee",
        "2x0ee",
        [(0, 0, 0, "uuu", has_weight) for has_weight in weighted],
        internal_weights=internal,
        shared_weights=shared,
    )
    batch = () if shared else (2, 3)
    weights = tp.weight if internal else torch.randn(batch + (tp.weight_numel,))
    argument = None if internal else weights
    offset = 0
    for index, instruction, expected in tp.weight_views(argument, yield_instruction=True):
        actual = tp.weight_view_for_instruction(index, argument)
        assert actual.shape == batch + instruction.path_shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.data_ptr() == expected.data_ptr()
        assert torch.equal(actual, expected)
        before = weights.clone()
        with torch.no_grad():
            actual.fill_(index + 10)
        assert torch.equal(weights[..., :offset], before[..., :offset])
        assert torch.equal(weights[..., offset : offset + 2], torch.full_like(actual, index + 10))
        assert torch.equal(weights[..., offset + 2 :], before[..., offset + 2 :])
        offset += 2
    for index, has_weight in enumerate(weighted):
        if not has_weight:
            with pytest.raises(ValueError, match="has no weights"):
                tp.weight_view_for_instruction(index, argument)


@pytest.mark.parametrize("shared", [True, False])
def test_zero_length_weight_path_does_not_shift_later_view(shared):
    tp = o3.TensorProduct(
        "0x0ee + 2x0ee",
        "0x0ee + 2x0ee",
        "0x0ee + 2x0ee",
        [(1, 1, 1, "uuu", False), (0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=shared,
    )
    batch = () if shared else (2, 3)
    weights = torch.randn(batch + (tp.weight_numel,))
    assert tp.weight_view_for_instruction(1, weights).shape == batch + (0,)
    view = tp.weight_view_for_instruction(2, weights)
    assert view.data_ptr() == weights.data_ptr()
    assert torch.equal(view, weights)


@pytest.mark.parametrize("batch", [(3,), (2, 3), (2, 1, 3), (2, 0, 3)])
def test_unshared_weight_list_preserves_batch_axes(batch):
    tp = o3.TensorProduct(
        "2x0eo",
        "2x0oo",
        "2x0oe",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
        compile_right=True,
    )
    x, y, weight = (torch.randn(batch + (2,)) for _ in range(3))
    for method, args in [(tp, (x, y)), (tp.right, (y,))]:
        expected = method(*args, weight)
        actual = method(*args, [weight])
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


def test_unshared_list_without_batch_axes_retains_singleton_broadcast():
    tp = o3.TensorProduct(
        "2x0ee",
        "2x0ee",
        "2x0ee",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
    )
    x, y = (torch.randn(2, 3, 2) for _ in range(2))
    weight = torch.randn(2, 2, 2)
    assert torch.equal(tp(x, y, [weight]), tp(x, y, weight.reshape(1, -1)))


@pytest.mark.parametrize("shared", [True, False])
def test_zero_length_weight_list_entry_broadcasts_with_nonempty_entry(shared):
    tp = o3.TensorProduct(
        "0x0ee + 2x0ee",
        "0x0ee + 2x0ee",
        "0x0ee + 2x0ee",
        [(0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=shared,
    )
    x, y = (torch.randn(2, 3, 2) for _ in range(2))
    empty = torch.empty((0,) if shared else (2, 1, 0))
    nonempty = torch.randn((2,) if shared else (1, 3, 2))
    packed = nonempty if shared else nonempty.expand(2, 3, 2)
    assert torch.equal(tp(x, y, [empty, nonempty]), tp(x, y, packed))


@pytest.mark.parametrize("shared", [True, False])
def test_weight_list_broadcast_and_gradients(shared, float_tolerance):
    tp = o3.TensorProduct(
        "2x0eo",
        "2x0oo",
        "2x0oe",
        [(0, 0, 0, "uuu", True), (0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=shared,
        compile_right=True,
    )
    batch = () if shared else (2, 3, 4)
    batches = [(), ()] if shared else [(2, 1, 4), (1, 3, 1)]
    weights = []
    packed = []
    for ins, leading in zip(tp.instructions, batches):
        shape = leading + ins.path_shape
        # Slicing retains a non-contiguous leaf, exercising reshape's copy path.
        w = torch.randn(shape[:-1] + (2 * shape[-1],))[..., ::2].requires_grad_()
        weights.append(w)
        expanded = w.expand(batch + ins.path_shape)
        packed.append(expanded.reshape(batch + (-1,)))
    packed = torch.cat(packed, dim=-1)
    x = torch.randn(2, 1, 4, 2, requires_grad=True)
    y = torch.randn(1, 3, 1, 2, requires_grad=True)
    for method, args in [(tp, (x, y)), (tp.right, (y,))]:
        expected = method(*args, packed)
        actual = method(*args, weights)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=float_tolerance, rtol=0)
        inputs = args + tuple(weights)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert actual_grad.shape == expected_grad.shape
            assert torch.allclose(actual_grad, expected_grad, atol=float_tolerance, rtol=0)


@pytest.mark.parametrize("shared", [True, False])
def test_weight_list_supports_second_derivatives(shared):
    tp = o3.TensorProduct(
        "1x0ee",
        "1x0ee",
        "1x0ee",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=shared,
    ).double()
    x = torch.randn(2, 1, 1, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 2, 1, dtype=torch.float64, requires_grad=True)
    shape = (1,) if shared else (1, 2, 1)
    weight = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    def forward(a, b, w):
        return tp(a, b, [w])

    assert torch.autograd.gradcheck(forward, (x, y, weight))
    assert torch.autograd.gradgradcheck(forward, (x, y, weight))


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("instructions", [[], [(0, 0, 0, "uuu", False)]])
def test_empty_weight_list_matches_omitted_weights(shared, instructions):
    tp = o3.TensorProduct(
        "2x0ee",
        "2x0ee",
        "2x0ee",
        instructions,
        internal_weights=False,
        shared_weights=shared,
        compile_right=True,
    )
    x = torch.randn(2, 1, 2)
    y = torch.randn(1, 3, 2)
    for method, args in [(tp, (x, y)), (tp.right, (y,))]:
        expected = method(*args)
        actual = method(*args, [])
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("count", [0, 2])
def test_weight_list_requires_one_entry_per_weighted_instruction(shared, count):
    tp = o3.TensorProduct(
        "2x0ee",
        "2x0ee",
        "2x0ee",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=shared,
    )
    shape = (2,) if shared else (3, 2)
    with pytest.raises(ValueError, match="weight.*list|list.*weight"):
        tp(torch.randn(3, 2), torch.randn(3, 2), [torch.randn(shape) for _ in range(count)])


def test_tensor_weight_torchscript_path_is_unchanged(float_tolerance):
    tp = o3.TensorProduct(
        "2x0eo",
        "2x0oo",
        "2x0oe",
        [(0, 0, 0, "uuu", False), (0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
        compile_right=True,
    )
    scripted = torch.jit.script(tp)
    x, y, weights = (torch.randn(2, 3, 2) for _ in range(3))
    assert set(scripted.state_dict()) == set(tp.state_dict())
    assert torch.allclose(tp(x, y, weights), scripted(x, y, weights), atol=float_tolerance, rtol=0)
    assert torch.allclose(tp.right(y, weights), scripted.right(y, weights), atol=float_tolerance, rtol=0)


@pytest.mark.parametrize("mode,muls", [("uuu", (2, 2, 2)), ("uvu", (2, 3, 2)), ("uvw", (2, 3, 4))])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("representation", ["structured", "flat"])
@pytest.mark.parametrize("batch", [(4,), (2, 1, 3)])
def test_flattened_path_list_matches_tensor(mode, muls, shared, representation, batch, float_tolerance):
    tp = o3.TensorProduct(
        "{}x0ee".format(muls[0]),
        "{}x0ee".format(muls[1]),
        "{}x0ee".format(muls[2]),
        [(0, 0, 0, mode, True)],
        internal_weights=False,
        shared_weights=shared,
        compile_right=True,
    )
    leading = () if shared else batch
    shape = tp.instructions[0].path_shape if representation == "structured" else (tp.weight_numel,)
    # Both structured and flattened lists accept non-contiguous path storage.
    weight = torch.randn(leading + shape[:-1] + (2 * shape[-1],))[..., ::2].requires_grad_()
    packed = weight.reshape(leading + (tp.weight_numel,))
    x = torch.randn(batch + (muls[0],), requires_grad=True)
    y = torch.randn(batch + (muls[1],), requires_grad=True)
    for method, args in [(tp, (x, y)), (tp.right, (y,))]:
        expected = method(*args, packed)
        actual = method(*args, [weight])
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=float_tolerance, rtol=0)
        actual_grad = torch.autograd.grad(actual.square().sum(), args + (weight,), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), args + (weight,), retain_graph=True)
        for a, b in zip(actual_grad, expected_grad):
            assert a.shape == b.shape
            assert torch.allclose(a, b, atol=float_tolerance, rtol=0)


def test_mixed_flat_and_structured_weight_entries_broadcast(float_tolerance):
    tp = o3.TensorProduct(
        "2x0ee",
        "3x0ee",
        "2x0ee",
        [(0, 0, 0, "uvu", True), (0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
        compile_right=True,
    )
    first = torch.randn(2, 1, 6, requires_grad=True)
    second = torch.randn(1, 3, 2, 3, 2, requires_grad=True)
    packed = torch.cat([first.expand(2, 3, 6), second.expand(2, 3, 2, 3, 2).reshape(2, 3, 12)], dim=-1)
    x = torch.randn(2, 1, 2, requires_grad=True)
    y = torch.randn(1, 3, 3, requires_grad=True)
    for method, args in [(tp, (x, y)), (tp.right, (y,))]:
        actual = method(*args, [first, second])
        expected = method(*args, packed)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=float_tolerance, rtol=0)
        inputs = args + (first, second)
        a = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)
        assert all(torch.allclose(u, v, atol=float_tolerance, rtol=0) for u, v in zip(a, b))


@pytest.mark.parametrize("mode,muls,shape", [("uvu", (1, 3, 1), (4, 1, 3)), ("uvw", (1, 1, 3), (4, 1, 1, 3))])
def test_singleton_path_axes_prefer_structured_interpretation(mode, muls, shape):
    tp = o3.TensorProduct(
        "{}x0ee".format(muls[0]),
        "{}x0ee".format(muls[1]),
        "{}x0ee".format(muls[2]),
        [(0, 0, 0, mode, True)],
        internal_weights=False,
        shared_weights=False,
    )
    weight = torch.randn(shape)
    x = torch.randn(4, muls[0])
    y = torch.randn(4, muls[1])
    actual = tp(x, y, [weight])
    expected = tp(x, y, weight.reshape(4, tp.weight_numel))
    assert actual.shape == expected.shape == (4, muls[2])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(4, 6, 4), (4, 4, 6), (96,)])
def test_legacy_partial_weight_layout_keeps_original_interpretation(shape):
    tp = o3.TensorProduct(
        "2x0ee",
        "3x0ee",
        "4x0ee",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
    )
    weight = torch.randn(shape)
    x, y = torch.randn(4, 2), torch.randn(4, 3)
    assert torch.equal(tp(x, y, [weight]), tp(x, y, weight.reshape(4, 24)))


def test_scalar_list_weight_retains_singleton_batch():
    tp = o3.TensorProduct(
        "1x0ee",
        "1x0ee",
        "1x0ee",
        [(0, 0, 0, "uvw", True)],
        internal_weights=False,
        shared_weights=False,
    )
    x, y, weight = torch.randn(4, 1), torch.randn(4, 1), torch.randn(())
    assert torch.equal(tp(x, y, [weight]), tp(x, y, weight.reshape(1, 1)))
