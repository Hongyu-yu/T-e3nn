"""Check optimized contractions against a dense, independently expressed tensor product."""

import io

import pytest
import torch

import e3nn
from e3nn import o3
from e3nn.util.test import assert_equivariant


def _dense_reference(module, x, y, weight):
    """Keep the full channel outer product so selection cannot share the optimized implementation."""
    inputs = [x[..., :1], y[..., :1]]
    if not module.shared_weights and module.weight_numel:
        inputs.append(weight[..., :1])
    batch_shape = torch.broadcast_tensors(*inputs)[0].shape[:-1]
    x = x.expand(batch_shape + (x.shape[-1],))
    y = y.expand(batch_shape + (y.shape[-1],))
    outputs = [x.new_zeros(batch_shape + (mul_ir.dim,)) for mul_ir in module.irreps_out]
    offset = 0
    for instruction in module.instructions:
        i, j, k = instruction.i_in1, instruction.i_in2, instruction.i_out
        left, right, out = module.irreps_in1[i], module.irreps_in2[j], module.irreps_out[k]
        a = x[..., module.irreps_in1.slices()[i]].reshape(batch_shape + (left.mul, left.ir.dim))
        b = y[..., module.irreps_in2.slices()[j]].reshape(batch_shape + (right.mul, right.ir.dim))
        coupling = o3.wigner_3j(left.ir.l, right.ir.l, out.ir.l).to(dtype=x.dtype, device=x.device)
        full = torch.einsum("...ui,...vj,ijk->...uvk", a, b, coupling)
        mode = instruction.connection_mode
        if instruction.has_weight:
            count = 1
            for extent in instruction.path_shape:
                count *= extent
            w = weight[..., offset : offset + count].reshape(weight.shape[:-1] + instruction.path_shape)
            offset += count
        if mode in ("uvu<v", "u<vw"):
            indices = torch.triu_indices(left.mul, right.mul, 1, device=x.device)
            value = full[..., indices[0], indices[1], :]
            if mode == "u<vw":
                value = torch.einsum("...qw,...qk->...wk", w, value)
            elif instruction.has_weight:
                value = value * w.unsqueeze(-1)
        elif mode == "uvuv":
            value = full * w.unsqueeze(-1) if instruction.has_weight else full
        elif mode in ("uuu", "uuw"):
            index = torch.arange(left.mul, device=x.device)
            value = full[..., index, index, :]
            if mode == "uuw":
                value = torch.einsum("...uw,...uk->...wk", w, value)
            elif instruction.has_weight:
                value = value * w.unsqueeze(-1)
        elif mode == "uvw":
            value = torch.einsum("...uvw,...uvk->...wk", w, full)
        else:
            raise AssertionError("Unexpected reference connection mode: " + mode)
        outputs[k] = outputs[k] + instruction.path_weight * value.reshape(batch_shape + (out.dim,))
    return torch.cat(outputs, dim=-1)


def _compare_derivatives(actual, reference, arguments, tolerance):
    out = actual(*arguments)
    expected = reference(*arguments)
    assert torch.allclose(out, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(out)
    gradients = torch.autograd.grad((out.tanh() * cotangent).sum(), arguments, create_graph=True)
    expected_gradients = torch.autograd.grad((expected.tanh() * cotangent).sum(), arguments, create_graph=True)
    for value, target in zip(gradients, expected_gradients):
        assert torch.allclose(value, target, atol=tolerance, rtol=tolerance)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), arguments, allow_unused=True)
    expected_second = torch.autograd.grad(sum(g.square().sum() for g in expected_gradients), arguments, allow_unused=True)
    for value, target in zip(second, expected_second):
        assert (value is None) == (target is None)
        if value is not None:
            assert torch.allclose(value, target, atol=tolerance, rtol=tolerance)


def _triangle(mode, weighted, shared=True, optimize=True, labels=(1, 1), normalization="component", mul=4):
    p, t = labels
    output_mul = 2 if mode == "u<vw" else mul * (mul - 1) // 2
    return o3.TensorProduct(
        [(mul, (1, p, t))],
        [(mul, (1, -p, t))],
        [(output_mul, (ell, -1, 1)) for ell in (0, 1, 2)],
        [(0, 0, k, mode, weighted) for k in range(3)],
        internal_weights=False,
        shared_weights=shared,
        _optimize_einsums=optimize,
        irrep_normalization=normalization,
    )


@pytest.mark.parametrize("mode,weighted", [("uvu<v", False), ("uvu<v", True), ("u<vw", True)])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("labels", [(1, 1), (-1, -1)])
def test_triangle_dense_derivatives(mode, weighted, shared, optimize, labels, float_tolerance):
    module = _triangle(mode, weighted, shared, optimize, labels)
    x = (0.2 * torch.randn(2, module.irreps_in1.dim)).requires_grad_()
    y = (0.2 * torch.randn(2, module.irreps_in2.dim)).requires_grad_()
    w = (0.2 * torch.randn((() if shared else (2,)) + (module.weight_numel,))).requires_grad_()
    if weighted:
        _compare_derivatives(module, lambda a, b, c: _dense_reference(module, a, b, c), (x, y, w), float_tolerance)
    else:
        _compare_derivatives(
            lambda a, b: module(a, b), lambda a, b: _dense_reference(module, a, b, w), (x, y), float_tolerance
        )


@pytest.mark.parametrize("mode,weighted", [("uvu<v", False), ("uvu<v", True), ("u<vw", True)])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("layout", ["broadcast", "noncontiguous", "empty"])
def test_triangle_layouts(mode, weighted, shared, layout, float_tolerance):
    module = _triangle(mode, weighted, shared)
    n = module.irreps_in1.dim
    if layout == "broadcast":
        x, y = torch.randn(2, 1, n), torch.randn(1, 3, n)
        weight_batch = (1, 3)
    elif layout == "noncontiguous":
        x, y = torch.randn(n, 3, 2).permute(2, 1, 0), torch.randn(n, 3, 2).permute(2, 1, 0)
        assert not x.is_contiguous() and not y.is_contiguous()
        weight_batch = (2, 3)
    else:
        x, y = torch.randn(0, 2, n), torch.randn(1, 2, n)
        weight_batch = (1, 2)
    w = torch.randn((() if shared else weight_batch) + (module.weight_numel,))
    assert torch.allclose(module(x, y, w), _dense_reference(module, x, y, w), atol=float_tolerance, rtol=float_tolerance)
    if layout != "empty":
        x, y, w = (value.mul(0.2).requires_grad_() for value in (x, y, w))
        if weighted:
            _compare_derivatives(module, lambda a, b, c: _dense_reference(module, a, b, c), (x, y, w), float_tolerance)
        else:
            _compare_derivatives(
                lambda a, b: module(a, b), lambda a, b: _dense_reference(module, a, b, w), (x, y), float_tolerance
            )


@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("weighted", [False, True])
def test_tensor_square_alias_derivatives(normalization, weighted, float_tolerance):
    module = o3.TensorSquare(
        [(3, (1, -1, -1))],
        [(2, (0, 1, 1)), (2, (2, 1, 1))] if weighted else None,
        irrep_normalization=normalization,
        internal_weights=False,
    )
    x = (0.2 * torch.randn(2, module.irreps_in.dim)).requires_grad_()
    w = (0.2 * torch.randn(module.weight_numel)).requires_grad_()
    if weighted:
        _compare_derivatives(module, lambda a, b: _dense_reference(module, a, a, b), (x, w), float_tolerance)
    else:
        _compare_derivatives(module, lambda a: _dense_reference(module, a, a, w), (x,), float_tolerance)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_mixed_dense_and_triangle_outputs(optimize, shared, float_tolerance):
    module = o3.TensorProduct(
        "3x1o",
        "3x1o",
        "3x1e + 2x2e + 9x0e + 2x1e + 3x2e",
        [
            (0, 0, 0, "uvu<v", True),
            (0, 0, 1, "u<vw", True),
            (0, 0, 2, "uvuv", True),
            (0, 0, 3, "u<vw", True),
            (0, 0, 4, "uvu<v", True),
        ],
        internal_weights=False,
        shared_weights=shared,
        _optimize_einsums=optimize,
    )
    x = (0.2 * torch.randn(2, 9)).requires_grad_()
    y = (0.2 * torch.randn(2, 9)).requires_grad_()
    w = (0.2 * torch.randn((() if shared else (2,)) + (module.weight_numel,))).requires_grad_()
    _compare_derivatives(module, lambda a, b, c: _dense_reference(module, a, b, c), (x, y, w), float_tolerance)


@pytest.mark.parametrize("mode,weighted", [("uvu<v", False), ("uvu<v", True), ("u<vw", True)])
def test_triangle_gradcheck(mode, weighted):
    module = _triangle(mode, weighted, mul=2).double()
    x = (0.2 * torch.randn(1, module.irreps_in1.dim, dtype=torch.float64)).requires_grad_()
    y = (0.2 * torch.randn(1, module.irreps_in2.dim, dtype=torch.float64)).requires_grad_()
    w = (0.2 * torch.randn(module.weight_numel, dtype=torch.float64)).requires_grad_()
    args = (x, y, w) if weighted else (x, y)
    assert torch.autograd.gradcheck(module, args)
    assert torch.autograd.gradgradcheck(module, args)


@pytest.mark.parametrize("mode,weighted", [("uvu<v", False), ("uvu<v", True), ("u<vw", True)])
def test_triangle_script_roundtrip(mode, weighted, float_tolerance):
    module = _triangle(mode, weighted)
    scripted = torch.jit.script(module)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    restored = torch.jit.load(buffer)
    x, y = torch.randn(2, 12), torch.randn(2, 12)
    w = torch.randn(module.weight_numel)
    assert torch.allclose(restored(x, y, w), module(x, y, w), atol=float_tolerance, rtol=float_tolerance)
    assert_equivariant(
        lambda a, b: module(a, b, w),
        irreps_in=[module.irreps_in1, module.irreps_in2],
        irreps_out=module.irreps_out,
        do_time_reversal=True,
    )


@pytest.mark.parametrize("mode,weighted", [("uvu<v", False), ("uvu<v", True), ("u<vw", True)])
@pytest.mark.parametrize("mul", [0, 1])
def test_triangle_zero_paths(mode, weighted, mul, float_tolerance):
    module = _triangle(mode, weighted, mul=mul)
    x, y = torch.randn(2, module.irreps_in1.dim), torch.randn(2, module.irreps_in2.dim)
    w = torch.randn(module.weight_numel)
    assert torch.allclose(module(x, y, w), _dense_reference(module, x, y, w), atol=float_tolerance, rtol=float_tolerance)


@pytest.mark.skipif(not hasattr(torch, "compile") or not hasattr(torch, "export"), reason="Requires PyTorch compile/export")
def test_triangle_compile_export(float_tolerance):
    torch._dynamo.reset()
    defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(jit_script_fx=False)
        module = _triangle("u<vw", True, shared=False)
    finally:
        e3nn.set_optimization_defaults(**defaults)
    args = (torch.randn(2, 12), torch.randn(2, 12), torch.randn(2, module.weight_numel))
    expected = module(*args)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    exported = torch.export.export(module, args).module()
    for function in (compiled, exported):
        assert torch.allclose(function(*args), expected, atol=float_tolerance, rtol=float_tolerance)


def _uvuv(ells, weighted, shared=True, optimize=True, specialized=True, normalization="component", right=False):
    l1, l2, lo = ells
    return o3.TensorProduct(
        [(2, (l1, -1, -1))],
        [(3, (l2, -1, 1))],
        [(6, (lo, 1, -1))],
        [(0, 0, 0, "uvuv", weighted)],
        internal_weights=False,
        shared_weights=shared,
        _optimize_einsums=optimize,
        _specialized_code=specialized,
        irrep_normalization=normalization,
        compile_right=right,
    )


@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_uvuv_dense_derivatives(ells, weighted, shared, optimize, float_tolerance):
    module = _uvuv(ells, weighted, shared, optimize)
    x = (0.2 * torch.randn(2, module.irreps_in1.dim)).requires_grad_()
    y = (0.2 * torch.randn(2, module.irreps_in2.dim)).requires_grad_()
    w = (0.2 * torch.randn((() if shared else (2,)) + (module.weight_numel,))).requires_grad_()
    if weighted:
        _compare_derivatives(module, lambda a, b, c: _dense_reference(module, a, b, c), (x, y, w), float_tolerance)
    else:
        _compare_derivatives(
            lambda a, b: module(a, b), lambda a, b: _dense_reference(module, a, b, w), (x, y), float_tolerance
        )


@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 1, 1), (3, 0, 3), (1, 1, 0)])
@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("weighted", [False, True])
def test_uvuv_specialization_switch_and_right(ells, normalization, weighted, float_tolerance):
    module = _uvuv(ells, weighted, normalization=normalization, right=True)
    general = _uvuv(ells, weighted, normalization=normalization, specialized=False)
    x, y, w = torch.randn(2, module.irreps_in1.dim), torch.randn(2, module.irreps_in2.dim), torch.randn(module.weight_numel)
    result = module(x, y, w)
    assert torch.allclose(result, general(x, y, w), atol=float_tolerance, rtol=float_tolerance)
    assert torch.allclose(
        result, torch.einsum("...i,...ij->...j", x, module.right(y, w)), atol=float_tolerance, rtol=float_tolerance
    )
    assert_equivariant(
        lambda a, b: module(a, b, w),
        irreps_in=[module.irreps_in1, module.irreps_in2],
        irreps_out=module.irreps_out,
        do_time_reversal=True,
    )


@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)])
@pytest.mark.parametrize("optimize", [False, True])
def test_uvuv_strict_checkpoint(ells, optimize):
    module = _uvuv(ells, True, optimize=optimize)
    general = _uvuv(ells, True, optimize=optimize, specialized=False)
    assert set(module.state_dict()) == set(general.state_dict())
    module.load_state_dict(general.state_dict(), strict=True)
    general.load_state_dict(module.state_dict(), strict=True)


@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 1), (2, 2, 0)])
@pytest.mark.parametrize("optimize", [False, True])
def test_uvuv_checkpoint_dtype_conversion(ells, optimize):
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        module = _uvuv(ells, True, optimize=optimize).double()
        general = _uvuv(ells, True, optimize=optimize, specialized=False).double()
    finally:
        torch.set_default_dtype(default_dtype)
    module.load_state_dict(general.state_dict(), strict=True)
    x = torch.randn(2, module.irreps_in1.dim, dtype=torch.float64).requires_grad_()
    y = torch.randn(2, module.irreps_in2.dim, dtype=torch.float64).requires_grad_()
    w = torch.randn(module.weight_numel, dtype=torch.float64).requires_grad_()
    _compare_derivatives(module, general, (x, y, w), 1e-9)


@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)])
@pytest.mark.parametrize("layout", ["broadcast", "noncontiguous", "empty"])
def test_uvuv_layout_and_script(ells, layout, float_tolerance):
    module = _uvuv(ells, True, shared=False)
    n, m = module.irreps_in1.dim, module.irreps_in2.dim
    if layout == "broadcast":
        x, y, w = torch.randn(2, 1, n), torch.randn(1, 3, m), torch.randn(1, 3, module.weight_numel)
    elif layout == "noncontiguous":
        x = torch.randn(n, 3, 2).permute(2, 1, 0)
        y = torch.randn(m, 3, 2).permute(2, 1, 0)
        w = torch.randn(module.weight_numel, 3, 2).permute(2, 1, 0)
        assert not x.is_contiguous() and not w.is_contiguous()
    else:
        x, y, w = torch.randn(0, 2, n), torch.randn(1, 2, m), torch.randn(1, 2, module.weight_numel)
    expected = _dense_reference(module, x, y, w)
    serialized = io.BytesIO()
    torch.jit.save(torch.jit.script(module), serialized)
    serialized.seek(0)
    restored = torch.jit.load(serialized)
    for function in (module, restored):
        assert torch.allclose(function(x, y, w), expected, atol=float_tolerance, rtol=float_tolerance)
    if layout != "empty":
        x, y, w = (value.mul(0.2).requires_grad_() for value in (x, y, w))
        _compare_derivatives(module, lambda a, b, c: _dense_reference(module, a, b, c), (x, y, w), float_tolerance)


@pytest.mark.skipif(not hasattr(torch, "compile") or not hasattr(torch, "export"), reason="Requires PyTorch compile/export")
@pytest.mark.parametrize("ells", [(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)])
def test_uvuv_compile_export(ells, float_tolerance):
    torch._dynamo.reset()
    defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(jit_script_fx=False)
        module = _uvuv(ells, True, shared=False)
    finally:
        e3nn.set_optimization_defaults(**defaults)
    args = (
        torch.randn(2, module.irreps_in1.dim),
        torch.randn(2, module.irreps_in2.dim),
        torch.randn(2, module.weight_numel),
    )
    expected = module(*args)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    exported = torch.export.export(module, args).module()
    for function in (compiled, exported):
        assert torch.allclose(function(*args), expected, atol=float_tolerance, rtol=float_tolerance)
