"""Exercise uvu assembly independently of its generated split/stack graph."""

from contextlib import contextmanager
import importlib
import io
from unittest.mock import patch

import pytest
import torch

import e3nn
from e3nn import o3
from e3nn.util.test import assert_equivariant

codegen = importlib.import_module("e3nn.o3._tensor_product._codegen")


@contextmanager
def _raw_graphs():
    defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(jit_script_fx=False)
        yield
    finally:
        e3nn.set_optimization_defaults(**defaults)


def _module(count=18, mul=2, shared=False, optimize=True, specialized=True, duplicate=False, zero=False, right=False):
    outputs = [(mul, (ell, 1, -1)) for ell in (0, 1, 2) for _ in range(count // 3)]
    instructions = [(0, 0, index, "uvu", True, 0.0 if zero else 1.0) for index in reversed(range(len(outputs)))]
    if duplicate:
        instructions = instructions[1:] + [instructions[0], instructions[3], instructions[1]]
    return o3.TensorProduct(
        [(mul, (1, -1, -1))],
        [(1, (1, -1, 1))],
        outputs,
        instructions,
        shared_weights=shared,
        internal_weights=False,
        _optimize_einsums=optimize,
        _specialized_code=specialized,
        compile_right=right,
    )


def _dense(module, x, y, weights):
    shape = torch.broadcast_tensors(x[..., :1], y[..., :1], weights[..., :1])[0].shape[:-1]
    x = x.expand(shape + (x.shape[-1],))
    y = y.expand(shape + (y.shape[-1],))
    weights = weights.expand(shape + (weights.shape[-1],))
    terms = [[] for _ in module.irreps_out]
    offset = 0
    for instruction in module.instructions:
        left, right, output = (
            module.irreps_in1[instruction.i_in1],
            module.irreps_in2[instruction.i_in2],
            module.irreps_out[instruction.i_out],
        )
        a = x[..., module.irreps_in1.slices()[instruction.i_in1]].reshape(shape + (left.mul, left.ir.dim))
        b = y[..., module.irreps_in2.slices()[instruction.i_in2]].reshape(shape + (right.mul, right.ir.dim))
        wigner = o3.wigner_3j(left.ir.l, right.ir.l, output.ir.l).to(x)
        outer = torch.einsum("...ui,...vj->...uvij", a, b)
        angular = torch.einsum("...uvij,ijk->...uvk", outer, wigner)
        size = left.mul * right.mul
        weight = weights[..., offset : offset + size].reshape(shape + (left.mul, right.mul))
        offset += size
        value = (angular * weight.unsqueeze(-1)).sum(-2)
        terms[instruction.i_out].append(instruction.path_weight * value.reshape(shape + (output.dim,)))
    return torch.cat([sum(values[1:], values[0]) for values in terms], dim=-1)


def _derivatives(function, values, cotangent):
    unused = values[0].new_tensor(0.3, requires_grad=True)
    zero = values[0].new_tensor(0.2, requires_grad=True)
    variables = tuple(values) + (unused, zero)
    output = function(*values) + zero * 0
    first = torch.autograd.grad((output.tanh() * cotangent).sum(), variables, create_graph=True, allow_unused=True)
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first if value is not None), variables, allow_unused=True
    )
    assert first[-2] is None and second[-2] is None
    assert first[-1] is not None and first[-1].item() == 0
    assert second[-1] is not None and second[-1].item() == 0
    return (output,) + first + second


def _compare(actual, expected, values, tolerance):
    cotangent = torch.randn_like(actual(*values))
    first = _derivatives(actual, values, cotangent)
    second = _derivatives(expected, values, cotangent)
    for value, target in zip(first, second):
        assert (value is None) == (target is None)
        if value is not None:
            torch.testing.assert_close(value, target, rtol=tolerance, atol=tolerance)


def _arguments(module, layout="flat"):
    n, m, w = module.irreps_in1.dim, module.irreps_in2.dim, module.weight_numel
    if layout == "broadcast":
        shapes = ((2, 1, n), (1, 3, m), (2, 3, w))
    elif layout == "empty":
        shapes = ((0, 2, n), (1, 2, m), (1, 2, w))
    elif layout == "noncontiguous":
        return tuple((0.1 * torch.randn(width, 3, 2)).permute(2, 1, 0).requires_grad_() for width in (n, m, w))
    else:
        shapes = ((2, n), (2, m), (2, w))
    return tuple((0.1 * torch.randn(shape)).requires_grad_() for shape in shapes)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_uvu_dense_high_order(optimize, specialized, duplicate, float_tolerance):
    with _raw_graphs():
        module = _module(optimize=optimize, specialized=specialized, duplicate=duplicate)
    assert "torch.functional.split" in module._compiled_main_left_right.code
    assert "torch.stack" in module._compiled_main_left_right.code
    _compare(module, lambda *args: _dense(module, *args), _arguments(module), float_tolerance)


@pytest.mark.parametrize("layout", ["broadcast", "noncontiguous", "empty"])
def test_uvu_layout_high_order(layout, float_tolerance):
    module = _module(duplicate=True)
    values = _arguments(module, layout)
    _compare(module, lambda *args: _dense(module, *args), values, float_tolerance)


def test_uvu_nonleaf_weights_and_alias(float_tolerance):
    module = o3.TensorProduct(
        "1x1o",
        "1x1o",
        [(1, (ell, 1, 1)) for ell in (0, 2) for _ in range(4)],
        [(0, 0, index, "uvu", True) for index in reversed(range(8))],
        shared_weights=False,
        internal_weights=False,
    )
    assert (
        codegen._uvu_assembly_groups(
            module.irreps_in1, module.irreps_in2, module.irreps_out, module.instructions, module.shared_weights
        )
        is not None
    )
    x = (0.1 * torch.randn(2, 3)).requires_grad_()
    weights = (0.1 * torch.randn(1, module.weight_numel)).requires_grad_()
    _compare(lambda a, w: module(a, a, 2 * w), lambda a, w: _dense(module, a, a, 2 * w), (x, weights), float_tolerance)


def test_uvu_minimum_profitable_gradcheck():
    module = o3.TensorProduct(
        "1x1o",
        "1x1o",
        [(1, (1, 1, 1))] * 5,
        [(0, 0, index, "uvu", True) for index in range(5)],
        shared_weights=False,
        internal_weights=False,
    ).double()
    assert (
        codegen._uvu_assembly_groups(
            module.irreps_in1, module.irreps_in2, module.irreps_out, module.instructions, module.shared_weights
        )
        is not None
    )
    values = tuple(
        (0.1 * torch.randn(1, width, dtype=torch.float64)).requires_grad_() for width in (3, 3, module.weight_numel)
    )
    assert torch.autograd.gradcheck(module, values)
    assert torch.autograd.gradgradcheck(module, values)


def test_uvu_list_weights(float_tolerance):
    module = _module(duplicate=True)
    x, y, weight = _arguments(module, "broadcast")
    views = list(module.weight_views(weight))
    torch.testing.assert_close(module(x, y, views), module(x, y, weight), rtol=float_tolerance, atol=float_tolerance)
    _compare(lambda a, b, w: module(a, b, list(module.weight_views(w))), module, (x, y, weight), float_tolerance)


def test_uvu_zero_scale_retains_zero_gradients(float_tolerance):
    with _raw_graphs():
        module = _module(zero=True)
    assert "torch.stack" in module._compiled_main_left_right.code
    values = _arguments(module)
    _compare(module, lambda *args: _dense(module, *args), values, float_tolerance)
    gradients = torch.autograd.grad(module(*values).sum(), values)
    assert all(gradient is not None and torch.count_nonzero(gradient) == 0 for gradient in gradients)
    assert torch.count_nonzero(module.output_mask) == 0


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_uvu_small_graph_fallback(count):
    with _raw_graphs():

        def construct():
            return o3.TensorProduct(
                "1x1o",
                "1x1o",
                [(1, (1, 1, 1))] * count,
                [(0, 0, index, "uvu", True) for index in range(count)],
                shared_weights=False,
            )

        module = construct()
        with patch.object(codegen, "_uvu_assembly_groups", return_value=None):
            previous = construct()
    assert module._compiled_main_left_right.code == previous._compiled_main_left_right.code


@pytest.mark.parametrize("count", [5, 17, 32])
@pytest.mark.parametrize("ell", [0, 1])
def test_uvu_scalar_outputs_preserve_original_graph(count, ell):
    with _raw_graphs():

        def construct():
            return o3.TensorProduct(
                [(1, (ell, 1, 1))],
                [(1, (ell, 1, 1))],
                [(1, (0, 1, 1))] * count,
                [(0, 0, index, "uvu", True) for index in range(count)],
                shared_weights=False,
            )

        module = construct()
        with patch.object(codegen, "_uvu_assembly_groups", return_value=None):
            previous = construct()
    assert module._compiled_main_left_right.code == previous._compiled_main_left_right.code
    assert list(module.state_dict()) == list(previous.state_dict())
    assert all(torch.equal(value, previous.state_dict()[name]) for name, value in module.state_dict().items())


@pytest.mark.parametrize(
    "reason",
    [
        "shared",
        "unweighted",
        "mixed_mode",
        "missing_output",
        "zero_width",
        "zero_path",
        "empty_instructions",
        "right_multiplicity",
        "singleton_runs",
    ],
)
def test_uvu_ineligible_graph_fallback(reason, float_tolerance):
    def construct():
        if reason == "shared":
            return _module(shared=True)
        left, right = [(1, (1, 1, 1))], [(1, (1, 1, 1))]
        outputs = [(1, (ell, 1, 1)) for ell in (0, 0, 0, 1, 1, 1, 2, 2)]
        instructions = [(0, 0, index, "uvu", True) for index in range(8)]
        if reason == "unweighted":
            instructions[2] = (0, 0, 2, "uvu", False)
        elif reason == "mixed_mode":
            instructions[2] = (0, 0, 2, "uvw", True)
        elif reason == "missing_output":
            instructions.pop(2)
        elif reason == "zero_width":
            outputs.append((0, (0, 1, 1)))
        elif reason == "zero_path":
            left.append((0, (1, 1, 1)))
            outputs.append((0, (0, 1, 1)))
            instructions.append((1, 0, 8, "uvu", True))
        elif reason == "empty_instructions":
            instructions = []
        elif reason == "right_multiplicity":
            right = [(2, (1, 1, 1))]
        elif reason == "singleton_runs":
            left = [(1, (1, 1, 1))]
            right = [(1, (1, 1, 1))]
            outputs = [(1, (index % 3, 1, 1)) for index in range(8)]
        return o3.TensorProduct(left, right, outputs, instructions, internal_weights=False, shared_weights=False)

    with _raw_graphs():
        module = construct()
        with patch.object(codegen, "_uvu_assembly_groups", return_value=None):
            previous = construct()
    assert module._compiled_main_left_right.code == previous._compiled_main_left_right.code
    assert module.state_dict().keys() == previous.state_dict().keys()
    assert all(torch.equal(value, previous.state_dict()[name]) for name, value in module.state_dict().items())


@pytest.mark.parametrize("scripted", [False, True])
def test_uvu_legacy_state_and_weight_order(scripted, float_tolerance):
    with _raw_graphs():
        with patch.object(codegen, "_uvu_assembly_groups", return_value=None):
            previous = _module(duplicate=True, right=True)
        module = _module(duplicate=True, right=True)
    identities = [(name, id(parameter)) for name, parameter in module.named_parameters()]
    module.load_state_dict(previous.state_dict(), strict=True)
    assert identities == [(name, id(parameter)) for name, parameter in module.named_parameters()]
    assert list(module.state_dict()) == list(previous.state_dict())
    assert all(torch.equal(value, previous.state_dict()[name]) for name, value in module.state_dict().items())
    values = _arguments(module)
    views = list(module.weight_views(values[-1]))
    assert torch.equal(torch.cat([value.reshape(2, -1) for value in views], dim=-1), values[-1])
    assert module.instructions == previous.instructions
    if scripted:
        saved = io.BytesIO()
        torch.jit.save(torch.jit.script(module), saved)
        saved.seek(0)
        module = torch.jit.load(saved)
    _compare(module, previous, values, float_tolerance)
    x, y, w = values
    torch.testing.assert_close(
        module(x, y, w), torch.einsum("...i,...ij->...j", x, previous.right(y, w)), rtol=float_tolerance, atol=float_tolerance
    )


def test_uvu_fp32_construction_to_double():
    saved_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with _raw_graphs():
            with patch.object(codegen, "_uvu_assembly_groups", return_value=None):
                previous = _module(specialized=False).double()
            module = _module(specialized=False).double()
        module.load_state_dict(previous.state_dict(), strict=True)
        torch.set_default_dtype(torch.float64)
        _compare(module, previous, _arguments(module), 1e-9)
    finally:
        torch.set_default_dtype(saved_dtype)


def test_uvu_symmetries(float_tolerance):
    module = _module()
    assert_equivariant(
        lambda x, y, w: module(x, y, w),
        irreps_in=[module.irreps_in1, module.irreps_in2, o3.Irreps([(module.weight_numel, (0, 1, 1))])],
        irreps_out=module.irreps_out,
        do_time_reversal=True,
    )


@pytest.mark.skipif(not hasattr(torch, "compile") or not hasattr(torch, "export"), reason="Requires compile/export")
def test_uvu_dynamic_compile_export(float_tolerance):
    torch._dynamo.reset()
    try:
        with _raw_graphs():
            module = _module(count=18, mul=1)
        values = _arguments(module)
        dimension = torch.export.Dim("uvu_batch", min=1, max=8)
        exported = torch.export.export(module, values, dynamic_shapes=({0: dimension},) * 3).module()
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
        for batch in (2, 5):
            args = tuple((0.1 * torch.randn(batch, value.shape[-1])).requires_grad_() for value in values)
            for function in (exported, compiled):
                torch.testing.assert_close(function(*args), module(*args), rtol=float_tolerance, atol=float_tolerance)
                expected = torch.autograd.grad(module(*args).square().sum(), args)
                actual = torch.autograd.grad(function(*args).square().sum(), args)
                for value, target in zip(actual, expected):
                    torch.testing.assert_close(value, target, rtol=float_tolerance, atol=float_tolerance)
    finally:
        torch._dynamo.reset()
