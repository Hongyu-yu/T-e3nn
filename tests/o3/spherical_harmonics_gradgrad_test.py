import io

import pytest
import torch

from e3nn import o3


def _value_and_derivatives(function, value):
    value = value.clone().requires_grad_()
    output = function(value)
    loss = (output.sin() + 0.03 * output.square()).sum()
    first = torch.autograd.grad(loss, value, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), value)[0]
    return output, first, second


@pytest.mark.parametrize("lmax", [1, 2, 3])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("parity,time_reversal", [(False, False), (False, True), (True, False), (True, True)])
def test_clamped_normalization_derivatives_match_linear_input(lmax, normalization, parity, time_reversal, float_tolerance):
    module = o3.SphericalHarmonics(
        list(range(lmax + 1)),
        True,
        normalization,
        parity=parity,
        time_reversal=time_reversal,
    )
    polynomial = o3.SphericalHarmonics(
        list(range(lmax + 1)),
        False,
        normalization,
        parity=parity,
        time_reversal=time_reversal,
    )
    value = torch.tensor([[0.0, 0.0, 0.0], [1e-30, -1e-30, 1e-30], [2e-13, -3e-13, 1e-13]])
    actual = _value_and_derivatives(module, value)
    expected = _value_and_derivatives(lambda x: polynomial(x / 1e-12), value)
    for candidate, reference, scale in zip(actual, expected, [1.0, 1e-12, 1e-36]):
        assert torch.isfinite(candidate).all()
        assert torch.isfinite(reference).all()
        # Dimensionless comparisons keep the existing dtype tolerances meaningful near eps.
        assert torch.allclose(candidate * scale, reference * scale, atol=float_tolerance, rtol=float_tolerance)


@pytest.mark.parametrize("point", [[0.0, 0.0, 0.0], [0.2e-12, -0.3e-12, 0.1e-12]])
def test_normalized_zero_neighborhood_gradcheck(point):
    module = o3.SphericalHarmonics([0, 1, 2], True, "component")
    value = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    # The finite-difference perturbation must stay inside the original eps neighborhood.
    assert torch.autograd.gradcheck(module, (value,), eps=1e-15)
    cotangent = torch.linspace(-0.7, 0.9, 9, dtype=torch.float64)
    assert torch.autograd.gradgradcheck(module, (value,), grad_outputs=cotangent, eps=1e-15)
    # A dimensionless input also checks cotangent derivatives without perturbing an
    # O(1) cotangent by only 1e-15; the physical input step here is 1e-18.
    unit_value = (torch.tensor(point, dtype=torch.float64) / 1e-12).requires_grad_()
    assert torch.autograd.gradgradcheck(lambda x: module(1e-12 * x), (unit_value,))


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_nonzero_values_and_derivatives_match_original_normalization(normalization, float_tolerance):
    module = o3.SphericalHarmonics([0, 1, 2, 3], True, normalization)
    polynomial = o3.SphericalHarmonics([0, 1, 2, 3], False, normalization)
    direction = torch.tensor([[1.0, -0.3, 0.5], [0.7, 0.4, -1.0]])
    for magnitude in [5e-13, 9e-13, 1e-12, 2e-12, 1e-9, 1e-6, 1.0, 1e6, 1e12]:
        value = direction * magnitude
        actual = _value_and_derivatives(module, value)
        expected = _value_and_derivatives(lambda x: polynomial(torch.nn.functional.normalize(x, dim=-1)), value)
        for candidate, reference, power in zip(actual, expected, [0, 1, 3]):
            assert torch.isfinite(candidate).all()
            assert torch.isfinite(reference).all()
            assert torch.allclose(
                candidate * magnitude**power,
                reference * magnitude**power,
                atol=float_tolerance,
                rtol=float_tolerance,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_forward_keeps_original_normalization(dtype):
    module = o3.SphericalHarmonics([0, 1, 2, 3], True, "component")
    polynomial = o3.SphericalHarmonics([0, 1, 2, 3], False, "component")
    value = torch.tensor([[0, 0, 0], [1, -0.3, 0.5], [1000, -300, 500]], dtype=dtype)
    expected = polynomial(torch.nn.functional.normalize(value, dim=-1))
    assert torch.allclose(module(value), expected, atol=0, rtol=0, equal_nan=True)


def test_zero_rows_unused_by_gather_do_not_contaminate_derivatives(float_tolerance):
    module = o3.SphericalHarmonics([0, 1, 2, 3], True, "component")
    value = torch.tensor([[0.2, 0.6, -0.5], [0.9, -0.7, 0.4], [0.0, 0.0, 0.0]])
    index = torch.tensor([0, 1, 0])
    actual = _value_and_derivatives(lambda x: module(x)[index], value)
    expected = _value_and_derivatives(lambda x: module(x[index]), value)
    for candidate, reference in zip(actual, expected):
        assert torch.isfinite(candidate).all()
        assert torch.allclose(candidate, reference, atol=float_tolerance, rtol=float_tolerance)


def test_zero_derivatives_survive_torchscript_roundtrip(float_tolerance):
    module = o3.SphericalHarmonics([0, 1, 2, 3], True, "component")
    scripted = torch.jit.script(module)
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    stream.seek(0)
    restored = torch.jit.load(stream)
    value = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.3, 0.5]])
    reference = _value_and_derivatives(module, value)
    for implementation in [scripted, restored]:
        for actual, expected, scale in zip(_value_and_derivatives(implementation, value), reference, [1, 1e-12, 1e-36]):
            assert torch.isfinite(actual).all()
            assert torch.allclose(actual * scale, expected * scale, atol=float_tolerance, rtol=float_tolerance)
