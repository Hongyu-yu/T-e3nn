import math

import pytest
import torch
from e3nn import o3
from e3nn.util.test import assert_auto_jitable, assert_equivariant


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_weird_call(parity, time_reversal):
    o3.spherical_harmonics([4, 1, 2, 3, 3, 1, 0], torch.randn(2, 1, 2, 3), False, parity=parity, time_reversal=time_reversal)


def test_weird_irreps():
    # string input
    o3.spherical_harmonics("0ee + 1oe", torch.randn(1, 3), False)

    # Parity test
    # Weird multipliciteis
    irreps = o3.Irreps("1x0ee + 4x1oe + 3x2ee")
    out = o3.spherical_harmonics(irreps, torch.randn(7, 3), True)
    assert out.shape[-1] == irreps.dim

    # Time Reversal test
    # string input
    o3.spherical_harmonics("0ee + 1eo", torch.randn(1, 3), False, parity=False, time_reversal=True)

    # Weird multipliciteis
    irreps = o3.Irreps("1x0ee + 4x1eo + 3x2ee")
    out = o3.spherical_harmonics(irreps, torch.randn(7, 3), True, parity=False, time_reversal=True)
    assert out.shape[-1] == irreps.dim

    # Parity and Time Reversal test together
    # string input
    o3.spherical_harmonics("0ee + 1oo", torch.randn(1, 3), False, parity=True, time_reversal=True)

    # Weird multipliciteis
    irreps = o3.Irreps("1x0ee + 4x1oo + 3x2ee")
    out = o3.spherical_harmonics(irreps, torch.randn(7, 3), True, parity=True, time_reversal=True)
    assert out.shape[-1] == irreps.dim

    # Bad parity
    with pytest.raises(ValueError):
        # L = 1 shouldn't be even for a vector input
        o3.SphericalHarmonics(
            irreps_out="1x0ee + 4x1ee + 3x2ee",
            normalize=True,
            normalization="integral",
            irreps_in="1oe",
        )
        o3.SphericalHarmonics(
            irreps_out="1x0ee + 4x1ee + 3x2ee",
            normalize=True,
            normalization="integral",
            irreps_in="1eo",
            parity=True,
            time_reversal=False,
        )
        o3.SphericalHarmonics(
            irreps_out="1x0ee + 4x1ee + 3x2ee",
            normalize=True,
            normalization="integral",
            irreps_in="1oo",
            parity=True,
            time_reversal=True,
        )
        o3.SphericalHarmonics(
            irreps_out="1x0ee + 4x1ee + 3x2eo",
            normalize=True,
            normalization="integral",
            irreps_in="1ee",
            parity=False,
            time_reversal=False,
        )

    # Good parity but psuedovector input
    _ = o3.SphericalHarmonics(irreps_in="1ee", irreps_out="1x0ee + 4x1ee + 3x2ee", normalize=True)  # spin X spin
    _ = o3.SphericalHarmonics(irreps_in="1eo", irreps_out="1x0ee + 4x1eo + 3x2ee", normalize=True)  # spin
    _ = o3.SphericalHarmonics(irreps_in="1oe", irreps_out="1x0ee + 4x1oe + 3x2ee", normalize=True)  # rij
    _ = o3.SphericalHarmonics(irreps_in="1oo", irreps_out="1x0ee + 4x1oo + 3x2ee", normalize=True)  # rij X spin

    # Invalid input
    with pytest.raises(ValueError):
        _ = o3.SphericalHarmonics(irreps_in="1ee + 3oe", irreps_out="1x0ee + 4x1ee + 3x2ee", normalize=True)  # invalid l


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_zeros(parity, time_reversal):
    assert torch.allclose(
        o3.spherical_harmonics(
            [0, 1], torch.zeros(1, 3), False, normalization="norm", parity=parity, time_reversal=time_reversal
        ),
        torch.tensor([[1, 0, 0, 0.0]]),
    )


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_equivariance(float_tolerance, parity, time_reversal):
    lmax = 5
    p = -1 if parity else 1
    t = -1 if time_reversal else 1
    irreps = o3.Irreps.spherical_harmonics(lmax, p=p, t=t)
    x = torch.randn(2, 3)
    abc = o3.rand_angles()
    y1 = o3.spherical_harmonics(irreps, x @ o3.angles_to_matrix(*abc).T, False, parity=parity, time_reversal=time_reversal)
    y2 = o3.spherical_harmonics(irreps, x, False, parity=parity, time_reversal=time_reversal) @ irreps.D_from_angles(*abc).T

    assert (y1 - y2).abs().max() < 10 * float_tolerance


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_backwardable(parity, time_reversal):
    lmax = 3
    ls = list(range(lmax + 1))

    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0, 0],
            [0.0, 10.0, 0],
            [0.435, 0.7644, 0.023],
        ],
        requires_grad=True,
        dtype=torch.float64,
    )

    def func(pos):
        return o3.spherical_harmonics(ls, pos, False, parity=parity, time_reversal=time_reversal)

    assert torch.autograd.gradcheck(func, (xyz,), check_undefined_grad=False)


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
@pytest.mark.parametrize("l", range(10 + 1))
def test_normalization(float_tolerance, l, parity, time_reversal):

    n = (
        o3.spherical_harmonics(
            l, torch.randn(3), normalize=True, normalization="integral", parity=parity, time_reversal=time_reversal
        )
        .pow(2)
        .mean()
    )
    assert abs(n - 1 / (4 * math.pi)) < float_tolerance

    n = o3.spherical_harmonics(
        l, torch.randn(3), normalize=True, normalization="norm", parity=parity, time_reversal=time_reversal
    ).norm()
    assert abs(n - 1) < float_tolerance

    n = (
        o3.spherical_harmonics(
            l, torch.randn(3), normalize=True, normalization="component", parity=parity, time_reversal=time_reversal
        )
        .pow(2)
        .mean()
    )
    assert abs(n - 1) < float_tolerance


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_closure(parity, time_reversal):
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    x = torch.randn(1_000_000, 3)
    Ys = [o3.spherical_harmonics(l, x, True, parity=parity, time_reversal=time_reversal) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * math.pi
            if l1 == l2:
                i = torch.eye(2 * l1 + 1)
                assert (m - i).abs().max() < 0.01
            else:
                assert m.abs().max() < 0.01


@pytest.mark.parametrize("l", range(11 + 1))
@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_parity_and_time_reversal(float_tolerance, l, parity, time_reversal):
    r"""
    (-1)^l Y(x) = Y(-x)
    """
    x = torch.randn(3)
    Y1 = (-1) ** l * o3.spherical_harmonics(l, x, False, parity=parity, time_reversal=time_reversal)
    Y2 = o3.spherical_harmonics(l, -x, False, parity=parity, time_reversal=time_reversal)
    assert (Y1 - Y2).abs().max() < float_tolerance


@pytest.mark.parametrize("l", range(9 + 1))
@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_recurrence_relation(float_tolerance, l, parity, time_reversal):
    if torch.get_default_dtype() != torch.float64 and l > 6:
        pytest.xfail("we expect this to fail for high l and single precision")

    x = torch.randn(3, requires_grad=True)

    a = o3.spherical_harmonics(l + 1, x, False, parity=parity, time_reversal=time_reversal)

    b = torch.einsum(
        "ijk,j,k->i",
        o3.wigner_3j(l + 1, l, 1),
        o3.spherical_harmonics(l, x, False, parity=parity, time_reversal=time_reversal),
        x,
    )

    alpha = b.norm() / a.norm()

    assert (a / a.norm() - b / b.norm()).abs().max() < 10 * float_tolerance

    def f(x):
        return o3.spherical_harmonics(l + 1, x, False, parity=parity, time_reversal=time_reversal)

    a = torch.autograd.functional.jacobian(f, x)

    b = (
        (l + 1)
        / alpha
        * torch.einsum(
            "ijk,j->ik",
            o3.wigner_3j(l + 1, l, 1),
            o3.spherical_harmonics(l, x, False, parity=parity, time_reversal=time_reversal),
        )
    )

    assert (a - b).abs().max() < 100 * float_tolerance


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("time_reversal", [True, False])
def test_module(normalization, normalize, parity, time_reversal):
    if parity:
        if time_reversal:
            l = o3.Irreps("0ee + 1oo + 3oo")
        else:
            l = o3.Irreps("0ee + 1oe + 3oe")
    else:
        if time_reversal:
            l = o3.Irreps("0ee + 1eo + 3eo")
        else:
            l = o3.Irreps("0ee + 1ee + 3ee")

    sp = o3.SphericalHarmonics(l, normalize, normalization, parity=parity, time_reversal=time_reversal)
    sp_jit = assert_auto_jitable(sp)
    xyz = torch.randn(11, 3)
    assert torch.allclose(
        sp_jit(xyz), o3.spherical_harmonics(l, xyz, normalize, normalization, parity=parity, time_reversal=time_reversal)
    )
    assert_equivariant(sp)
