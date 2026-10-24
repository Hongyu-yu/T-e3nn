import pytest
import torch

from e3nn.util import test as test_utils
from e3nn.util.test import assert_equivariant, equivariance_error, format_equivariance_error


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -float("inf"), -1.0])
@pytest.mark.parametrize("tensor", [False, True])
def test_invalid_tolerance_is_rejected_before_argument_inference(tolerance, tensor, monkeypatch):
    def unexpected_inference(*args, **kwargs):
        pytest.fail("Invalid tolerance must not trigger argument inference or model execution")

    monkeypatch.setattr(test_utils, "_get_args_in", unexpected_inference)
    if tensor:
        tolerance = torch.tensor(tolerance)
    with pytest.raises(ValueError, match="tolerance must be finite and nonnegative"):
        assert_equivariant(lambda x: x, tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 1e-5, 2.0])
@pytest.mark.parametrize("tensor", [False, True])
def test_finite_nonnegative_tolerance_accepts_exact_equivariance(tolerance, tensor):
    if tensor:
        tolerance = torch.tensor(tolerance)
    errors = assert_equivariant(
        lambda x: x,
        args_in=[torch.tensor([[0.3], [-0.7]])],
        irreps_in=["0eo"],
        irreps_out="0eo",
        do_time_reversal=True,
        tolerance=tolerance,
    )
    assert all(torch.equal(error, torch.zeros_like(error)) for error in errors.values())


@pytest.mark.parametrize("tolerance", [0.0, 1e-5])
def test_zero_and_small_tolerance_reject_time_reversal_error(tolerance):
    with pytest.raises(AssertionError, match="tr_k=1"):
        assert_equivariant(
            lambda x: x.square(),
            args_in=[torch.tensor([[0.3], [-0.7]])],
            irreps_in=["0eo"],
            irreps_out="0eo",
            do_time_reversal=True,
            tolerance=tolerance,
        )


def test_default_tolerance_preserves_current_dtype_behavior(float_tolerance):
    assert torch.get_default_dtype() == float_tolerance.dtype
    kwargs = dict(
        args_in=[torch.tensor([[0.3], [-0.7]])],
        irreps_in=["0eo"],
        irreps_out="0eo",
        do_time_reversal=True,
    )

    def small_time_reversal_error(x):
        return x + float_tolerance * 0.2

    default = assert_equivariant(small_time_reversal_error, **kwargs)
    explicit = assert_equivariant(small_time_reversal_error, tolerance=float_tolerance, **kwargs)
    assert default.keys() == explicit.keys()
    assert all(torch.equal(default[key], explicit[key]) for key in default)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_call", [None, 1, 2, 3])
def test_nonfinite_output_cannot_pass_equivariance(value, bad_call):
    calls = [0]

    def output(x):
        calls[0] += 1
        result = x.clone()
        if bad_call is None or calls[0] == bad_call:
            result[0, 0] = value
        return x, result

    kwargs = dict(
        args_in=[torch.ones(2, 1)],
        irreps_in=["0ee"],
        irreps_out=["0ee", "0ee"],
        do_time_reversal=True,
        ntrials=2,
    )
    errors = equivariance_error(output, **kwargs)
    assert any(not torch.isfinite(error).all() for error in errors.values())
    calls[0] = 0
    with pytest.raises(AssertionError, match="parity_k=.*tr_k=.*only_rot_spin="):
        assert_equivariant(output, **kwargs)


@pytest.mark.parametrize("irreps", ["0ee", "0eo", "0oe", "0oo"])
def test_finite_identity_passes_all_parity_time_reversal_combinations(irreps, float_tolerance):
    errors = assert_equivariant(
        lambda x: x,
        args_in=[torch.tensor([[0.3], [-0.7]])],
        irreps_in=[irreps],
        irreps_out=irreps,
        do_time_reversal=True,
        do_only_rot_spin=True,
        ntrials=2,
    )
    assert set(errors) == {(p, False, t, spin) for p in (0, 1) for t in (0, 1) for spin in (False, True)}
    assert all(torch.isfinite(error).all() and error.max() <= float_tolerance for error in errors.values())


def test_time_reversal_violation_reports_its_context(float_tolerance):
    kwargs = dict(args_in=[torch.tensor([[0.3], [-0.7]])], irreps_in=["0eo"], irreps_out="0eo")
    assert_equivariant(lambda x: x.square(), do_time_reversal=False, **kwargs)
    errors = equivariance_error(lambda x: x.square(), do_time_reversal=True, **kwargs)
    assert all(error.max() <= float_tolerance for case, error in errors.items() if case[2] == 0)
    assert all(error.max() > float_tolerance for case, error in errors.items() if case[2] == 1)
    with pytest.raises(AssertionError, match="tr_k=1"):
        assert_equivariant(lambda x: x.square(), do_time_reversal=True, **kwargs)


@pytest.mark.parametrize("ntrials", [0, -1])
def test_equivariance_requires_a_measured_trial(ntrials):
    with pytest.raises(ValueError, match="ntrials"):
        assert_equivariant(
            lambda x: x,
            args_in=[torch.ones(2, 1)],
            irreps_in=["0ee"],
            irreps_out="0ee",
            ntrials=ntrials,
        )


def test_equivariance_rejects_empty_measurement():
    with pytest.raises(ValueError, match="output|component"):
        assert_equivariant(lambda x: x, args_in=[torch.empty(0, 1)], irreps_in=["0ee"], irreps_out="0ee")


def test_error_format_preserves_legacy_keys_and_reports_full_transform():
    old = format_equivariance_error({(1, False): torch.tensor([0.0, 0.5])})
    assert "parity_k=1, did_translate=False" in old
    assert "argument 1" in old
    current = format_equivariance_error({(0, True, 1, True): torch.tensor([0.5])})
    assert "parity_k=0, did_translate=True, tr_k=1, only_rot_spin=True" in current
