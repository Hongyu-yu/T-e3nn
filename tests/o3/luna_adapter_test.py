import pytest
import torch
from types import SimpleNamespace

from e3nn import o3


def test_luna_adapter_rejects_incompatible_time_reversal_path():
    tensor_product = SimpleNamespace(
        irreps_in1=[SimpleNamespace(ir=SimpleNamespace(t=1))],
        irreps_in2=[SimpleNamespace(ir=SimpleNamespace(t=1))],
        irreps_out=[SimpleNamespace(ir=SimpleNamespace(t=-1))],
        instructions=[SimpleNamespace(i_in1=0, i_in2=0, i_out=0)],
    )
    with pytest.raises(ValueError, match="time-reversal-compatible"):
        o3.validate_luna_tensor_product(tensor_product)


@pytest.mark.skipif(not o3.luna_available(), reason="optional LunaEquivariance is not installed")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_luna_adapter_preserves_checkpoint_keys_and_outputs(dtype):
    tensor_product = o3.TensorProduct(
        "4x0e",
        "1x1o",
        "4x1o",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(dtype=dtype)
    adapter = o3.LunaTensorProduct(tensor_product).to(dtype=dtype)
    assert set(adapter.state_dict()) == set(tensor_product.state_dict())
    assert adapter.irreps_out == tensor_product.irreps_out


def test_luna_conv_contract_rejects_unsorted_destinations():
    with pytest.raises(ValueError, match="sorted"):
        o3.LunaTensorProductConv.validate_destination_order(torch.tensor([0, 2, 1]))


def test_luna_conv_contract_rejects_mixed_modes():
    tensor_product = SimpleNamespace(
        irreps_in1=[SimpleNamespace(mul=1, ir=SimpleNamespace(t=1))],
        irreps_in2=[SimpleNamespace(mul=1, ir=SimpleNamespace(t=1))],
        irreps_out=[SimpleNamespace(mul=1, ir=SimpleNamespace(t=1))],
        instructions=[
            SimpleNamespace(i_in1=0, i_in2=0, i_out=0, connection_mode="uuu", has_weight=False),
            SimpleNamespace(i_in1=0, i_in2=0, i_out=0, connection_mode="uvu", has_weight=True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    with pytest.raises(ValueError, match="one connection mode"):
        o3.validate_luna_tensor_product_conv(tensor_product)


@pytest.mark.skipif(not o3.luna_available(), reason="optional LunaEquivariance is not installed")
def test_luna_conv_preserves_checkpoint_keys():
    tensor_product = o3.TensorProduct(
        "4x0e",
        "4x1o",
        "4x1o",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    adapter = o3.LunaTensorProductConv(tensor_product)
    assert set(adapter.state_dict()) == set(tensor_product.state_dict())
    assert adapter.irreps_out == tensor_product.irreps_out
