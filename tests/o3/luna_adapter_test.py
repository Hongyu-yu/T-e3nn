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
