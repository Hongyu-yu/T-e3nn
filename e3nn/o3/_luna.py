"""Optional LunaEquivariance adapter for existing e3nn tensor products."""

from __future__ import annotations

import importlib
from typing import Any, Optional

import torch


def _time_reversal(irrep: Any) -> Optional[int]:
    """Return a fork-specific time-reversal sign when the Irrep exposes one."""
    value = getattr(irrep, "t", None)
    return None if value is None else int(value)


def validate_luna_tensor_product(tp: torch.nn.Module) -> None:
    """Validate time-reversal products before handing a TP to Luna."""
    for instruction in tp.instructions:
        irrep1 = tp.irreps_in1[instruction.i_in1].ir
        irrep2 = tp.irreps_in2[instruction.i_in2].ir
        irrep_out = tp.irreps_out[instruction.i_out].ir
        t1, t2, tout = (_time_reversal(irrep) for irrep in (irrep1, irrep2, irrep_out))
        if t1 is not None and t2 is not None and tout is not None and t1 * t2 != tout:
            raise ValueError(
                "LunaTensorProduct requires time-reversal-compatible instructions: "
                f"{t1} * {t2} != {tout} for instruction {instruction}"
            )


def luna_available() -> bool:
    """Return whether the optional LunaEquivariance package is importable."""
    try:
        importlib.import_module("LunaEquivariance")
    except ModuleNotFoundError:
        return False
    return True


class LunaTensorProduct(torch.nn.Module):
    """Opt-in Luna implementation preserving an existing TP's public state."""

    def __init__(self, tensor_product: torch.nn.Module):
        super().__init__()
        required = ("irreps_in1", "irreps_in2", "irreps_out", "instructions")
        if not all(hasattr(tensor_product, name) for name in required):
            raise TypeError("LunaTensorProduct requires an e3nn TensorProduct module")
        validate_luna_tensor_product(tensor_product)
        luna = importlib.import_module("LunaEquivariance")
        if not hasattr(luna, "TensorProduct"):
            raise ImportError("Installed LunaEquivariance has no TensorProduct adapter")

        object.__setattr__(self, "_reference", tensor_product)
        for name, parameter in tensor_product._parameters.items():
            if parameter is not None:
                self.register_parameter(name, parameter)
        for name, buffer in tensor_product._buffers.items():
            if buffer is not None:
                self.register_buffer(name, buffer)
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.instructions = tensor_product.instructions
        self.weight_numel = tensor_product.weight_numel
        self.internal_weights = tensor_product.internal_weights
        self.shared_weights = tensor_product.shared_weights
        object.__setattr__(self, "_luna", luna.TensorProduct(tensor_product))

    def _sync_reference_state(self) -> None:
        for name, parameter in self._parameters.items():
            self._reference._parameters[name] = parameter
        for name, buffer in self._buffers.items():
            self._reference._buffers[name] = buffer

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._sync_reference_state()
        luna = importlib.import_module("LunaEquivariance")
        object.__setattr__(self, "_luna", luna.TensorProduct(self._reference))
        return result

    @property
    def last_execution_backend(self):
        return getattr(self._luna, "last_execution_backend", None)

    def forward(self, x, y, weight=None):
        if weight is None and self.internal_weights:
            weight = self.weight
        return self._luna(x, y, weight)


__all__ = ["LunaTensorProduct", "luna_available", "validate_luna_tensor_product"]
