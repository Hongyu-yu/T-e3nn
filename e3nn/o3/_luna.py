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


def validate_luna_tensor_product_conv(tp: torch.nn.Module) -> str:
    """Validate the single-mode graph-convolution subset implemented by Luna."""
    validate_luna_tensor_product(tp)
    modes = {instruction.connection_mode for instruction in tp.instructions}
    if len(modes) != 1 or not modes.issubset({"uuu", "uvu"}):
        raise ValueError(
            "LunaTensorProductConv requires one connection mode, either 'uuu' or 'uvu'"
        )
    mode = next(iter(modes))
    for instruction in tp.instructions:
        if mode == "uuu":
            mul_in1 = tp.irreps_in1[instruction.i_in1].mul
            mul_in2 = tp.irreps_in2[instruction.i_in2].mul
            mul_out = tp.irreps_out[instruction.i_out].mul
            if not (mul_in1 == mul_in2 == mul_out):
                raise ValueError("Luna UUU paths require equal input and output multiplicities")
        elif not instruction.has_weight:
            raise ValueError("Luna UVU paths require explicit weights")
    if mode == "uvu" and (tp.internal_weights or tp.shared_weights):
        raise ValueError(
            "Luna UVU convolution requires external per-edge weights "
            "(internal_weights=False, shared_weights=False)"
        )
    return mode


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


class LunaTensorProductConv(torch.nn.Module):
    """Opt-in fused Luna graph convolution around an existing T-e3nn TP.

    The source tensor product remains the sole registered parameter owner, so
    checkpoints retain exactly the same local keys as the source module.
    Destination indices must be sorted ascending, and all edge tensors must
    use the same order.
    """

    def __init__(self, tensor_product, *, dtype=None):
        super().__init__()
        mode = validate_luna_tensor_product_conv(tensor_product)
        luna = importlib.import_module("LunaEquivariance")
        if not hasattr(luna, "TensorProductConv"):
            raise ImportError("Installed LunaEquivariance has no TensorProductConv adapter")
        object.__setattr__(self, "_reference", tensor_product)
        for name, parameter in tensor_product._parameters.items():
            if parameter is not None:
                self.register_parameter(name, parameter)
        for name, buffer in tensor_product._buffers.items():
            if buffer is not None:
                self.register_buffer(name, buffer)
        # Keep the optional backend out of Module._modules: its reference to
        # the same TP would otherwise duplicate every checkpoint key.
        object.__setattr__(self, "_luna", luna.TensorProductConv(tensor_product, dtype=dtype))
        self.mode = mode
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.instructions = tensor_product.instructions
        self.weight_numel = tensor_product.weight_numel
        self.internal_weights = tensor_product.internal_weights
        self.shared_weights = tensor_product.shared_weights
        self.train(tensor_product.training)

    @staticmethod
    def validate_destination_order(dst):
        """Reject unsorted destination indices before Luna graph construction."""
        if not isinstance(dst, torch.Tensor) or dst.ndim != 1:
            raise TypeError("dst must be a one-dimensional tensor")
        if dst.dtype not in (torch.int32, torch.int64):
            raise TypeError("dst must use int32 or int64 indices")
        if dst.numel() > 1 and bool(torch.any(dst[1:] < dst[:-1])):
            raise ValueError(
                "LunaTensorProductConv requires destination indices sorted in ascending order"
            )

    @staticmethod
    def create_graph(src, dst, num_nodes=None, **kwargs):
        """Build a Luna graph after checking the destination ordering contract."""
        LunaTensorProductConv.validate_destination_order(dst)
        if not isinstance(src, torch.Tensor) or src.ndim != 1:
            raise TypeError("src must be a one-dimensional tensor")
        if src.dtype not in (torch.int32, torch.int64):
            raise TypeError("src must use int32 or int64 indices")
        luna = importlib.import_module("LunaEquivariance")
        return luna.make_graph(src, dst, num_nodes, **kwargs)

    @property
    def last_execution_backend(self):
        return getattr(self._luna, "last_execution_backend", None)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        for name, parameter in self._parameters.items():
            self._reference._parameters[name] = parameter
        for name, buffer in self._buffers.items():
            self._reference._buffers[name] = buffer
        luna = importlib.import_module("LunaEquivariance")
        object.__setattr__(self, "_luna", luna.TensorProductConv(self._reference, dtype=self.weight.dtype))
        return result

    def forward(
        self,
        x,
        edge_attr,
        weight=None,
        *,
        graph=None,
        src=None,
        dst=None,
        num_nodes=None,
        env=None,
        return_l0_feat=False
    ):
        if graph is None:
            if src is None or dst is None:
                raise TypeError("Provide graph= or both src= and dst= for Luna convolution")
            graph = self.create_graph(src, dst, x.shape[0] if num_nodes is None else num_nodes)
        elif dst is not None:
            self.validate_destination_order(dst)
        return self._luna(
            x,
            edge_attr,
            weight,
            graph=graph,
            env=env,
            return_l0_feat=return_l0_feat,
        )


__all__ = [
    "LunaTensorProduct",
    "LunaTensorProductConv",
    "luna_available",
    "validate_luna_tensor_product",
    "validate_luna_tensor_product_conv",
]
