"""MPS implementation of comfy-kitchen's ``int8_linear`` contract.

The backend uses PyTorch's native MPS ``aten::_weight_int8pack_mm`` kernel.
That operator keeps the stored weights in INT8 and performs the dequantizing
matmul inside the Metal kernel, avoiding a full floating-point weight copy.

Two activation modes are supported:

``dynamic`` (default)
    Match comfy-kitchen's eager contract as closely as the native weight-only
    kernel permits: apply the optional activation and ConvRot, dynamically
    quantize each activation row to INT8, convert those small activation rows
    to the output dtype, execute the native weight-INT8 kernel, and apply the
    activation row scales.

``weight_only``
    Skip dynamic activation quantization and execute the dequantized-weight
    linear directly. This is less lossy and uses fewer activation temporaries,
    but it is not numerically identical to comfy-kitchen's eager INT8 path.

Select the mode before starting ComfyUI with
``COMFY_MPS_INT8_ACTIVATION_MODE=dynamic|weight_only``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
import sys
import threading
from typing import Any

import torch

from comfy_kitchen.backends import _activations
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


apply_input_act = _activations.apply_input_act
apply_residual = getattr(_activations, "apply_residual", None)


LOGGER = logging.getLogger("ComfyUI-MPS-INT8")

BACKEND_NAME = "mps_int8_native"
SUPPORTED_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)
SUPPORTED_ACTIVATION_MODES = frozenset({"dynamic", "weight_only"})


def _configured_activation_mode() -> str:
    value = os.environ.get("COMFY_MPS_INT8_ACTIVATION_MODE", "dynamic")
    mode = value.strip().lower()
    if mode not in SUPPORTED_ACTIVATION_MODES:
        expected = ", ".join(sorted(SUPPORTED_ACTIVATION_MODES))
        raise ValueError(
            "COMFY_MPS_INT8_ACTIVATION_MODE must be one of "
            f"{expected}; got {value!r}"
        )
    return mode


ACTIVATION_MODE = _configured_activation_mode()


@dataclass
class _BackendState:
    registered: bool = False
    available: bool = False
    reason: str = "not initialized"
    calls: int = 0
    first_call_logged: bool = False


_STATE = _BackendState()
_STATE_LOCK = threading.Lock()


def _has_native_mps_kernel() -> bool:
    if not hasattr(torch, "_weight_int8pack_mm"):
        return False
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(  # type: ignore[attr-defined]
            "aten::_weight_int8pack_mm", "MPS"
        )
    except (AttributeError, RuntimeError):
        return False


def _mps_runtime_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _build_constraints() -> FunctionConstraints:
    cpu_or_mps = frozenset({"cpu", "mps"})
    return FunctionConstraints(
        params={
            "x": ParamConstraint(
                dtypes=SUPPORTED_FLOAT_DTYPES,
                devices=frozenset({"mps"}),
            ),
            "weight": ParamConstraint(
                dtypes=frozenset({torch.int8}),
                devices=cpu_or_mps,
            ),
            "weight_scale": ParamConstraint(
                dtypes=SUPPORTED_FLOAT_DTYPES,
                devices=cpu_or_mps,
            ),
            "bias": ParamConstraint(
                dtypes=SUPPORTED_FLOAT_DTYPES,
                devices=cpu_or_mps,
            ),
            "out_dtype": ParamConstraint(dtypes=SUPPORTED_FLOAT_DTYPES),
            "convrot": ParamConstraint(dtypes=frozenset({bool})),
            "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            "input_act": ParamConstraint(dtypes=frozenset({str})),
        },
        # Tensor arguments above carry explicit device rules. The wildcard
        # prevents non-tensor parameters from inheriting a needless rule.
        default_devices=frozenset({"*"}),
    )


def _insert_backend_priority(name: str) -> None:
    current = list(getattr(registry, "_priority", ["cuda", "triton", "eager"]))
    current = [item for item in current if item != name]
    try:
        eager_index = current.index("eager")
    except ValueError:
        current.append(name)
    else:
        current.insert(eager_index, name)
    registry.set_priority(current)


def register_backend() -> bool:
    """Register the native MPS INT8 backend when the runtime supports it.

    The function is safe to call repeatedly, which matters because ComfyUI may
    reload custom-node modules during development.
    """

    with _STATE_LOCK:
        if registry.is_available(BACKEND_NAME):
            _insert_backend_priority(BACKEND_NAME)
            _STATE.registered = True
            _STATE.available = True
            _STATE.reason = "registered"
            return True

        if not _mps_runtime_available():
            _STATE.registered = False
            _STATE.available = False
            _STATE.reason = "MPS runtime is not available in this process"
            LOGGER.info("Backend inactive: %s", _STATE.reason)
            return False

        if not _has_native_mps_kernel():
            _STATE.registered = False
            _STATE.available = False
            _STATE.reason = (
                "PyTorch has no MPS kernel for aten::_weight_int8pack_mm"
            )
            LOGGER.warning("Backend inactive: %s", _STATE.reason)
            return False

        registry.register(
            name=BACKEND_NAME,
            module=sys.modules[__name__],
            capabilities={"int8_linear": _build_constraints()},
        )
        _insert_backend_priority(BACKEND_NAME)

        _STATE.registered = True
        _STATE.available = True
        _STATE.reason = "registered"
        LOGGER.info(
            "Registered %s backend with activation_mode=%s (torch=%s)",
            BACKEND_NAME,
            ACTIVATION_MODE,
            torch.__version__,
        )
        return True


def _normalize_weight_scale(
    weight_scale: torch.Tensor,
    out_features: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scales = weight_scale.to(device=device, dtype=dtype).reshape(-1)
    if scales.numel() == 1:
        scales = scales.expand(out_features)
    elif scales.numel() != out_features:
        raise ValueError(
            "INT8 weight scale must be scalar or per-output-channel; "
            f"got {tuple(weight_scale.shape)} for {out_features} outputs"
        )

    # The current native MPS kernel reads scales as the activation dtype even
    # though the operator's meta function does not enforce that contract.
    # A contiguous, dtype-matched vector avoids corrupted FP16/BF16 results.
    return scales.contiguous()


def _validate_native_shapes(x_2d: torch.Tensor, weight: torch.Tensor) -> None:
    if x_2d.dim() != 2:
        raise ValueError(f"INT8 linear input must flatten to 2D, got {x_2d.dim()}D")
    if weight.dim() != 2:
        raise ValueError(f"INT8 linear weight must be 2D, got {weight.dim()}D")
    if weight.dtype is not torch.int8:
        raise TypeError(f"INT8 linear weight must be torch.int8, got {weight.dtype}")
    if x_2d.shape[1] != weight.shape[1]:
        raise ValueError(
            "Input and weight inner dimensions must match; got "
            f"{x_2d.shape[1]} and {weight.shape[1]}"
        )

    out_features, in_features = weight.shape
    if out_features % 32 or in_features % 32:
        raise RuntimeError(
            "The native MPS INT8 kernel requires output and input features "
            "to be divisible by 32; got weight shape "
            f"{tuple(weight.shape)}. Refusing to fall through to aten::_int_mm."
        )


def _native_weight_int8_mm(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    weight_mps = weight.to(device=x_2d.device).contiguous()
    scales = _normalize_weight_scale(
        weight_scale,
        weight_mps.shape[0],
        device=x_2d.device,
        dtype=x_2d.dtype,
    )
    return torch._weight_int8pack_mm(x_2d.contiguous(), weight_mps, scales)


def _int8_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    *,
    activation_mode: str,
) -> torch.Tensor:
    if x.device.type != "mps":
        raise RuntimeError(
            f"{BACKEND_NAME} requires an MPS input tensor, got {x.device}"
        )
    if activation_mode not in SUPPORTED_ACTIVATION_MODES:
        raise ValueError(f"Unsupported activation mode: {activation_mode!r}")

    if out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype not in SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"Unsupported output dtype for MPS INT8 linear: {out_dtype}")

    activated = apply_input_act(x, input_act)
    if convrot:
        if activated.shape[-1] % convrot_groupsize:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide "
                f"input features {activated.shape[-1]}"
            )
        hadamard = _build_hadamard(
            convrot_groupsize,
            device=activated.device,
            dtype=activated.dtype,
        )
        activated = _rotate_activation(
            activated,
            hadamard,
            convrot_groupsize,
        )

    original_shape = activated.shape
    x_2d = activated.reshape(-1, activated.shape[-1])
    _validate_native_shapes(x_2d, weight)

    if activation_mode == "dynamic":
        quantized_x, row_scale = quantize_int8_rowwise(x_2d)
        native_input = quantized_x.to(dtype=out_dtype).contiguous()
        result = _native_weight_int8_mm(native_input, weight, weight_scale)
        result.mul_(row_scale.to(device=result.device, dtype=result.dtype))
    else:
        native_input = x_2d.to(dtype=out_dtype).contiguous()
        result = _native_weight_int8_mm(native_input, weight, weight_scale)

    if bias is not None:
        if bias.numel() != weight.shape[0]:
            raise ValueError(
                f"Bias must have {weight.shape[0]} elements, got {bias.numel()}"
            )
        result.add_(
            bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
        )

    return result.reshape(*original_shape[:-1], weight.shape[0])


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute a comfy-kitchen INT8 linear call with the native MPS kernel."""

    if apply_residual is not None and input_act == "rms_norm":
        x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
        input_act = None

    result = _int8_linear_impl(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        input_act=input_act,
        activation_mode=ACTIVATION_MODE,
    )
    if apply_residual is not None:
        result = apply_residual(result, residual, residual_scale)

    with _STATE_LOCK:
        _STATE.calls += 1
        if not _STATE.first_call_logged:
            LOGGER.info(
                "First MPS INT8 linear call: x=%s weight=%s out=%s mode=%s",
                tuple(x.shape),
                tuple(weight.shape),
                result.dtype,
                ACTIVATION_MODE,
            )
            _STATE.first_call_logged = True
    return result


def backend_status() -> dict[str, Any]:
    """Return a serializable snapshot for diagnostics and the status node."""

    with _STATE_LOCK:
        state = asdict(_STATE)
    state.update(
        {
            "backend": BACKEND_NAME,
            "activation_mode": ACTIVATION_MODE,
            "torch_version": torch.__version__,
            "mps_runtime": _mps_runtime_available(),
            "native_operator": _has_native_mps_kernel(),
            "priority": list(getattr(registry, "_priority", [])),
        }
    )
    return state


def backend_status_text() -> str:
    info = backend_status()
    ordered_keys = (
        "backend",
        "registered",
        "available",
        "reason",
        "activation_mode",
        "torch_version",
        "mps_runtime",
        "native_operator",
        "calls",
        "priority",
    )
    return "\n".join(f"{key}: {info[key]}" for key in ordered_keys)
