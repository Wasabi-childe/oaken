
import torch
import torch.nn.functional as F
from typing import Optional


# ================================================================
# FIXED FP4 E2M1
# ================================================================

def _e2m1_round_magnitude(m):
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=m.device,
        dtype=m.dtype,
    )

    idx = (
        (m > 0.25).long()
        + (m >= 0.75).long()
        + (m > 1.25).long()
        + (m >= 1.75).long()
        + (m > 2.5).long()
        + (m >= 3.5).long()
        + (m > 5.0).long()
    )

    return levels[idx.clamp(max=len(levels) - 1)]


class FP4Quantizer:
    @classmethod
    def quantize_dequantize(cls, tensor: torch.Tensor):
        if tensor.numel() == 0:
            return tensor.clone()

        x = tensor.float()
        magnitude = _e2m1_round_magnitude(
            x.abs().clamp(max=6.0)
        )

        return torch.copysign(
            magnitude, x
        ).reshape(tensor.shape).to(tensor.dtype)


class FP4E4M3Quantizer:
    @staticmethod
    def quantize_dequantize(
        tensor: torch.Tensor,
        scale_mode: str = "token",
        group_size: int = 32,
    ):
        if scale_mode not in ("token", "group"):
            raise ValueError(
                "scale_mode must be 'token' or 'group'"
            )

        if tensor.numel() == 0:
            return tensor.clone()

        if scale_mode == "group" and group_size <= 0:
            raise ValueError("group_size must be positive")

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        x = tensor.float().reshape(-1, last_dim)

        if scale_mode == "token":
            amax = x.abs().amax(dim=-1, keepdim=True)

            scale = torch.where(
                amax > 0,
                amax / 6.0,
                torch.ones_like(amax),
            )

            scale = scale.to(torch.float8_e4m3fn).float()
            scale = scale.clamp_min(1e-8)

            normalized = (x / scale).clamp(-6.0, 6.0)

            magnitude = _e2m1_round_magnitude(
                normalized.abs()
            )

            reconstructed = normalized.sign() * magnitude * scale

        else:
            padding = (
                group_size - last_dim % group_size
            ) % group_size

            if padding:
                x = F.pad(x, (0, padding))

            padded_dim = x.shape[-1]

            grouped = x.reshape(
                x.shape[0],
                padded_dim // group_size,
                group_size,
            )

            amax = grouped.abs().amax(
                dim=-1, keepdim=True
            )

            scale = torch.where(
                amax > 0,
                amax / 6.0,
                torch.ones_like(amax),
            )

            scale = scale.to(torch.float8_e4m3fn).float()
            scale = scale.clamp_min(1e-8)

            normalized = (grouped / scale).clamp(-6.0, 6.0)

            magnitude = _e2m1_round_magnitude(
                normalized.abs()
            )

            reconstructed = (
                normalized.sign() * magnitude * scale
            ).reshape(x.shape[0], padded_dim)

            reconstructed = reconstructed[:, :last_dim]

        return reconstructed.reshape(
            original_shape
        ).to(original_dtype)


# ================================================================
# FIXED FP5-LIKE FORMAT
#
# This is a software-defined format:
# sign + 16 magnitude levels + token-wise E4M3 scale.
# It is not a standard IEEE FP5 format.
# ================================================================

class FP5E4M3Quantizer:
    @staticmethod
    def quantize_dequantize(tensor: torch.Tensor):
        if tensor.numel() == 0:
            return tensor.clone()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        x = tensor.float().reshape(-1, last_dim)

        amax = x.abs().amax(dim=-1, keepdim=True)

        scale = torch.where(
            amax > 0,
            amax / 7.0,
            torch.ones_like(amax),
        )

        scale = scale.to(torch.float8_e4m3fn).float()
        scale = scale.clamp_min(1e-8)

        normalized = (x / scale).clamp(-7.0, 7.0)

        levels = torch.tensor(
            [
                0.0, 0.25, 0.5, 0.75,
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 7.0,
            ],
            device=x.device,
            dtype=x.dtype,
        )

        edges = (levels[:-1] + levels[1:]) / 2.0

        indices = torch.bucketize(
            normalized.abs().contiguous(),
            edges,
        )

        magnitude = levels[indices]

        reconstructed = normalized.sign() * magnitude * scale

        return reconstructed.reshape(
            original_shape
        ).to(original_dtype)


# ================================================================
# LEARNED CODEBOOK QUANTIZER
#
# levels:
#   FP5: 16 learned magnitude levels, expected range [0, 7].
#   FP4:  8 learned magnitude levels, expected range [0, 6].
#
# One E4M3 scale per token and group.
# ================================================================

class LearnedCodebookQuantizer:
    @staticmethod
    def quantize_dequantize(
        tensor: torch.Tensor,
        levels,
        max_level: float,
        mask: Optional[torch.Tensor] = None,
    ):
        if tensor.numel() == 0:
            return tensor.clone()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        x = tensor.float().reshape(-1, last_dim)

        if mask is not None:
            selected_mask = mask.reshape(
                -1, last_dim
            ).bool()

            x = x * selected_mask

        else:
            selected_mask = None

        # Token-wise scale computed from the selected group only.
        amax = x.abs().amax(dim=-1, keepdim=True)

        scale = torch.where(
            amax > 0,
            amax / max_level,
            torch.ones_like(amax),
        )

        scale = scale.to(torch.float8_e4m3fn).float()
        scale = scale.clamp_min(1e-8)

        normalized = (x / scale).abs().clamp(
            0.0, max_level
        )

        levels = torch.as_tensor(
            levels,
            device=x.device,
            dtype=x.dtype,
        ).flatten()

        if levels.numel() not in (8, 16):
            raise ValueError(
                "Expected 8 FP4 or 16 FP5 magnitude levels."
            )

        if not torch.all(levels[1:] >= levels[:-1]):
            raise ValueError(
                "Codebook levels must be sorted."
            )

        edges = (levels[:-1] + levels[1:]) / 2.0

        indices = torch.bucketize(
            normalized.contiguous(),
            edges,
        )

        magnitude = levels[indices]

        reconstructed = (
            x.sign() * magnitude * scale
        )

        if selected_mask is not None:
            reconstructed = reconstructed * selected_mask

        return reconstructed.reshape(
            original_shape
        ).to(original_dtype)


# ================================================================
# OAKEN THRESHOLD QUANTIZER
# ================================================================

class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float,
    ):
        outlier_mask = (
            (input_tensor <= threshold_lower)
            | (input_tensor >= threshold_upper)
        )

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list,
        threshold_uppers: list,
    ):
        if len(threshold_lowers) != len(threshold_uppers):
            raise ValueError(
                "Threshold lower/upper lists must have equal lengths"
            )

        if len(threshold_lowers) < 2:
            raise ValueError(
                "At least two threshold groups are required"
            )

        group_masks = []
        group_tensors = []

        prev_thr_low = None
        prev_thr_up = None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):
            if idx == len(threshold_lowers) - 1:
                mask = (
                    (input_tensor > prev_thr_low)
                    & (input_tensor < prev_thr_up)
                )

            elif prev_thr_low is not None:
                mask = (
                    (
                        (input_tensor > prev_thr_low)
                        & (input_tensor <= thr_low)
                    )
                    |
                    (
                        (input_tensor >= thr_up)
                        & (input_tensor < prev_thr_up)
                    )
                )

            else:
                mask = (
                    (input_tensor <= thr_low)
                    | (input_tensor >= thr_up)
                )

            group_masks.append(mask)
            group_tensors.append(input_tensor * mask)

            prev_thr_low = thr_low
            prev_thr_up = thr_up

        return group_tensors, group_masks

    @staticmethod
    def uniform_quantization_threshold(
        tensor,
        bits: int,
        minval,
        maxval,
    ):
        rangeval = maxval - minval

        if torch.is_tensor(rangeval):
            safe_range = torch.where(
                rangeval == 0,
                torch.ones_like(rangeval),
                rangeval,
            )
        else:
            safe_range = rangeval if rangeval != 0 else 1.0

        qx = (2 ** bits - 1) / safe_range
        offset = minval * qx

        quantized = torch.round(qx * tensor - offset)

        quantized = torch.nan_to_num(
            quantized,
            nan=2 ** bits - 1,
        )

        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()

        return OakenQuantizer.uniform_quantization_threshold(
            tensor, bits, minval, maxval
        )

    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)


# ================================================================
# MULTI-GROUP TOKEN-WISE QUANTIZER
#
# With three groups:
#   outer  -> FP5
#   middle -> FP4
#   inner  -> FP5
#
# Supply codebooks=None for fixed codebooks.
# Supply the learned codebooks dictionary to use learned levels.
# ================================================================

class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list,
        threshold_uppers: list,
        quantize_outlier: bool = True,
        use_group_shift: bool = True,
        codebooks: Optional[dict] = None,
        tensor_kind: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        if len(threshold_lowers) != 3:
            raise ValueError(
                "This six-codebook configuration expects "
                "three groups: outer, middle, inner."
            )

        if len(threshold_uppers) != 3:
            raise ValueError(
                "Expected three upper thresholds."
            )

        if codebooks is not None:
            if tensor_kind not in ("key", "value"):
                raise ValueError(
                    "tensor_kind must be 'key' or 'value' "
                    "when learned codebooks are supplied."
                )

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers,
        )

        result_tensor = torch.zeros_like(
            input_tensor
        ).half()

        # Preserve the original non-outlier-quantization mode.
        if not quantize_outlier and codebooks is None:
            grouped_tensors[-2] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-2],
                    scale_mode="token",
                )
            )

            for group, mask in zip(grouped_tensors, masks):
                result_tensor += (
                    group * mask
                ).to(result_tensor.dtype)

            val_frac = [
                (
                    torch.count_nonzero(mask) / mask.numel()
                ).item()
                for mask in masks
            ]

            return result_tensor, val_frac, None

        for idx, (group, mask) in enumerate(
            zip(grouped_tensors, masks)
        ):
            if idx == 0:
                group_name = "outer"
            elif idx == 1:
                group_name = "middle"
            else:
                group_name = "inner"

            # Middle uses FP4; outer and inner use FP5.
            if group_name == "middle":
                bits = 4
                max_level = 6.0
            else:
                bits = 5
                max_level = 7.0

            positive_mask = mask & (input_tensor > 0)
            negative_mask = mask & (input_tensor < 0)

            shifted = input_tensor * mask

            # Match the supplied Oaken group-shifting behavior:
            # outer and middle shift; inner does not.
            if use_group_shift and group_name != "inner":
                low = torch.as_tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )

                high = torch.as_tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )

                shifted = torch.where(
                    positive_mask,
                    input_tensor - high,
                    shifted,
                )

                shifted = torch.where(
                    negative_mask,
                    input_tensor - low,
                    shifted,
                )

                shifted = shifted * mask

            if codebooks is not None:
                class_name = (
                    ("K_" if tensor_kind == "key" else "V_")
                    + group_name
                )

                if class_name not in codebooks:
                    raise KeyError(
                        "Missing codebook: " + class_name
                    )

                # Load Q3.2 integer entries and reconstruct their float values.
                levels_int = codebooks[class_name]
                
                levels = [
                    value / 4.0
                    for value in levels_int
                ]
                
                quantized = (
                    LearnedCodebookQuantizer.quantize_dequantize(
                        shifted,
                        levels=levels,
                        max_level=max_level,
                        mask=mask,
                    )
                )

            elif bits == 4:
                quantized = FP4E4M3Quantizer.quantize_dequantize(
                    shifted,
                    scale_mode="token",
                )

            else:
                quantized = FP5E4M3Quantizer.quantize_dequantize(
                    shifted,
                )

            # Restore group offsets after quantization.
            if use_group_shift and group_name != "inner":
                quantized = torch.where(
                    positive_mask,
                    quantized + high,
                    quantized,
                )

                quantized = torch.where(
                    negative_mask,
                    quantized + low,
                    quantized,
                )

            quantized = quantized * mask

            result_tensor += quantized.to(
                result_tensor.dtype
            )

        val_frac = [
            (
                torch.count_nonzero(mask) / mask.numel()
            ).item()
            for mask in masks
        ]

        return result_tensor, val_frac, None
