
import torch
from typing import Optional


import torch
import torch.nn.functional as F

def _e2m1_round_magnitude(m):
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          device=m.device, dtype=m.dtype)
    idx = ((m > 0.25).long() + (m >= 0.75).long() + (m > 1.25).long()
           + (m >= 1.75).long() + (m > 2.5).long() + (m >= 3.5).long()
           + (m > 5.0).long())
    return levels[idx]

class FP4Quantizer:
    @classmethod
    def quantize_dequantize(cls, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.clone()

        shape = tensor.shape
        dtype = tensor.dtype

        x = tensor.float()

        # No scaling: quantize directly to the E2M1 FP4 codebook.
        mag = _e2m1_round_magnitude(x.abs().clamp(max=6.0))
        out = torch.copysign(mag, x)

        return out.reshape(shape).to(dtype)




class FP4E4M3Quantizer:

    @staticmethod
    def quantize_dequantize(
        tensor,
        scale_mode="token",
        group_size=32
    ):
        """
        FP4 E2M1 quantization with E4M3 scales.

        Parameters
        ----------
        tensor:
            Input tensor. The last dimension is the feature dimension.

        scale_mode:
            "token": one scale for the complete last dimension
                     of each token.

            "group": one scale per group along the last dimension.

        group_size:
            Number of values covered by each scale in group mode.
            Ignored in token mode.

        Returns
        -------
        Dequantized tensor in the original shape and dtype.
        """

        if scale_mode not in ("token", "group"):
            raise ValueError(
                "scale_mode must be 'token' or 'group'"
            )

        if tensor.numel() == 0:
            return tensor.clone()

        if scale_mode == "group" and group_size <= 0:
            raise ValueError(
                "group_size must be greater than zero"
            )

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        # Work in FP32 for the quantization calculations.
        x = tensor.float()

        # Flatten all leading dimensions into rows.
        # For [batch, sequence, hidden], each row represents a token.
        x = x.reshape(-1, last_dim)

        if scale_mode == "token":

            # One scale for the entire feature vector of each token.
            amax = x.abs().amax(
                dim=-1,
                keepdim=True
            )

            scale = amax / 6.0

            # Handle all-zero vectors safely.
            scale = torch.where(
                amax > 0,
                scale,
                torch.ones_like(scale)
            )

            # Store scales in E4M3, then use their represented values.
            scale = scale.to(
                torch.float8_e4m3fn
            ).float()

            scale = scale.clamp_min(1e-8)

            normalized = x / scale

            normalized = normalized.clamp(-6.0, 6.0)

            magnitude = _e2m1_round_magnitude(
                normalized.abs()
            )

            quantized = normalized.sign() * magnitude

            reconstructed = quantized * scale

        else:

            # Pad the feature dimension to complete the last group.
            padding = (
                group_size - last_dim % group_size
            ) % group_size

            if padding:
                x = torch.nn.functional.pad(
                    x,
                    (0, padding)
                )

            padded_dim = x.shape[-1]
            n_groups = padded_dim // group_size

            # [rows, groups, values_per_group]
            grouped = x.reshape(
                x.shape[0],
                n_groups,
                group_size
            )

            # One scale per group.
            amax = grouped.abs().amax(
                dim=-1,
                keepdim=True
            )

            scale = amax / 6.0

            scale = torch.where(
                amax > 0,
                scale,
                torch.ones_like(scale)
            )

            scale = scale.to(
                torch.float8_e4m3fn
            ).float()

            scale = scale.clamp_min(1e-8)

            normalized = grouped / scale

            normalized = normalized.clamp(-6.0, 6.0)

            magnitude = _e2m1_round_magnitude(
                normalized.abs()
            )

            quantized = normalized.sign() * magnitude

            reconstructed = quantized * scale

            # Restore the padded groups to a flat feature dimension.
            reconstructed = reconstructed.reshape(
                x.shape[0],
                padded_dim
            )

            # Remove padding.
            reconstructed = reconstructed[:, :last_dim]

        return reconstructed.reshape(
            original_shape
        ).to(original_dtype)



class FP5E4M3Quantizer:
    """
    Custom FP5 E2M2 value quantization with E4M3 token-wise scales.

    Each row (the last dimension of the input) gets one scale. The value
    codebook uses sign + E2M2 magnitudes represented by the levels below.
    This is a software-defined E2M2 codebook, not a claim of compatibility
    with a specific vendor's FP5 hardware format.
    """

    @staticmethod
    def quantize_dequantize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.clone()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        # Flatten leading dimensions so each row represents one token vector.
        x = tensor.float().reshape(-1, last_dim)

        # One scale per token vector. The largest codebook magnitude is 7.
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / 7.0
        scale = torch.where(amax > 0, scale, torch.ones_like(scale))

        # Store/use the represented E4M3 scale value.
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)

        normalized = (x / scale).clamp(-7.0, 7.0)

        # Non-negative magnitudes in the selected E2M2 codebook.
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

        # Nearest-level rounding; ties select the lower listed level.
        distances = (normalized.abs().unsqueeze(-1) - levels).abs()
        indices = distances.argmin(dim=-1)
        magnitude = levels[indices]
        quantized = normalized.sign() * magnitude
        reconstructed = quantized * scale

        return reconstructed.reshape(original_shape).to(original_dtype)


class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        outlier_mask = torch.logical_or(
            input_tensor <= threshold_lower,
            threshold_upper <= input_tensor
        )

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        group_masks = list()
        group_tensors = list()
        prev_thr_low, prev_thr_up = None, None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):
            if idx == len(threshold_lowers) - 1:
                # Inner-most Group
                group_masks.append(
                    mask := torch.logical_and(
                        input_tensor > prev_thr_low,
                        input_tensor < prev_thr_up
                    )
                )

            elif (
                prev_thr_low is not None
                and prev_thr_up is not None
            ):
                group_masks.append(
                    mask := torch.logical_or(
                        torch.logical_and(
                            prev_thr_low < input_tensor,
                            input_tensor <= thr_low
                        ),
                        torch.logical_and(
                            thr_up <= input_tensor,
                            input_tensor < prev_thr_up
                        )
                    )
                )

            else:
                # Outer-most Group
                group_masks.append(
                    mask := torch.logical_or(
                        input_tensor <= thr_low,
                        thr_up <= input_tensor
                    )
                )

            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_tensors.append(input_tensor * mask)

        assert (
            len(threshold_lowers)
            == len(threshold_uppers)
            == len(group_tensors)
            == len(group_masks)
        )

        return group_tensors, group_masks

    @staticmethod
    def uniform_quantization_threshold(
        tensor,
        bits: int,
        minval: torch.Tensor,
        maxval: torch.Tensor
    ):
        rangeval = maxval - minval
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx

        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(
            quantized,
            nan=2 ** bits - 1
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



class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

@classmethod
def downsample(
    cls,
    input_tensor: torch.Tensor,
    threshold_lowers: list[float],
    threshold_uppers: list[float],
    quantize_outlier: bool = True,
    use_group_shift: bool = True,
    is_key: bool = True
) -> tuple[torch.Tensor, list[float], torch.Tensor]:

    # V: quantize the complete tensor using FP4.
    # No threshold grouping and no outlier handling.
    if not is_key:
        result_tensor = FP4E4M3Quantizer.quantize_dequantize(
            input_tensor,
            scale_mode="token"
        )

        val_frac = [1.0]
        heat_map = None

        return result_tensor, val_frac, heat_map

    # K: split into threshold groups.
    grouped_tensors, masks = cls.get_multigroup_threshold(
        input_tensor,
        threshold_lowers,
        threshold_uppers
    )

    result_tensor = torch.zeros_like(input_tensor).half()

    # Quantize the middle group using FP4.
    middle_idx = len(grouped_tensors) - 2

    grouped_tensors[middle_idx] = (
        FP4E4M3Quantizer.quantize_dequantize(
            grouped_tensors[middle_idx],
            scale_mode="token"
        )
    )

    # Quantize only the outer outlier groups using FP5.
    if quantize_outlier:
        for idx in range(len(grouped_tensors)):
            if idx == middle_idx:
                continue

            group = grouped_tensors[idx]

            if use_group_shift:
                low = torch.tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                ) if idx < len(threshold_lowers) else None

                high = torch.tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                ) if idx < len(threshold_uppers) else None

                positive_mask = group > 0
                negative_mask = group < 0

                shifted = group.clone()

                if high is not None:
                    shifted = torch.where(
                        positive_mask, group - high, shifted
                    )

                if low is not None:
                    shifted = torch.where(
                        negative_mask, group - low, shifted
                    )

                quantized = FP5E4M3Quantizer.quantize_dequantize(
                    shifted
                )

                if high is not None:
                    quantized = torch.where(
                        positive_mask, quantized + high, quantized
                    )

                if low is not None:
                    quantized = torch.where(
                        negative_mask, quantized + low, quantized
                    )

                grouped_tensors[idx] = quantized
            else:
                grouped_tensors[idx] = (
                    FP5E4M3Quantizer.quantize_dequantize(group)
                )

    # Reconstruct using the original masks.
    for group, mask in zip(grouped_tensors, masks):
        result_tensor += (group * mask).to(result_tensor.dtype)

    val_frac = [
        (torch.count_nonzero(mask) / mask.numel()).item()
        for mask in masks
    ]

    return result_tensor, val_frac, None
