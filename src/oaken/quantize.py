
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
        quantize_outlier: bool = False,
        use_group_shift: bool = True
    ) -> tuple[torch.Tensor, list[float], torch.Tensor]:

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(input_tensor).to(
            input_tensor.device
        ).half()

        if quantize_outlier:

            # Quantize the inner-most group using FP4 + E4M3.
            grouped_tensors[-1] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-1], scale_mode="group", group_size=32
                )
            )

            # Quantize the outer groups using FP4 + E4M3.
            for idx in range(len(threshold_lowers) - 1):

                threshold_lower_tensor = torch.tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device
                ).half()

                threshold_upper_tensor = torch.tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device
                ).half()

                higher_mask = grouped_tensors[idx] > 0
                lower_mask = grouped_tensors[idx] < 0

                higher_outlier = (
                    grouped_tensors[idx] * higher_mask
                )
                lower_outlier = (
                    grouped_tensors[idx] * lower_mask
                )

                # Preserve the original group-shifting behavior.
                if use_group_shift:
                    higher_outlier -= threshold_upper_tensor
                    lower_outlier -= threshold_lower_tensor

                # FP4 + E4M3 replaces INT5 for all outlier groups.
                total_outlier = (
                    FP4E4M3Quantizer.quantize_dequantize(
                        higher_outlier * higher_mask
                        + lower_outlier * lower_mask, scale_mode="group", group_size=32
                    )
                )

                higher_outlier = total_outlier * higher_mask
                lower_outlier = total_outlier * lower_mask

                # Restore the original threshold offsets.
                if use_group_shift:
                    higher_outlier += threshold_upper_tensor
                    lower_outlier += threshold_lower_tensor

                grouped_tensors[idx] = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

        else:

            # Quantize the middle group using FP4 + E4M3.
            grouped_tensors[-2] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-2], scale_mode="group", group_size=32
                )
            )

        # Reconstruct the result using the original group masks.
        for tensor, mask in zip(grouped_tensors, masks):
            result_tensor += tensor * mask

        heat_map = None

        val_frac = [
            (
                torch.count_nonzero(mask) / torch.numel(mask)
            ).item()
            for mask in masks
        ]

        return result_tensor, val_frac, heat_map
