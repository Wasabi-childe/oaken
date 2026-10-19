
import torch
import torch.nn.functional as F


# ================================================================
# HADAMARD TRANSFORM
# Normalized Walsh-Hadamard transform along the last dimension.
# Since H is orthogonal and symmetric, H(H(x)) = x.
# ================================================================

def hadamard_transform(tensor):
    original_shape = tensor.shape
    n = original_shape[-1]

    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(
            "Last dimension must be a positive power of 2."
        )

    x = tensor.float().contiguous()
    h = 1

    while h < n:
        x = x.reshape(*original_shape[:-1], n // (2 * h), 2, h)

        a = x[..., 0, :]
        b = x[..., 1, :]

        x = torch.stack((a + b, a - b), dim=-2)
        x = x.reshape(*original_shape[:-1], n)

        h *= 2

    return x / (n ** 0.5)


# ================================================================
# E2M1 FP4 ROUNDING
# Representable magnitudes:
# 0, 0.5, 1, 1.5, 2, 3, 4, 6
# ================================================================

def _e2m1_round_magnitude(m):
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=m.device,
        dtype=m.dtype
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

    return levels[idx]


# ================================================================
# FP4 WITHOUT SCALING
# ================================================================

class FP4Quantizer:

    @classmethod
    def quantize_dequantize(cls, tensor):
        if tensor.numel() == 0:
            return tensor.clone()

        shape = tensor.shape
        dtype = tensor.dtype

        x = tensor.float()

        mag = _e2m1_round_magnitude(
            x.abs().clamp(max=6.0)
        )

        out = torch.copysign(mag, x)

        return out.reshape(shape).to(dtype)


# ================================================================
# FP4 E2M1 + E4M3 GROUP SCALES
# One scale per group of 32 values along the last dimension.
# This is a simulation: values are reconstructed as floating point.
# ================================================================

class FP4E4M3Quantizer:

    BLOCK_SIZE = 32

    @classmethod
    def quantize_dequantize(cls, tensor, block_size=None):
        if tensor.numel() == 0:
            return tensor.clone()

        if block_size is None:
            block_size = cls.BLOCK_SIZE

        shape = tensor.shape
        dtype = tensor.dtype

        x = tensor.reshape(-1, shape[-1]).float()
        n, w = x.shape

        pad = (-w) % block_size

        if pad:
            x = F.pad(x, (0, pad))

        blocks = x.reshape(n, -1, block_size)

        amax = blocks.abs().amax(
            dim=-1,
            keepdim=True
        )

        # Calculate one scale for each group.
        scale = (amax / 6.0).clamp(
            min=2.0 ** -9,
            max=448.0
        )

        # Store scale in E4M3, then convert back to FP32
        # for simulated quantization/dequantization.
        scale = scale.to(
            torch.float8_e4m3fn
        ).float()

        normalized = blocks / scale

        mag = _e2m1_round_magnitude(
            normalized.abs().clamp(max=6.0)
        )

        quantized = torch.copysign(
            mag, normalized
        )

        # Dequantize.
        out = quantized * scale

        out = out.reshape(n, -1)[:, :w]

        return out.reshape(shape).to(dtype)


# ================================================================
# OAKEN QUANTIZER UTILITIES
# ================================================================

class OakenQuantizer:

    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor,
        threshold_lower,
        threshold_upper
    ):
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
        input_tensor,
        threshold_lowers,
        threshold_uppers
    ):
        group_masks = []
        group_tensors = []

        prev_thr_low = None
        prev_thr_up = None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):
            if idx == len(threshold_lowers) - 1:
                # Inner-most group.
                mask = torch.logical_and(
                    input_tensor > prev_thr_low,
                    input_tensor < prev_thr_up
                )

            elif (
                prev_thr_low is not None
                and prev_thr_up is not None
            ):
                mask = torch.logical_or(
                    torch.logical_and(
                        prev_thr_low < input_tensor,
                        input_tensor <= thr_low
                    ),
                    torch.logical_and(
                        thr_up <= input_tensor,
                        input_tensor < prev_thr_up
                    )
                )

            else:
                # Outer-most group.
                mask = torch.logical_or(
                    input_tensor <= thr_low,
                    thr_up <= input_tensor
                )

            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_masks.append(mask)
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
        bits,
        minval,
        maxval
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
    def uniform_quantization(tensor, bits):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()

        return OakenQuantizer.uniform_quantization_threshold(
            tensor,
            bits,
            minval,
            maxval
        )

    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)

        truncated = (
            int16_tensor & 0b1_11111_1110_0000_00
        )

        return truncated.view(torch.float16)


# ================================================================
# OAKEN MULTI-THRESHOLD TOKENWISE QUANTIZER
#
# apply_hadamard=True:
#     Rotate K before threshold grouping and quantization.
#     Inverse-rotate the reconstructed result before returning it.
#
# apply_hadamard=False:
#     No rotation. Use this for V.
# ================================================================

class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor,
        threshold_lowers,
        threshold_uppers,
        quantize_outlier=False,
        use_group_shift=True,
        apply_hadamard=False
    ):
        if input_tensor.numel() == 0:
            return (
                input_tensor.clone(),
                [],
                None
            )

        original_dtype = input_tensor.dtype

        # --------------------------------------------------------
        # 1. Rotate K only, if requested.
        # --------------------------------------------------------
        if apply_hadamard:
            working_tensor = hadamard_transform(
                input_tensor
            ).to(original_dtype)
        else:
            working_tensor = input_tensor

        # --------------------------------------------------------
        # 2. Apply Oaken threshold grouping in the domain being
        #    quantized. For rotated K, thresholds must be calibrated
        #    on rotated K data.
        # --------------------------------------------------------
        grouped_tensors, masks = cls.get_multigroup_threshold(
            working_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(
            working_tensor
        ).half()

        # --------------------------------------------------------
        # 3. Quantize selected groups using FP4 + E4M3.
        # --------------------------------------------------------
        if quantize_outlier:

            # Quantize the inner-most group.
            grouped_tensors[-1] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-1]
                )
            )

            # Quantize the outer groups.
            for idx in range(len(threshold_lowers) - 1):

                threshold_lower_tensor = torch.tensor(
                    threshold_lowers[idx],
                    device=working_tensor.device
                ).half()

                threshold_upper_tensor = torch.tensor(
                    threshold_uppers[idx],
                    device=working_tensor.device
                ).half()

                higher_mask = grouped_tensors[idx] > 0
                lower_mask = grouped_tensors[idx] < 0

                higher_outlier = (
                    grouped_tensors[idx] * higher_mask
                )

                lower_outlier = (
                    grouped_tensors[idx] * lower_mask
                )

                # Preserve original group-shifting behavior.
                if use_group_shift:
                    higher_outlier -= threshold_upper_tensor
                    lower_outlier -= threshold_lower_tensor

                total_outlier = (
                    FP4E4M3Quantizer.quantize_dequantize(
                        higher_outlier * higher_mask
                        + lower_outlier * lower_mask
                    )
                )

                higher_outlier = total_outlier * higher_mask
                lower_outlier = total_outlier * lower_mask

                # Restore threshold offsets.
                if use_group_shift:
                    higher_outlier += threshold_upper_tensor
                    lower_outlier += threshold_lower_tensor

                grouped_tensors[idx] = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

        else:
            # Quantize the middle group.
            grouped_tensors[-2] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-2]
                )
            )

        # --------------------------------------------------------
        # 4. Reconstruct the threshold groups.
        # --------------------------------------------------------
        for group_tensor, mask in zip(
            grouped_tensors, masks
        ):
            result_tensor += group_tensor * mask

        # --------------------------------------------------------
        # 5. Inverse Hadamard rotation for K.
        #
        # The normalized Hadamard transform is self-inverse.
        # Return the reconstructed tensor in the original K domain.
        # --------------------------------------------------------
        if apply_hadamard:
            result_tensor = hadamard_transform(
                result_tensor.float()
            )

        result_tensor = result_tensor.to(original_dtype)

        # --------------------------------------------------------
        # 6. Group fractions.
        # --------------------------------------------------------
        val_frac = [
            (
                torch.count_nonzero(mask)
                / torch.numel(mask)
            ).item()
            for mask in masks
        ]

        heat_map = None

        return result_tensor, val_frac, heat_map


# ================================================================
# EXAMPLE USAGE
# ================================================================

# Use apply_hadamard=True for K:
#
# K_quantized, K_group_fractions, _ = (
#     MultiThresholdTokenwiseQuantizer.downsample(
#         input_tensor=K,
#         threshold_lowers=K_threshold_lowers,
#         threshold_uppers=K_threshold_uppers,
#         quantize_outlier=True,
#         use_group_shift=True,
#         apply_hadamard=True
#     )
# )
#
# Use apply_hadamard=False for V:
#
# V_quantized, V_group_fractions, _ = (
#     MultiThresholdTokenwiseQuantizer.downsample(
#         input_tensor=V,
#         threshold_lowers=V_threshold_lowers,
#         threshold_uppers=V_threshold_uppers,
#         quantize_outlier=True,
#         use_group_shift=True,
#         apply_hadamard=False
#     )
# )
