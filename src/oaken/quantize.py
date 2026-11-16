
import torch
from typing import Tuple


class OakenQuantizer:

    QUANTIZE_BITS = 4
    FLOAT_TOLERANCE = 1e-6

    # FP4 E2M1 positive magnitude levels
    FP4_LEVELS = [
        0.0, 0.5, 1.0, 1.5,
        2.0, 3.0, 4.0, 6.0
    ]

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        outlier_mask = torch.logical_or(
            input_tensor <= threshold_lower,
            input_tensor >= threshold_upper
        )

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * (~outlier_mask)

        return inlier, outlier, outlier_mask

    @classmethod
    def fp4_quantize(
        cls,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        FP4 E2M1 fake quantization.

        One scale per token, shared across the entire vector.
        Scale is rounded to FP8 E4M3.
        """

        tensor = tensor.float()

        # Calculate one scale per token
        max_abs = tensor.abs().amax(
            dim=-1,
            keepdim=True
        )

        # Calculate and store scale in FP8 E4M3
        scale = (max_abs / 6.0).to(
            torch.float8_e4m3fn
        ).float()

        # Avoid division by zero
        scale = torch.where(
            scale == 0,
            torch.full_like(scale, 2.0 ** -9),
            scale
        )

        # Normalize
        normalized = tensor / scale

        # FP4 E2M1 levels
        levels = torch.tensor(
            cls.FP4_LEVELS,
            dtype=normalized.dtype,
            device=normalized.device
        )

        # Midpoints between adjacent levels
        boundaries = (levels[:-1] + levels[1:]) / 2.0

        # Find nearest FP4 magnitude
        indices = torch.bucketize(
            normalized.abs().contiguous(),
            boundaries
        )

        quantized_magnitude = levels[indices]

        # Restore sign
        quantized = (
            normalized.sign() * quantized_magnitude
        )

        # Dequantize
        reconstructed = quantized * scale

        return reconstructed


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers=None,
        threshold_uppers=None,
        quantize_outlier: bool = True,
        use_group_shift: bool = False,
        use_mean_compensation: bool = False
    ):
        """
        K: threshold-based outlier detection, one absolute outlier
           mean per token, signed mean compensation, then FP4.

        V: direct whole-vector FP4 quantization without mean
           compensation.

        Both K and V use one FP8 E4M3 scale per token.
        """

        x = input_tensor.float()

        # ------------------------------------------------
        # V PATH: Direct FP4 quantization
        # ------------------------------------------------

        if not use_mean_compensation:

            reconstructed = cls.fp4_quantize(x)
            reconstructed = reconstructed.to(torch.float16)

            val_frac = [0.0]
            heat_map = None

            return reconstructed, val_frac, heat_map

        # ------------------------------------------------
        # K PATH: Mean-based outlier compensation
        # ------------------------------------------------

        if threshold_lowers is None or threshold_uppers is None:
            raise ValueError(
                "K quantization requires lower and upper thresholds."
            )

        if len(threshold_lowers) != 1 or len(threshold_uppers) != 1:
            raise ValueError(
                "This algorithm requires exactly one lower and "
                "one upper threshold."
            )

        threshold_lower = threshold_lowers[0]
        threshold_upper = threshold_uppers[0]

        # 1. Detect outliers
        _, _, outlier_mask = cls.get_outlier_threshold(
            x,
            threshold_lower,
            threshold_upper
        )

        # 2. Calculate one absolute outlier mean per token
        outlier_count = outlier_mask.sum(
            dim=-1,
            keepdim=True
        ).clamp(min=1)

        mean_abs = (
            (x.abs() * outlier_mask).sum(
                dim=-1,
                keepdim=True
            ) / outlier_count
        )

        # 3. Store mean in FP8 E4M3
        mean_abs = mean_abs.to(
            torch.float8_e4m3fn
        ).float()

        # 4. Subtract signed mean from outlier positions
        signed_mean = x.sign() * mean_abs

        offset = torch.where(
            outlier_mask,
            signed_mean,
            torch.zeros_like(x)
        )

        modified_tensor = x - offset

        # 5. Quantize the entire modified vector using FP4
        reconstructed = cls.fp4_quantize(
            modified_tensor
        )

        # 6. Add signed mean back at original outlier positions
        reconstructed = reconstructed + offset

        reconstructed = reconstructed.to(torch.float16)

        # 7. Calculate outlier fraction
        val_frac = [
            outlier_mask.float().mean().item()
        ]

        heat_map = None

        return reconstructed, val_frac, heat_map
