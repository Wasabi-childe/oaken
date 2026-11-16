
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
        """

        tensor = tensor.float()

        # Calculate one scale per token
        max_abs = tensor.abs().amax(
            dim=-1,
            keepdim=True
        )

        scale = max_abs / 6.0

        # Avoid division by zero
        scale = torch.where(
            scale < cls.FLOAT_TOLERANCE,
            torch.ones_like(scale),
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

        return reconstructed.to(torch.float16)


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers,
        threshold_uppers,
        quantize_outlier: bool = True,
        use_group_shift: bool = False
    ):
        """
        Sign-aware outlier mean + whole-vector FP4 quantization.

        1. Detect outliers using thresholds.
        2. Separate positive and negative outliers.
        3. Calculate separate means for positive and negative outliers
           for each token.
        4. Subtract the corresponding mean from outliers.
        5. Quantize the entire modified vector using FP4 E2M1.
        6. Add the corresponding mean back during reconstruction.

        The same function is used for K and V.
        """

        if len(threshold_lowers) != 1 or len(threshold_uppers) != 1:
            raise ValueError(
                "This algorithm requires exactly one lower and "
                "one upper threshold."
            )

        threshold_lower = threshold_lowers[0]
        threshold_upper = threshold_uppers[0]

        # ------------------------------------------------
        # 1. Detect outliers
        # ------------------------------------------------

        inlier, outlier, outlier_mask = cls.get_outlier_threshold(
            input_tensor,
            threshold_lower,
            threshold_upper
        )

        # ------------------------------------------------
        # 2. Separate positive and negative outliers
        # ------------------------------------------------

        positive_mask = (
            outlier_mask & (input_tensor > 0)
        )

        negative_mask = (
            outlier_mask & (input_tensor < 0)
        )

        # ------------------------------------------------
        # 3. Calculate separate means per token
        # ------------------------------------------------

        positive_count = positive_mask.sum(
            dim=-1,
            keepdim=True
        ).clamp(min=1)

        negative_count = negative_mask.sum(
            dim=-1,
            keepdim=True
        ).clamp(min=1)

        positive_mean = (
            (input_tensor * positive_mask).sum(
                dim=-1,
                keepdim=True
            ) / positive_count
        )

        negative_mean = (
            (input_tensor * negative_mask).sum(
                dim=-1,
                keepdim=True
            ) / negative_count
        )

        # Handle tokens with no positive outliers
        positive_mean = torch.where(
            positive_mask.any(dim=-1, keepdim=True),
            positive_mean,
            torch.zeros_like(positive_mean)
        )

        # Handle tokens with no negative outliers
        negative_mean = torch.where(
            negative_mask.any(dim=-1, keepdim=True),
            negative_mean,
            torch.zeros_like(negative_mean)
        )

        # ------------------------------------------------
        # 4. Subtract corresponding mean from outliers
        # ------------------------------------------------

        modified_tensor = torch.where(
            positive_mask,
            input_tensor - positive_mean,
            torch.where(
                negative_mask,
                input_tensor - negative_mean,
                input_tensor
            )
        )

        # ------------------------------------------------
        # 5. Quantize the entire modified vector to FP4
        # ------------------------------------------------

        reconstructed = cls.fp4_quantize(
            modified_tensor
        )

        # ------------------------------------------------
        # 6. Add means back to original outlier positions
        # ------------------------------------------------

        reconstructed = torch.where(
            positive_mask,
            reconstructed + positive_mean,
            torch.where(
                negative_mask,
                reconstructed + negative_mean,
                reconstructed
            )
        )

        reconstructed = reconstructed.to(torch.float16)

        # ------------------------------------------------
        # 7. Calculate outlier fraction
        # ------------------------------------------------

        val_frac = [
            outlier_mask.float().mean().item()
        ]

        heat_map = None

        return reconstructed, val_frac, heat_map
