
import torch
from typing import Tuple


class OakenQuantizer:

    QUANTIZE_BITS = 4
    FLOAT_TOLERANCE = 1e-6

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

        tensor = tensor.float()

        # One scale per token
        max_abs = tensor.abs().amax(
            dim=-1,
            keepdim=True
        )

        # FP8 E4M3 scale
        scale = (max_abs / 6.0).to(
            torch.float8_e4m3fn
        ).float()

        scale = torch.where(
            scale == 0,
            torch.full_like(scale, 2.0 ** -9),
            scale
        )

        # Normalize
        normalized = tensor / scale

        levels = torch.tensor(
            cls.FP4_LEVELS,
            dtype=normalized.dtype,
            device=normalized.device
        )

        boundaries = (levels[:-1] + levels[1:]) / 2.0

        indices = torch.bucketize(
            normalized.abs().contiguous(),
            boundaries
        )

        quantized_magnitude = levels[indices]

        quantized = (
            normalized.sign() * quantized_magnitude
        )

        reconstructed = quantized * scale

        return reconstructed


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers,
        threshold_uppers,
        quantize_outlier: bool = True,
        use_group_shift: bool = False,
        mean_abs_outlier=None
    ):

        """
        Offline-calibrated absolute outlier mean + FP4 quantization.

        1. Detect outliers using calibrated thresholds.
        2. Load the pre-calibrated absolute outlier mean.
        3. Round mean to FP8 E4M3.
        4. Subtract signed mean at outlier positions.
        5. Quantize the modified vector using FP4 E2M1.
        6. Add the signed mean back at outlier positions.

        The mean is shared across tokens for each layer and
        is supplied by the offline calibration JSON.
        """

        if len(threshold_lowers) != 1 or len(threshold_uppers) != 1:
            raise ValueError(
                "This algorithm requires exactly one lower and "
                "one upper threshold."
            )

        if mean_abs_outlier is None:
            raise ValueError(
                "A calibrated mean_abs_outlier must be provided."
            )

        threshold_lower = threshold_lowers[0]
        threshold_upper = threshold_uppers[0]

        x = input_tensor.float()

        # -----------------------------------------
        # 1. Detect outliers
        # -----------------------------------------

        _, _, outlier_mask = cls.get_outlier_threshold(
            x,
            threshold_lower,
            threshold_upper
        )

        # -----------------------------------------
        # 2. Load offline-calibrated mean
        # -----------------------------------------

        if isinstance(mean_abs_outlier, (list, tuple)):
            if len(mean_abs_outlier) != 1:
                raise ValueError(
                    "Expected exactly one calibrated mean."
                )

            mean_abs_outlier = mean_abs_outlier[0]

        mean_abs = torch.as_tensor(
            mean_abs_outlier,
            dtype=torch.float32,
            device=x.device
        )

        # Store/use mean in FP8 E4M3
        mean_abs = mean_abs.to(
            torch.float8_e4m3fn
        ).float()

        # -----------------------------------------
        # 3. Subtract signed mean at outlier positions
        # -----------------------------------------

        signed_mean = x.sign() * mean_abs

        offset = torch.where(
            outlier_mask,
            signed_mean,
            torch.zeros_like(x)
        )

        modified_tensor = x - offset

        # -----------------------------------------
        # 4. Quantize modified vector using FP4
        # -----------------------------------------

        reconstructed = cls.fp4_quantize(
            modified_tensor
        )

        # -----------------------------------------
        # 5. Add mean back at original outlier positions
        # -----------------------------------------

        reconstructed = reconstructed + offset

        reconstructed = reconstructed.to(torch.float16)

        # -----------------------------------------
        # 6. Calculate outlier fraction
        # -----------------------------------------

        val_frac = [
            outlier_mask.float().mean().item()
        ]

        heat_map = None

        return reconstructed, val_frac, heat_map
