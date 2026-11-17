import torch
from typing import Tuple


class OakenQuantizer:

    QUANTIZE_BITS = 4
    FLOAT_TOLERANCE = 1e-6

    # Set once (e.g. OakenQuantizer.HEAD_DIM = 64) if you cannot change the call site.
    HEAD_DIM = 64

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
    def fp4_mean_quantize(cls, mean: torch.Tensor) -> torch.Tensor:
        """Nearest FP4 E2M1 value for a positive mean (no extra scale)."""

        mean = mean.float()

        levels = torch.tensor(
            cls.FP4_LEVELS, dtype=mean.dtype, device=mean.device
        )

        boundaries = (levels[:-1] + levels[1:]) / 2.0

        indices = torch.bucketize(mean.contiguous(), boundaries)

        return levels[indices]

    @staticmethod
    def calculate_storage(
        num_kv_heads,
        head_dim,
        k_outlier_counts
    ):
        # FP4 K and V payload
        payload_bits = num_kv_heads * head_dim * 2 * 4

        # FP8 K and V scales
        scale_bits = num_kv_heads * 2 * 8

        # One FP4 mean per K head
        mean_bits = num_kv_heads * 4

        # Outlier index (7 bits) + sign (1 bit)
        outlier_bits = sum(k_outlier_counts) * 8

        # One 8-bit count per K head
        count_bits = num_kv_heads * 8

        total_bits = (
            payload_bits
            + scale_bits
            + mean_bits
            + outlier_bits
            + count_bits
        )

        # Round up to complete bytes
        total_bytes = (total_bits + 7) // 8

        baseline_bits = num_kv_heads * head_dim * 2 * 16
        reduction = (1 - total_bits / baseline_bits) * 100

        return {
            "payload_bits": payload_bits,
            "scale_bits": scale_bits,
            "mean_bits": mean_bits,
            "outlier_bits": outlier_bits,
            "count_bits": count_bits,
            "total_bits": total_bits,
            "total_bytes": total_bytes,
            "baseline_bits": baseline_bits,
            "reduction_percent": reduction
        }
    
    @classmethod
    def fp4_quantize(cls, tensor: torch.Tensor) -> torch.Tensor:
        """
        FP4 E2M1 fake quantization.

        One FP8 E4M3 scale per group, where a group is the LAST dimension.
        Callers reshape so that the last dimension is one head.
        """

        tensor = tensor.float()

        max_abs = tensor.abs().amax(dim=-1, keepdim=True)

        scale = (max_abs / 6.0).to(torch.float8_e4m3fn).float()

        scale = torch.where(
            scale == 0,
            torch.full_like(scale, 2.0 ** -9),
            scale
        )

        normalized = tensor / scale

        levels = torch.tensor(
            cls.FP4_LEVELS, dtype=normalized.dtype, device=normalized.device
        )

        boundaries = (levels[:-1] + levels[1:]) / 2.0

        indices = torch.bucketize(normalized.abs().contiguous(), boundaries)

        quantized = normalized.sign() * levels[indices]

        return quantized * scale


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers=None,
        threshold_uppers=None,
        quantize_outlier: bool = True,
        use_group_shift: bool = False,
        use_mean_compensation: bool = False,
        head_dim: int = None
    ):
        """
        K (use_mean_compensation=True):
            Threshold outliers, one absolute outlier mean per head per token
            (rounded to FP4), signed mean compensation, FP4 quantization.

        V (use_mean_compensation=False):
            Direct FP4 quantization.

        Scale (FP8 E4M3), outlier mean and outlier count are all per head
        per token.
        """

        head_dim = head_dim if head_dim is not None else cls.HEAD_DIM

        orig_shape = input_tensor.shape
        x = input_tensor.float()

        # ------------------------------------------------
        # Group the last dimension into heads
        # ------------------------------------------------

        if head_dim is not None and orig_shape[-1] != head_dim:
            if orig_shape[-1] % head_dim != 0:
                raise ValueError(
                    f"Last dim {orig_shape[-1]} is not divisible by head_dim {head_dim}."
                )
            x = x.reshape(*orig_shape[:-1], orig_shape[-1] // head_dim, head_dim)

        # ------------------------------------------------
        # V PATH: direct FP4
        # ------------------------------------------------

        if not use_mean_compensation:

            reconstructed = cls.fp4_quantize(x)
            reconstructed = reconstructed.reshape(orig_shape).to(torch.float16)

            return reconstructed, [0.0], None

        # ------------------------------------------------
        # K PATH: mean-based outlier compensation
        # ------------------------------------------------

        if threshold_lowers is None or threshold_uppers is None:
            raise ValueError("K quantization requires lower and upper thresholds.")

        if len(threshold_lowers) != 1 or len(threshold_uppers) != 1:
            raise ValueError(
                "This algorithm requires exactly one lower and one upper threshold."
            )

        threshold_lower = threshold_lowers[0]
        threshold_upper = threshold_uppers[0]

        # 1. Detect outliers
        _, _, outlier_mask = cls.get_outlier_threshold(
            x, threshold_lower, threshold_upper
        )
        # Keep actual count for every head and token
        actual_outlier_counts = (
            outlier_mask.sum(dim=-1).to(torch.int64)
        )

        # 2. One absolute outlier mean per head per token
        outlier_count = outlier_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        mean_abs = (
            (x.abs() * outlier_mask).sum(dim=-1, keepdim=True) / outlier_count
        )

        # 3. Round the mean to FP4 E2M1
        mean_abs = cls.fp4_mean_quantize(mean_abs)

        # 4. Subtract the signed mean at outlier positions
        offset = torch.where(
            outlier_mask,
            x.sign() * mean_abs,
            torch.zeros_like(x)
        )

        # 5. Quantize the whole modified head vector (scale per head)
        reconstructed = cls.fp4_quantize(x - offset)

        # 6. Add the mean back
        reconstructed = reconstructed + offset

        reconstructed = reconstructed.reshape(orig_shape).to(torch.float16)

        val_frac = [outlier_mask.float().mean().item()]

        # Count heads in the grouped tensor
        num_heads = x.shape[-2]

        # Each leading-dimension position represents a token/head group.
        # Calculate storage separately for each token.
        storage_results = []

        flat_counts = actual_outlier_counts.reshape(-1, num_heads)

        for token_counts in flat_counts:
            storage = cls.calculate_storage(
                head_dim=head_dim,
                num_kv_heads=num_heads,
                outlier_counts=token_counts.tolist()
            )
            storage_results.append(storage)

        return reconstructed, val_frac, storage_results