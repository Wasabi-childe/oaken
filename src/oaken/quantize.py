
import torch
from typing import Tuple


class OakenQuantizer:

    QUANTIZE_BITS = 4
    FLOAT_TOLERANCE = 1e-6

    # Fixed hardware processing unit
    CHUNK_SIZE = 128

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
    def int4_mean_quantize(cls, mean: torch.Tensor) -> torch.Tensor:

        mean = mean.float()

        quantized = torch.round(mean)
        quantized = torch.clamp(quantized, 0, 7)

        return quantized

    @staticmethod
    def calculate_storage(
        num_chunks,
        chunk_size,
        k_outlier_counts
    ):

        # FP4 K and V payload
        payload_bits = num_chunks * chunk_size * 2 * 4

        # FP8 K and V scales: one per chunk
        scale_bits = num_chunks * 2 * 8

        # One 4-bit mean per K chunk
        mean_bits = num_chunks * 4

        # Outlier index (7 bits) + sign (1 bit)
        outlier_bits = sum(k_outlier_counts) * 8

        # One 8-bit count per K chunk
        count_bits = num_chunks * 8

        total_bits = (
            payload_bits
            + scale_bits
            + mean_bits
            + outlier_bits
            + count_bits
        )

        total_bytes = (total_bits + 7) // 8

        baseline_bits = num_chunks * chunk_size * 2 * 16

        reduction = (
            1 - total_bits / baseline_bits
        ) * 100

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

        tensor = tensor.float()

        # One scale per 64-element chunk
        max_abs = tensor.abs().amax(dim=-1, keepdim=True)

        scale = (
            max_abs / 6.0
        ).to(torch.float8_e4m3fn).float()

        scale = torch.where(
            scale == 0,
            torch.full_like(scale, 2.0 ** -9),
            scale
        )

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
        use_mean_compensation: bool = False
    ):

        """
        K:
            Divide each head into 64-element chunks.
            Detect outliers independently per chunk.
            Calculate one outlier mean per chunk.
            Apply FP4 quantization with one FP8 scale per chunk.

        V:
            Direct FP4 quantization with one FP8 scale per chunk.

        Metadata:
            K: 4-bit mean + 8-bit count per chunk.
            K and V: FP8 scale per chunk.
        """

        chunk_size = cls.CHUNK_SIZE

        orig_shape = input_tensor.shape
        x = input_tensor.float()

        # ------------------------------------------------
        # Divide into attention heads and 64-element chunks
        # ------------------------------------------------

        orig_shape = input_tensor.shape
        x = input_tensor.float()

        if orig_shape[-1] % chunk_size != 0:
            raise ValueError(
                "Hidden dimension must be divisible by chunk_size."
            )

        num_chunks = orig_shape[-1] // chunk_size

        x = x.reshape(
            *orig_shape[:-1],
            num_chunks,
            chunk_size
)

        # ------------------------------------------------
        # V PATH: Direct FP4 quantization
        # ------------------------------------------------

        if not use_mean_compensation:

            reconstructed = cls.fp4_quantize(x)

            reconstructed = (
                reconstructed
                .reshape(orig_shape)
                .to(torch.float16)
            )

            return reconstructed, [0.0], None

        # ------------------------------------------------
        # K PATH: Mean-based outlier compensation
        # ------------------------------------------------

        if threshold_lowers is None or threshold_uppers is None:
            raise ValueError(
                "K quantization requires lower and upper thresholds."
            )

        if len(threshold_lowers) != 1 or len(threshold_uppers) != 1:
            raise ValueError(
                "Exactly one lower and upper threshold are required."
            )

        threshold_lower = threshold_lowers[0]
        threshold_upper = threshold_uppers[0]

        # 1. Detect outliers independently in each chunk

        _, _, outlier_mask = cls.get_outlier_threshold(
            x,
            threshold_lower,
            threshold_upper
        )

        # Actual outlier count per chunk

        actual_outlier_counts = (
            outlier_mask.sum(dim=-1).to(torch.int64)
        )


        # 2. Calculate absolute outlier mean per chunk

        outlier_count = (
            outlier_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        )

        mean_abs = (
            (x.abs() * outlier_mask).sum(
                dim=-1,
                keepdim=True
            ) / outlier_count
        )

        # 3. Quantize mean to unsigned 4-bit integer

        mean_abs = cls.int4_mean_quantize(mean_abs)

        # 4. Subtract signed mean at outlier positions

        offset = torch.where(
            outlier_mask,
            x.sign() * mean_abs,
            torch.zeros_like(x)
        )

        # 5. FP4 quantization independently per 64-element chunk

        reconstructed = cls.fp4_quantize(x - offset)

        # 6. Add mean back

        reconstructed = reconstructed + offset

        reconstructed = (
            reconstructed
            .reshape(orig_shape)
            .to(torch.float16)
        )

        # Outlier fraction

        val_frac = [
            outlier_mask.float().mean().item()
        ]

        # ------------------------------------------------
        # Calculate storage per token
        # ------------------------------------------------

        storage_results = []

        flat_counts = actual_outlier_counts.reshape(
            -1,
            num_chunks
        )

        for token_counts in flat_counts:

            storage = cls.calculate_storage(
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                k_outlier_counts=token_counts.tolist()
            )

            storage_results.append(storage)

        return reconstructed, val_frac, storage_results
