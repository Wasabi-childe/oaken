
import torch
from typing import Optional


class OakenQuantizer:
    # NVFP4 parameters
    BLOCK_SIZE = 16
    FP4_MAX = 6.0
    E4M3_MAX = 448.0

    # Positive E2M1 magnitudes
    E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    @staticmethod
    def nvfp4_quantize_dequantize(
        input_tensor: torch.Tensor,
        return_metadata: bool = False
    ):
        """
        NVFP4-style quantize/dequantize simulation.

        - E2M1 values: 4 bits per value
        - E4M3 block scale: one per 16 consecutive values
        - FP32 global scale: one per tensor
        - Blocks are formed along the last dimension.
        - Returns reconstructed values in the input dtype.

        This simulates numerical quantization. It does not pack
        the E2M1 values into a physical 4-bit memory representation.
        """

        if not input_tensor.is_floating_point():
            raise TypeError("input_tensor must have a floating-point dtype")

        original_shape = input_tensor.shape
        original_dtype = input_tensor.dtype
        device = input_tensor.device

        if input_tensor.numel() == 0:
            raise ValueError("input_tensor must not be empty")

        if not torch.isfinite(input_tensor).all():
            raise ValueError("input_tensor must contain only finite values")

        # Use FP32 for scale calculation and quantization.
        x = input_tensor.to(torch.float32)

        # The last dimension is divided into blocks of 16.
        last_dim = x.shape[-1]
        pad_size = (-last_dim) % OakenQuantizer.BLOCK_SIZE

        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size))

        padded_last_dim = x.shape[-1]
        n_blocks = padded_last_dim // OakenQuantizer.BLOCK_SIZE

        # Reshape into blocks along the last dimension.
        blocks = x.reshape(
            *x.shape[:-1],
            n_blocks,
            OakenQuantizer.BLOCK_SIZE
        )

        # -------------------------------------------------
        # Level 1: one FP32 global scale for the tensor
        # -------------------------------------------------
        global_amax = x.abs().amax()

        if global_amax.item() == 0:
            global_scale = torch.ones(
                (), device=device, dtype=torch.float32
            )

            block_scale = torch.zeros_like(
                blocks[..., :1]
            )

            reconstructed = torch.zeros_like(blocks)

        else:
            global_scale = global_amax / (
                OakenQuantizer.E4M3_MAX *
                OakenQuantizer.FP4_MAX
            )

            # ---------------------------------------------
            # Level 2: one E4M3 scale per 16-value block
            # ---------------------------------------------
            block_amax = blocks.abs().amax(
                dim=-1, keepdim=True
            )

            block_scale_fp32 = (
                block_amax /
                OakenQuantizer.FP4_MAX /
                global_scale
            )

            # Store/simulate the local scale in E4M3.
            block_scale = block_scale_fp32.to(
                torch.float8_e4m3fn
            ).to(torch.float32)

            # Reconstruct the effective scale.
            effective_scale = block_scale * global_scale

            # Avoid division by zero for zero-valued blocks.
            safe_scale = torch.where(
                effective_scale > 0,
                effective_scale,
                torch.ones_like(effective_scale)
            )

            # Normalize the values for E2M1 quantization.
            normalized = blocks / safe_scale

            # E2M1 quantizes magnitudes to:
            # 0, 0.5, 1, 1.5, 2, 3, 4, 6.
            levels = torch.tensor(
                OakenQuantizer.E2M1_LEVELS,
                device=device,
                dtype=torch.float32
            )

            distances = (
                normalized.abs().unsqueeze(-1) -
                levels
            ).abs()

            # Select the nearest representable magnitude.
            indices = distances.argmin(dim=-1)

            quantized_magnitude = levels[indices]

            # Restore sign and apply both scaling factors.
            reconstructed = (
                normalized.sign()
                * quantized_magnitude
                * effective_scale
            )

        # Restore the original tensor shape.
        reconstructed = reconstructed.reshape(
            *x.shape[:-1], padded_last_dim
        )

        if pad_size > 0:
            reconstructed = reconstructed[..., :last_dim]

        reconstructed = reconstructed.reshape(original_shape)
        reconstructed = reconstructed.to(original_dtype)

        if return_metadata:
            metadata = {
                "global_scale": global_scale,
                "block_scale": block_scale,
                "block_size": OakenQuantizer.BLOCK_SIZE,
                "format": "NVFP4-style E2M1 + E4M3",
            }
            return reconstructed, metadata

        return reconstructed


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: Optional[list[float]] = None,
        threshold_uppers: Optional[list[float]] = None,
        quantize_outlier: bool = False,
        use_group_shift: bool = True
    ) -> tuple[torch.Tensor, list[float], None]:
        """
        Compatibility wrapper for the previous Oaken interface.

        Thresholds, quantize_outlier and use_group_shift are retained
        so existing calls do not fail, but are intentionally ignored:
        NVFP4 quantizes the entire tensor without Oaken threshold groups.

        Returns:
            reconstructed tensor,
            [1.0] as a compatibility placeholder,
            None as the heat map.
        """

        reconstructed = cls.nvfp4_quantize_dequantize(input_tensor)

        # NVFP4 does not use Oaken's outer/middle/inner groups.
        val_frac = [1.0]
        heat_map = None

        return reconstructed, val_frac, heat_map
