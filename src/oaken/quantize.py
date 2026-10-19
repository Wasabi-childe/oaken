
import torch
import torch.nn.functional as F


# ================================================================
# FP4 E2M1 + E4M3 BLOCK-SCALE QUANTIZER
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


class FP4E2M1E4M3Quantizer:
    BLOCK_SIZE = 32

    @classmethod
    def quantize_dequantize(cls, tensor, block_size=None):
        if tensor.numel() == 0:
            return tensor.clone()

        if block_size is None:
            block_size = cls.BLOCK_SIZE

        shape = tensor.shape
        dtype = tensor.dtype

        # Treat each row independently. Quantization blocks run
        # along the last dimension.
        x = tensor.reshape(-1, shape[-1]).float()
        n, width = x.shape

        pad = (-width) % block_size
        if pad:
            x = F.pad(x, (0, pad))

        blocks = x.reshape(n, -1, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True)

        # E4M3 scale for each block.
        scale = (amax / 6.0).clamp(
            min=2.0 ** -9,
            max=448.0
        )
        scale = scale.to(torch.float8_e4m3fn).float()

        normalized = blocks / scale
        magnitude = _e2m1_round_magnitude(
            normalized.abs().clamp(max=6.0)
        )

        reconstructed = torch.copysign(
            magnitude, normalized
        ) * scale

        reconstructed = reconstructed.reshape(n, -1)[:, :width]

        return reconstructed.reshape(shape).to(dtype)


# ================================================================
# CUSTOM FP2 E1M0-STYLE QUANTIZER
#
# Explicit encoding:
#   00 = +0
#   10 = -0 (reconstructed as zero)
#   01 = +1 * block_scale
#   11 = -1 * block_scale
#
# The exponent bit is 0 for zero and 1 for a nonzero value.
# There is no mantissa bit. This is a custom ternary format,
# not a standard IEEE or industry-standard FP2 format.
#
# Values below half the block maximum are rounded to zero.
# ================================================================

class FP2E1M0StyleQuantizer:
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
        n, width = x.shape

        pad = (-width) % block_size
        if pad:
            x = F.pad(x, (0, pad))

        blocks = x.reshape(n, -1, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True)

        # One shared scale per block.
        scale = amax.clamp(min=2.0 ** -9, max=448.0)
        scale = scale.to(torch.float8_e4m3fn).float()

        normalized = blocks / scale

        # Exponent bit = 0 represents zero.
        # Exponent bit = 1 represents magnitude 1 * scale.
        nonzero = normalized.abs() >= 0.5

        reconstructed = (
            torch.sign(normalized)
            * nonzero.float()
            * scale
        )

        # Preserve exact zeros, including all-zero blocks.
        reconstructed = torch.where(
            blocks == 0,
            torch.zeros_like(reconstructed),
            reconstructed
        )

        reconstructed = reconstructed.reshape(n, -1)[:, :width]

        return reconstructed.reshape(shape).to(dtype)


# ================================================================
# MIXED-PRECISION K CHANNEL QUANTIZER
#
# Input shape: [batch, tokens, channels] or [tokens, channels]
#
# Calibration:
#   1. Compute mean absolute magnitude per channel.
#   2. Select the top 25% channels for FP4.
#   3. Assign all other channels to FP2.
#
# Application:
#   FP4 channels -> FP4 E2M1 + E4M3 block scales
#   FP2 channels -> custom FP2 E1M0-style quantizer
#
# No Oaken outlier thresholds or group shifting.
# ================================================================

class MixedPrecisionKQuantizer:
    FP4_FRACTION = 0.25
    BLOCK_SIZE = 32

    @classmethod
    def compute_channel_mask(cls, calibration_k):
        """
        calibration_k:
            [tokens, channels] or [batch, tokens, channels]

        Returns:
            Boolean mask of shape [channels].
            True  -> FP4
            False -> FP2
        """
        if calibration_k.ndim not in (2, 3):
            raise ValueError(
                "Expected K tensor with shape [tokens, channels] "
                "or [batch, tokens, channels]."
            )

        # Average across all dimensions except the channel axis.
        reduce_dims = tuple(range(calibration_k.ndim - 1))
        scores = calibration_k.float().abs().mean(dim=reduce_dims)

        num_channels = scores.numel()
        num_fp4 = max(
            1,
            min(
                num_channels,
                int(num_channels * cls.FP4_FRACTION + 0.5)
            )
        )

        selected_indices = torch.topk(
            scores, k=num_fp4, largest=True
        ).indices

        channel_mask = torch.zeros(
            num_channels,
            dtype=torch.bool,
            device=calibration_k.device
        )
        channel_mask[selected_indices] = True

        return channel_mask

    @classmethod
    def quantize_dequantize(cls, k_tensor, channel_mask):
        """
        k_tensor:
            K tensor with shape [tokens, channels] or
            [batch, tokens, channels].

        channel_mask:
            Boolean mask calculated using compute_channel_mask().
            True = FP4; False = FP2.

        Returns:
            Reconstructed K tensor, same shape and dtype as input.
        """
        if k_tensor.ndim not in (2, 3):
            raise ValueError(
                "Expected K tensor with shape [tokens, channels] "
                "or [batch, tokens, channels]."
            )

        if k_tensor.shape[-1] != channel_mask.numel():
            raise ValueError(
                "Channel mask length must match the last dimension "
                "of the K tensor."
            )

        mask = channel_mask.to(
            device=k_tensor.device, dtype=torch.bool
        )

        result = torch.empty_like(k_tensor)

        fp4_indices = torch.where(mask)[0]
        fp2_indices = torch.where(~mask)[0]

        # Gather selected channels, quantize, then restore positions.
        if fp4_indices.numel() > 0:
            fp4_values = k_tensor.index_select(
                -1, fp4_indices
            )

            fp4_values = FP4E2M1E4M3Quantizer.quantize_dequantize(
                fp4_values,
                block_size=cls.BLOCK_SIZE
            )

            result.index_copy_(-1, fp4_indices, fp4_values)

        if fp2_indices.numel() > 0:
            fp2_values = k_tensor.index_select(
                -1, fp2_indices
            )

            fp2_values = FP2E1M0StyleQuantizer.quantize_dequantize(
                fp2_values,
                block_size=cls.BLOCK_SIZE
            )

            result.index_copy_(-1, fp2_indices, fp2_values)

        return result

    @classmethod
    def quantize_with_calibration(cls, calibration_k, k_tensor):
        """
        Convenience method:
          1. Compute the mask from calibration K.
          2. Apply the same mask to k_tensor.

        The calibration and application tensors must have the
        same number of channels, but may have different token counts.
        """
        channel_mask = cls.compute_channel_mask(calibration_k)

        quantized_k = cls.quantize_dequantize(
            k_tensor, channel_mask
        )

        return quantized_k, channel_mask


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Example calibration and evaluation K tensors.
    # Replace these with actual K projection/cache tensors.
    calibration_k = torch.randn(1, 128, 512, dtype=torch.float16)
    evaluation_k = torch.randn(1, 64, 512, dtype=torch.float16)

    quantized_k, channel_mask = (
        MixedPrecisionKQuantizer.quantize_with_calibration(
            calibration_k,
            evaluation_k
        )
    )

    print("Original shape:", evaluation_k.shape)
    print("Quantized shape:", quantized_k.shape)
    print("FP4 channels:", int(channel_mask.sum().item()))
    print("FP2 channels:", int((~channel_mask).sum().item()))
    print("FP4 channel fraction:", channel_mask.float().mean().item())
    print("Original dtype:", evaluation_k.dtype)
    print("Output dtype:", quantized_k.dtype)

    mse = (
        evaluation_k.float() - quantized_k.float()
    ).pow(2).mean().item()

    print("Quantization MSE:", mse)
