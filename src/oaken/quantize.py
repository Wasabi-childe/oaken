import torch
from typing import Optional

class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6
    
    @classmethod
    def get_outlier_threshold(cls, input_tensor: torch.Tensor, threshold_lower: float, threshold_upper: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outlier_mask = torch.logical_or(input_tensor <= threshold_lower, threshold_upper <= input_tensor)

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    
    @staticmethod
    def mean_fp4_quantization(tensor):
        x = tensor.float()
    
        # Calculate token-wise mean
        mean = x.mean(dim=-1, keepdim=True)
    
        # Calculate residuals
        residual = x - mean
    
        # E2M1 FP4 levels
        fp4_levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=x.device
        )
    
        # Calculate scale
        max_abs = residual.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / 6.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
        # Normalize residuals
        normalized = residual / scale
    
        # Find nearest FP4 level
        distances = (
            normalized.abs().unsqueeze(-1) - fp4_levels
        ).abs()
    
        indices = distances.argmin(dim=-1)
    
        # Restore sign
        quantized = fp4_levels[indices] * torch.sign(residual)
    
        # Reconstruct values
        reconstructed = quantized * scale + mean
    
        return reconstructed

    @staticmethod
    def mean_symmetric_quantization(tensor, mask, bits):
        x = tensor.float()
        mask = mask.bool()
    
        # Calculate token-wise mean
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    
        mean = x.sum(dim=-1, keepdim=True) / count
    
        valid = mask.any(dim=-1, keepdim=True)
        mean = torch.where(valid, mean, torch.zeros_like(mean))
    
        # Subtract mean
        residual = (x - mean) * mask
    
        # Calculate symmetric quantization scale
        qmax = (2 ** (bits - 1)) - 1
    
        max_abs = residual.abs().amax(dim=-1, keepdim=True)
    
        scale = max_abs / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
        # Quantize residual
        quantized = torch.round(residual / scale)
        quantized = quantized.clamp(-qmax, qmax)
    
        # Reconstruct residual and restore mean
        reconstructed = quantized * scale + mean
    
        return reconstructed * mask


    @classmethod
    def get_multigroup_threshold(cls, input_tensor: torch.Tensor, threshold_lowers: list[float], threshold_uppers: list[float]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        group_masks = list()
        group_tensors = list()
        prev_thr_low, prev_thr_up = None, None
        for idx, (thr_low, thr_up) in enumerate(zip(threshold_lowers, threshold_uppers)):
            if idx == len(threshold_lowers) - 1: 
                # Inner-most Group
                group_masks.append(mask := torch.logical_and(input_tensor > prev_thr_low, input_tensor < prev_thr_up))
            elif (prev_thr_low is not None) and (prev_thr_up is not None):
                group_masks.append(mask := torch.logical_or(
                    torch.logical_and(prev_thr_low < input_tensor, input_tensor <= thr_low),
                    torch.logical_and(thr_up <= input_tensor, input_tensor < prev_thr_up))
                )
            else:
                # Outer-most Group
                group_masks.append(mask := torch.logical_or(               input_tensor <= thr_low, thr_up <= input_tensor))
            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_tensors.append(input_tensor * mask)

        assert(len(threshold_lowers) == len(threshold_uppers) == len(group_tensors) == len(group_masks))
        return group_tensors, group_masks
    
    @staticmethod
    def uniform_quantization_threshold(tensor, bits: int, minval: torch.Tensor, maxval: torch.Tensor):
        rangeval = maxval - minval
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx
        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(quantized, nan=2 ** bits - 1)
        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()
        return OakenQuantizer.uniform_quantization_threshold(tensor, bits, minval, maxval)
    
    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)



class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    FP4_LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    )

    OUTLIERS_PER_GROUP = 4
    HEADS_PER_GROUP = 2
    GROUP_SIZE = 128

    @classmethod
    def downsample_config_c(
        cls,
        input_tensor,
        num_heads,
        heads_per_group,
        group_size,
        is_key
    ):

        original_shape = input_tensor.shape
        original_dtype = input_tensor.dtype
        
        x = input_tensor.float()
        
        head_dim = x.shape[-1] // num_heads
        
        assert x.shape[-1] == num_heads * head_dim
        assert num_heads % heads_per_group == 0
        assert heads_per_group * head_dim == group_size
        
        x = x.reshape(
            *x.shape[:-1],
            num_heads,
            head_dim
        )
        
        num_groups = num_heads // heads_per_group
        
        x = x.reshape(
            *x.shape[:-2],
            num_groups,
            group_size
        )

        # ------------------------------------------------
        # 2. Select K outliers
        # ------------------------------------------------

        if is_key:

            # Select four largest absolute values
            # independently for each token and head-pair group.

            indices = torch.topk(
                x.abs(),
                k=cls.OUTLIERS_PER_GROUP,
                dim=-1
            ).indices

            outlier_mask = torch.zeros_like(x, dtype=torch.bool)

            outlier_mask.scatter_(
                -1,
                indices,
                True
            )

            # Direct FP8 E4M3 outlier storage
            outliers = x.to(torch.float8_e4m3fn).float()

        else:

            # V has no outliers in Configuration C.
            outlier_mask = torch.zeros_like(
                x,
                dtype=torch.bool
            )

            indices = None
            outliers = None

        # ------------------------------------------------
        # 3. Separate regular values
        # ------------------------------------------------

        regular = x.masked_fill(outlier_mask, 0.0)

        # ------------------------------------------------
        # 4. Calculate one scale per 128-value group
        # ------------------------------------------------

        max_abs = regular.abs().amax(
            dim=-1,
            keepdim=True
        )

        scale = max_abs / 6.0

        scale = torch.where(
            scale > 0,
            scale,
            torch.ones_like(scale)
        )

        # Store scale as FP8 E4M3
        scale = scale.to(torch.float8_e4m3fn).float()

        # Handle possible FP8 underflow
        scale = torch.where(
            scale > 0,
            scale,
            torch.full_like(scale, 2 ** -9)
        )

        # ------------------------------------------------
        # 5. Quantize regular values to FP4 E2M1
        # ------------------------------------------------

        normalized = regular / scale

        levels = cls.FP4_LEVELS.to(
            device=x.device,
            dtype=x.dtype
        )

        distances = (
            normalized.abs().unsqueeze(-1) - levels
        ).abs()

        nearest_indices = distances.argmin(dim=-1)

        quantized = (
            levels[nearest_indices] *
            torch.sign(regular)
        )

        reconstructed = quantized * scale

        # ------------------------------------------------
        # 6. Restore K outliers
        # ------------------------------------------------

        if is_key:

            reconstructed = torch.where(
                outlier_mask,
                outliers,
                reconstructed
            )

        # ------------------------------------------------
        # 7. Restore original tensor shape
        # ------------------------------------------------

        reconstructed = reconstructed.reshape(original_shape)

        reconstructed = reconstructed.to(original_dtype)

        # Fraction of values stored as outliers.
        # K = 4/128; V = 0/128.

        outlier_fraction = (
            cls.OUTLIERS_PER_GROUP / group_size
            if is_key else 0.0
        )

        return reconstructed, [outlier_fraction], None
