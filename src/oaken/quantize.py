import torch
from typing import Optional

class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6
    @staticmethod
    def tokenwise_masked_quantization(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        bits: int
    ):
        x = tensor.float()
        mask = mask.bool()
    
        qmax = (2 ** bits) - 1
    
        # Find valid values for each token
        minval = x.masked_fill(~mask, float('inf')).amin(
            dim=-1, keepdim=True
        )
    
        maxval = x.masked_fill(~mask, float('-inf')).amax(
            dim=-1, keepdim=True
        )
    
        # Handle tokens with no valid values
        valid = mask.any(dim=-1, keepdim=True)
    
        minval = torch.where(valid, minval, torch.zeros_like(minval))
        maxval = torch.where(valid, maxval, torch.zeros_like(maxval))
    
        rangeval = maxval - minval
        safe_range = torch.where(
            rangeval > 0, rangeval, torch.ones_like(rangeval)
        )
    
        # Uniform quantization
        quantized = torch.round(
            (x - minval) * qmax / safe_range
        )
    
        reconstructed = (
            quantized * safe_range / qmax
        ) + minval
    
        # Handle constant-valued groups
        reconstructed = torch.where(
            rangeval > 0, reconstructed, minval
        )
    
        return reconstructed * mask
    
    @classmethod
    def get_outlier_threshold(cls, input_tensor: torch.Tensor, threshold_lower: float, threshold_upper: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outlier_mask = torch.logical_or(input_tensor <= threshold_lower, threshold_upper <= input_tensor)

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

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
    def uniform_quantization_threshold(tensor, bits: int, minval, maxval):
        rangeval = maxval - minval
        qmax = (2 ** bits) - 1
    
        safe_range = torch.where(
            rangeval > 0,
            rangeval,
            torch.ones_like(rangeval)
        )
    
        quantized = torch.round(
            (tensor - minval) * qmax / safe_range
        )
    
        reconstructed = quantized * safe_range / qmax + minval
    
        reconstructed = torch.where(
            rangeval > 0,
            reconstructed,
            minval
        )
    
        return reconstructed

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
    
        result_tensor = torch.zeros_like(input_tensor).half()
    
        if quantize_outlier:
    
            # Quantize all groups using token-wise mean subtraction
            for idx in range(len(threshold_lowers)):
    
                mask = masks[idx]
                x = grouped_tensors[idx].float()
    
                # Calculate token-wise mean for this group
                count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    
                mean = (
                    (x * mask).sum(dim=-1, keepdim=True) / count
                )
    
                # Subtract mean
                residual = (x - mean) * mask
    
                # Preserve original bit allocation
                if idx == len(threshold_lowers) - 2:
                    bits = cls.QUANTIZE_BITS
                else:
                    bits = cls.OUTLIER_BITS
    
                # Quantize residuals
                quantized_residual = cls.tokenwise_masked_quantization(
                    residual,
                    mask,
                    bits
                )
    
                # Add mean back
                reconstructed = quantized_residual + mean * mask
    
                grouped_tensors[idx] = reconstructed.to(input_tensor.dtype)
    
        else:
    
            # Quantize middle group
            middle_mask = masks[-2]
            middle_tensor = grouped_tensors[-2]
    
            minval_tensor = middle_tensor.masked_fill(
                ~middle_mask, float('inf')
            ).amin(dim=-1, keepdim=True)
    
            maxval_tensor = middle_tensor.masked_fill(
                ~middle_mask, float('-inf')
            ).amax(dim=-1, keepdim=True)
    
            valid = middle_mask.any(dim=-1, keepdim=True)
    
            minval_tensor = torch.where(
                valid, minval_tensor, torch.zeros_like(minval_tensor)
            )
    
            maxval_tensor = torch.where(
                valid, maxval_tensor, torch.zeros_like(maxval_tensor)
            )
    
            grouped_tensors[-2] = cls.uniform_quantization_threshold(
                middle_tensor,
                cls.QUANTIZE_BITS,
                minval_tensor,
                maxval_tensor
            ) * middle_mask
    
        # Combine reconstructed groups
        for tensor, mask in zip(grouped_tensors, masks):
            result_tensor += tensor * mask
    
        val_frac = [
            (torch.count_nonzero(mask) / torch.numel(mask)).item()
            for mask in masks
        ]
    
        heat_map = None
    
        return result_tensor, val_frac, heat_map

