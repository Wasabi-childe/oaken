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
    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float],
        quantize_outlier: bool = True,
        use_group_shift: bool = True
    ) -> tuple[torch.Tensor, list[float], torch.Tensor]:

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor, threshold_lowers, threshold_uppers
        )

        result_tensor = torch.zeros_like(input_tensor).to(
            input_tensor.device
        ).half()
        
        if quantize_outlier:

            # Preserve original bit allocation
            if idx == len(threshold_lowers) - 2:
                bits = cls.QUANTIZE_BITS
            else:
                bits = cls.QUANTIZE_BITS
            
            # Mean subtraction + symmetric quantization + mean restoration
            grouped_tensors[idx] = cls.mean_symmetric_quantization(
                x, mask, bits
            )

        else:
            # Quantize middle group using mean subtraction + symmetric quantization
            grouped_tensors[-2] = cls.mean_symmetric_quantization(
                grouped_tensors[-2],
                masks[-2],
                cls.QUANTIZE_BITS
            )
        # Reconstruct final tensor
        for idx, (tensor, mask) in enumerate(
            zip(grouped_tensors, masks)
        ):
            result_tensor += tensor * mask

        heat_map = None

        val_frac = [
            (torch.count_nonzero(mask) / torch.numel(mask)).item()
            for mask in masks
        ]

        return result_tensor, val_frac, heat_map
now i wanna remove outlier handling
