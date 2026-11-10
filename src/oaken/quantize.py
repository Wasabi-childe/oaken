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
    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float] = None,
        threshold_uppers: list[float] = None,
        quantize_outlier: bool = False,
        use_group_shift: bool = False
    ) -> tuple[torch.Tensor, list[float], torch.Tensor]:

        # Quantize the entire input token-wise using FP4
        result_tensor = cls.mean_fp4_quantization(input_tensor)

        result_tensor = result_tensor.half()

        # No threshold groups or outliers
        val_frac = [1.0]
        heat_map = None

        return result_tensor, val_frac, heat_map

