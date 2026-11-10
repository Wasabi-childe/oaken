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
    def fp4_e2m1_masked_quantization(tensor, mask):
        x = tensor.float()
        mask = mask.bool()
    
        # E2M1 FP4 representable values
        codebook = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
              0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
            device=x.device
        )
    
        # Find maximum absolute value for each token
        max_abs = x.abs().masked_fill(~mask, 0.0).amax(
            dim=-1, keepdim=True
        )
    
        # Scale values into the FP4 range [-6, 6]
        scale = max_abs / 6.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
        normalized = x / scale
    
        # Find nearest E2M1 representable value
        distances = torch.abs(
            normalized.unsqueeze(-1) - codebook
        )
    
        indices = distances.argmin(dim=-1)
        quantized = codebook[indices]
    
        # Reconstruct original-scale values
        reconstructed = quantized * scale
    
        return reconstructed * mask
    @staticmethod
    def tokenwise_masked_quantization(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        bits: int
    ):
        x = tensor.float()
        mask = mask.bool()
    
        qmax = (2 ** bits) - 1
    
        # Find minimum and maximum valid values for each token
        minval = x.masked_fill(~mask, float('inf')).amin(
            dim=-1, keepdim=True
        )
    
        maxval = x.masked_fill(~mask, float('-inf')).amax(
            dim=-1, keepdim=True
        )
    
        # Handle tokens with no valid values
        valid = mask.any(dim=-1, keepdim=True)
    
        minval = torch.where(
            valid, minval, torch.zeros_like(minval)
        )
    
        maxval = torch.where(
            valid, maxval, torch.zeros_like(maxval)
        )
    
        # Calculate quantization range
        rangeval = maxval - minval
    
        safe_range = torch.where(
            rangeval > 0,
            rangeval,
            torch.ones_like(rangeval)
        )
    
        # Quantize
        quantized = torch.round(
            (x - minval) * qmax / safe_range
        )
    
        # Reconstruct
        reconstructed = (
            quantized * safe_range / qmax
        ) + minval
    
        # Handle constant-valued groups
        reconstructed = torch.where(
            rangeval > 0,
            reconstructed,
            minval
        )
    
        # Ignore invalid positions
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
        quantize_outlier: bool = False,
        use_group_shift: bool = True
    ) -> tuple[torch.Tensor, list[float], torch.Tensor]:

        # Divide input into threshold groups
        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(input_tensor).to(
            input_tensor.device
        ).half()

        # Quantize every threshold group independently
        for idx in range(len(threshold_lowers)):

            mask = masks[idx]
            x = grouped_tensors[idx].float()

            # Calculate separate token-wise mean for this group
            valid = mask.any(dim=-1, keepdim=True)

            mean = x.sum(dim=-1, keepdim=True) / mask.sum(
                dim=-1, keepdim=True
            ).clamp(min=1)

            mean = torch.where(
                valid,
                mean,
                torch.zeros_like(mean)
            )

            # Subtract group-specific mean
            residual = (x - mean) * mask

            # Apply E2M1 FP4 quantization
            print("[DEBUG] Executing E2M1 FP4 quantization")

            quantized_residual = cls.fp4_e2m1_masked_quantization(
                residual,
                mask
            )

            # Restore group-specific mean
            grouped_tensors[idx] = quantized_residual + mean * mask

        # Reconstruct final tensor
        for tensor, mask in zip(grouped_tensors, masks):
            result_tensor += tensor * mask

        heat_map = None

        # Calculate fraction of values in each group
        val_frac = [
            (torch.count_nonzero(mask) / torch.numel(mask)).item()
            for mask in masks
        ]

        return result_tensor, val_frac, heat_map

