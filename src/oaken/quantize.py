import torch
from typing import Optional
from .huffman import HuffmanCodec

class OakenQuantizer:
    QUANTIZE_BITS = 8
    OUTLIER_BITS = 9
    FLOAT_TOLERANCE = 1e-6
    
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
    def uniform_quantization_threshold(
        tensor,
        bits: int,
        minval,
        maxval,
        return_indices: bool = False
    ):
        # Convert min/max to tensors
        minval = torch.as_tensor(
            minval,
            device=tensor.device,
            dtype=tensor.dtype
        )
    
        maxval = torch.as_tensor(
            maxval,
            device=tensor.device,
            dtype=tensor.dtype
        )
    
        rangeval = maxval - minval
    
        # Prevent division by zero
        rangeval = torch.clamp(
            rangeval,
            min=1e-12
        )
    
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx
    
        # Quantization
        quantized = torch.round(
            qx * tensor - offset
        )
    
        quantized = torch.nan_to_num(
            quantized,
            nan=2 ** bits - 1
        )
    
        # Valid quantization range
        quantized = torch.clamp(
            quantized,
            0,
            2 ** bits - 1
        )
    
        # Return integer indices for Huffman
        if return_indices:
            return quantized.to(torch.int32)
    
        # Normal Oaken behavior: dequantization
        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()
        return OakenQuantizer.uniform_quantization_threshold(tensor, bits, minval, maxval)

    @staticmethod
    def huffman_test(tensor, bits: int):
    
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()
    
        quantized_indices = (
            OakenQuantizer.uniform_quantization_threshold(
                tensor,
                bits,
                minval,
                maxval,
                return_indices=True
            )
        )
    
        result = HuffmanCodec.measure_compression(
            quantized_indices,
            bits_per_symbol=bits
        )
    
        return result
    
    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)

import torch

from .huffman import HuffmanCodec


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float],
        quantize_outlier: bool = False,
        use_group_shift: bool = True,
        return_huffman_stats: bool = False
    ):

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(
            input_tensor
        ).to(input_tensor.device).half()

        # Store Huffman results
        huffman_stats = []

        # ========================================================
        # QUANTIZE OUTLIERS
        # ========================================================

        if quantize_outlier:

            # ----------------------------------------------------
            # INNER-MOST GROUP
            # ----------------------------------------------------

            minval_tensor = torch.min(
                grouped_tensors[-1],
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                grouped_tensors[-1],
                dim=-1
            ).values.unsqueeze(-1)

            # Get actual quantization indices
            inner_indices = cls.uniform_quantization_threshold(
                grouped_tensors[-1],
                cls.OUTLIER_BITS,
                minval_tensor,
                maxval_tensor,
                return_indices=True
            )

            # Dequantize as before
            grouped_tensors[-1] = cls.uniform_quantization_threshold(
                grouped_tensors[-1],
                cls.OUTLIER_BITS,
                minval_tensor,
                maxval_tensor
            )

            # Huffman measurement
            if return_huffman_stats:

                valid_indices = inner_indices[
                    masks[-1]
                ]

                if valid_indices.numel() > 0:

                    stats = HuffmanCodec.measure_compression(
                        valid_indices,
                        bits_per_symbol=cls.OUTLIER_BITS
                    )

                    huffman_stats.append({
                        "group": len(threshold_lowers) - 1,
                        "bits": cls.OUTLIER_BITS,
                        **stats
                    })

            # ----------------------------------------------------
            # OUTER GROUPS
            # ----------------------------------------------------

            for idx in range(
                len(threshold_lowers) - 1
            ):

                threshold_lower_tensor = torch.tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )

                threshold_upper_tensor = torch.tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )

                higher_mask = (
                    grouped_tensors[idx] > 0
                )

                lower_mask = (
                    grouped_tensors[idx] < 0
                )

                higher_outlier = (
                    grouped_tensors[idx]
                    * higher_mask
                )

                lower_outlier = (
                    grouped_tensors[idx]
                    * lower_mask
                )

                # ------------------------------------------------
                # GROUP SHIFT
                # ------------------------------------------------

                if use_group_shift:

                    higher_outlier -= (
                        threshold_upper_tensor
                    )

                    lower_outlier -= (
                        threshold_lower_tensor
                    )

                shifted_tensor = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

                # ------------------------------------------------
                # QUANTIZATION BIT WIDTH
                # ------------------------------------------------

                if idx == len(threshold_lowers) - 2:
                    quant_bits = cls.QUANTIZE_BITS
                else:
                    quant_bits = cls.OUTLIER_BITS

                # ------------------------------------------------
                # GET QUANTIZATION INDICES
                # ------------------------------------------------

                minval = torch.min(
                    shifted_tensor
                ).item()

                maxval = torch.max(
                    shifted_tensor
                ).item()

                outer_indices = (
                    cls.uniform_quantization_threshold(
                        shifted_tensor,
                        quant_bits,
                        minval,
                        maxval,
                        return_indices=True
                    )
                )

                # ------------------------------------------------
                # DEQUANTIZE
                # ------------------------------------------------

                total_outlier = (
                    cls.uniform_quantization_threshold(
                        shifted_tensor,
                        quant_bits,
                        minval,
                        maxval
                    )
                )

                higher_outlier = (
                    total_outlier
                    * higher_mask
                )

                lower_outlier = (
                    total_outlier
                    * lower_mask
                )

                # ------------------------------------------------
                # UNDO GROUP SHIFT
                # ------------------------------------------------

                if use_group_shift:

                    higher_outlier += (
                        threshold_upper_tensor
                    )

                    lower_outlier += (
                        threshold_lower_tensor
                    )

                grouped_tensors[idx] = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

                # ------------------------------------------------
                # HUFFMAN
                # ------------------------------------------------

                if return_huffman_stats:

                    valid_mask = masks[idx]

                    valid_indices = outer_indices[
                        valid_mask
                    ]

                    if valid_indices.numel() > 0:

                        stats = HuffmanCodec.measure_compression(
                            valid_indices,
                            bits_per_symbol=quant_bits
                        )

                        huffman_stats.append({
                            "group": idx,
                            "bits": quant_bits,
                            **stats
                        })

        # ========================================================
        # NORMAL OAKEN MODE
        # ========================================================

        else:

            # ----------------------------------------------------
            # MIDDLE GROUP
            # ----------------------------------------------------

            minval_tensor = torch.min(
                grouped_tensors[-2],
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                grouped_tensors[-2],
                dim=-1
            ).values.unsqueeze(-1)

            # Actual quantization indices
            middle_indices = (
                cls.uniform_quantization_threshold(
                    grouped_tensors[-2],
                    cls.QUANTIZE_BITS,
                    minval_tensor,
                    maxval_tensor,
                    return_indices=True
                )
            )

            # Normal Oaken dequantized output
            grouped_tensors[-2] = (
                cls.uniform_quantization_threshold(
                    grouped_tensors[-2],
                    cls.QUANTIZE_BITS,
                    minval_tensor,
                    maxval_tensor
                )
            )

            # ----------------------------------------------------
            # HUFFMAN
            # ----------------------------------------------------

            if return_huffman_stats:

                valid_indices = middle_indices[
                    masks[-2]
                ]

                if valid_indices.numel() > 0:

                    stats = HuffmanCodec.measure_compression(
                        valid_indices,
                        bits_per_symbol=cls.QUANTIZE_BITS
                    )

                    huffman_stats.append({
                        "group": len(threshold_lowers) - 2,
                        "bits": cls.QUANTIZE_BITS,
                        **stats
                    })

        # ========================================================
        # RECONSTRUCT RESULT
        # ========================================================

        for idx, (tensor, mask) in enumerate(
            zip(grouped_tensors, masks)
        ):

            result_tensor += (
                tensor * mask
            )

        # ========================================================
        # SPARSITY
        # ========================================================

        val_frac = [
            (
                torch.count_nonzero(mask)
                / torch.numel(mask)
            ).item()
            for mask in masks
        ]

        heat_map = None

        # ========================================================
        # RETURN
        # ========================================================

        if return_huffman_stats:

            return (
                result_tensor,
                val_frac,
                heat_map,
                huffman_stats
            )

        return (
            result_tensor,
            val_frac,
            heat_map
        )
