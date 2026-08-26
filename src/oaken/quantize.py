import torch
from typing import Optional


class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    # ============================================================
    # OUTLIER THRESHOLD
    # ============================================================

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float
    ):

        outlier_mask = torch.logical_or(
            input_tensor <= threshold_lower,
            input_tensor >= threshold_upper
        )

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * (~outlier_mask)

        return inlier, outlier, outlier_mask

    # ============================================================
    # MULTI-GROUP THRESHOLD
    # ============================================================

    @classmethod
    def get_multigroup_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float]
    ):

        group_masks = []
        group_tensors = []

        prev_thr_low = None
        prev_thr_up = None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):

            # ----------------------------------------------------
            # INNER-MOST GROUP
            # ----------------------------------------------------

            if idx == len(threshold_lowers) - 1:

                mask = torch.logical_and(
                    input_tensor > prev_thr_low,
                    input_tensor < prev_thr_up
                )

            # ----------------------------------------------------
            # MIDDLE GROUP
            # ----------------------------------------------------

            elif (
                prev_thr_low is not None
                and prev_thr_up is not None
            ):

                mask = torch.logical_or(

                    torch.logical_and(
                        input_tensor > prev_thr_low,
                        input_tensor <= thr_low
                    ),

                    torch.logical_and(
                        input_tensor >= thr_up,
                        input_tensor < prev_thr_up
                    )
                )

            # ----------------------------------------------------
            # OUTER-MOST GROUP
            # ----------------------------------------------------

            else:

                mask = torch.logical_or(
                    input_tensor <= thr_low,
                    input_tensor >= thr_up
                )

            group_masks.append(mask)
            group_tensors.append(
                input_tensor * mask
            )

            prev_thr_low = thr_low
            prev_thr_up = thr_up

        return group_tensors, group_masks

    # ============================================================
    # FP16 RECONSTRUCTION FROM QUANTIZATION
    # ============================================================

    @staticmethod
    def uniform_quantization_threshold(
        tensor: torch.Tensor,
        bits: int,
        minval,
        maxval
    ):

        levels = 2 ** bits - 1

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

        safe_range = torch.where(
            torch.abs(rangeval) < OakenQuantizer.FLOAT_TOLERANCE,
            torch.ones_like(rangeval),
            rangeval
        )

        qx = levels / safe_range
        offset = minval * qx

        quantized = torch.round(
            qx * tensor - offset
        )

        quantized = torch.nan_to_num(
            quantized,
            nan=float(levels),
            posinf=float(levels),
            neginf=0.0
        )

        quantized = torch.clamp(
            quantized,
            0,
            levels
        )

        reconstructed = (
            quantized + offset
        ) / qx

        return reconstructed

    # ============================================================
    # INTEGER QUANTIZATION INDICES
    # ============================================================

    @staticmethod
    def uniform_quantization_indices(
        tensor: torch.Tensor,
        bits: int,
        minval,
        maxval
    ):

        """
        Returns integer quantization indices.

        4-bit:
            0 ... 15

        5-bit:
            0 ... 31
        """

        levels = 2 ** bits - 1

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

        safe_range = torch.where(
            torch.abs(rangeval) < OakenQuantizer.FLOAT_TOLERANCE,
            torch.ones_like(rangeval),
            rangeval
        )

        qx = levels / safe_range
        offset = minval * qx

        indices = torch.round(
            qx * tensor - offset
        )

        indices = torch.nan_to_num(
            indices,
            nan=float(levels),
            posinf=float(levels),
            neginf=0.0
        )

        indices = torch.clamp(
            indices,
            0,
            levels
        )

        return indices.to(torch.uint8)

    # ============================================================
    # ORIGINAL UNIFORM QUANTIZATION
    # ============================================================

    @staticmethod
    def uniform_quantization(
        tensor: torch.Tensor,
        bits: int
    ):

        maxval = torch.max(tensor).item()
        minval = torch.min(tensor).item()

        return OakenQuantizer.uniform_quantization_threshold(
            tensor,
            bits,
            minval,
            maxval
        )

    # ============================================================
    # MANTISSA DOWNSAMPLING
    # ============================================================

    @classmethod
    def downsample_mantissa(
        cls,
        tensor: torch.Tensor
    ):

        int16_tensor = tensor.view(torch.int16)

        truncated = (
            int16_tensor
            & 0b1_11111_1110_0000_00
        )

        return truncated.view(torch.float16)


# =================================================================
# MULTI-THRESHOLD TOKENWISE QUANTIZER
# =================================================================

class MultiThresholdTokenwiseQuantizer(OakenQuantizer):

    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float],
        quantize_outlier: bool = False,
        use_group_shift: bool = True
    ):

        # ========================================================
        # CREATE GROUPS
        # ========================================================

        grouped_tensors, masks = (
            cls.get_multigroup_threshold(
                input_tensor,
                threshold_lowers,
                threshold_uppers
            )
        )

        n_groups = len(threshold_lowers)

        # ========================================================
        # OUTPUT FP16 TENSOR
        # ========================================================

        result_tensor = torch.zeros_like(
            input_tensor
        ).half()

        # ========================================================
        # QUANTIZATION INDICES
        #
        # Every value stores its integer quantization index.
        #
        # 4-bit -> 0 ... 15
        # 5-bit -> 0 ... 31
        # ========================================================

        quant_indices = torch.zeros_like(
            input_tensor,
            dtype=torch.uint8
        )

        # ========================================================
        # GROUP IDS
        #
        # 0 = outer-most
        # 1 = next group
        # ...
        # n_groups-1 = inner-most
        # ========================================================

        group_ids = torch.zeros_like(
            input_tensor,
            dtype=torch.uint8
        )

        # ========================================================
        # BIT WIDTH
        #
        # Stores the number of bits used for each value.
        # ========================================================

        quant_bits = torch.zeros_like(
            input_tensor,
            dtype=torch.uint8
        )

        # ========================================================
        # QUANTIZATION PARAMETERS
        #
        # These are required later to reconstruct FP16 values.
        #
        # minval
        # maxval
        #
        # Shape is compatible with the input tensor except that
        # the final dimension is reduced to 1 where appropriate.
        # ========================================================

        quant_min = torch.zeros_like(
            input_tensor,
            dtype=torch.float32
        )

        quant_max = torch.zeros_like(
            input_tensor,
            dtype=torch.float32
        )

        # ========================================================
        # QUANTIZE OUTLIERS
        # ========================================================

        if quantize_outlier:

            # ====================================================
            # INNER-MOST GROUP
            # ====================================================

            inner_idx = n_groups - 1

            inner_tensor = grouped_tensors[inner_idx]
            inner_mask = masks[inner_idx]

            # Oaken uses token-wise min/max here.
            minval_tensor = torch.min(
                inner_tensor,
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                inner_tensor,
                dim=-1
            ).values.unsqueeze(-1)

            bits = cls.OUTLIER_BITS

            # ----------------------------------------------------
            # SAVE QUANTIZATION PARAMETERS
            # ----------------------------------------------------

            quant_min[inner_mask] = (
                minval_tensor
                .expand_as(inner_tensor)[inner_mask]
                .float()
            )

            quant_max[inner_mask] = (
                maxval_tensor
                .expand_as(inner_tensor)[inner_mask]
                .float()
            )

            # ----------------------------------------------------
            # GET INTEGER INDICES
            # ----------------------------------------------------

            group_indices = (
                cls.uniform_quantization_indices(
                    inner_tensor,
                    bits,
                    minval_tensor,
                    maxval_tensor
                )
            )

            quant_indices[inner_mask] = (
                group_indices[inner_mask]
            )

            group_ids[inner_mask] = inner_idx
            quant_bits[inner_mask] = bits

            # ----------------------------------------------------
            # RECONSTRUCT FP16
            # ----------------------------------------------------

            grouped_tensors[inner_idx] = (
                cls.uniform_quantization_threshold(
                    inner_tensor,
                    bits,
                    minval_tensor,
                    maxval_tensor
                )
            )

            # ====================================================
            # OUTER / MIDDLE GROUPS
            # ====================================================

            for idx in range(n_groups - 1):

                current_tensor = grouped_tensors[idx]
                current_mask = masks[idx]

                # ------------------------------------------------
                # THRESHOLDS
                # ------------------------------------------------

                threshold_lower = torch.tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )

                threshold_upper = torch.tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )

                # ------------------------------------------------
                # POSITIVE / NEGATIVE OUTLIERS
                # ------------------------------------------------

                higher_mask = current_tensor > 0
                lower_mask = current_tensor < 0

                higher_outlier = (
                    current_tensor * higher_mask
                )

                lower_outlier = (
                    current_tensor * lower_mask
                )

                # ------------------------------------------------
                # GROUP SHIFT
                # ------------------------------------------------

                if use_group_shift:

                    higher_outlier = (
                        higher_outlier
                        - threshold_upper
                    )

                    lower_outlier = (
                        lower_outlier
                        - threshold_lower
                    )

                combined = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

                # ------------------------------------------------
                # BIT WIDTH
                #
                # Last outer group = 4-bit
                # Earlier groups = 5-bit
                # ------------------------------------------------

                if idx == n_groups - 2:
                    bits = cls.QUANTIZE_BITS
                else:
                    bits = cls.OUTLIER_BITS

                # ------------------------------------------------
                # SAME MIN/MAX USED FOR RECONSTRUCTION
                # ------------------------------------------------

                minval = torch.min(
                    combined
                ).item()

                maxval = torch.max(
                    combined
                ).item()

                # ------------------------------------------------
                # SAVE MIN/MAX
                # ------------------------------------------------

                quant_min[current_mask] = minval
                quant_max[current_mask] = maxval

                # ------------------------------------------------
                # INTEGER INDICES
                # ------------------------------------------------

                group_indices = (
                    cls.uniform_quantization_indices(
                        combined,
                        bits,
                        minval,
                        maxval
                    )
                )

                quant_indices[current_mask] = (
                    group_indices[current_mask]
                )

                group_ids[current_mask] = idx
                quant_bits[current_mask] = bits

                # ------------------------------------------------
                # RECONSTRUCT
                # ------------------------------------------------

                total_outlier = (
                    cls.uniform_quantization_threshold(
                        combined,
                        bits,
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
                # REVERSE GROUP SHIFT
                # ------------------------------------------------

                if use_group_shift:

                    higher_outlier = (
                        higher_outlier
                        + threshold_upper
                    )

                    lower_outlier = (
                        lower_outlier
                        + threshold_lower
                    )

                grouped_tensors[idx] = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

        # ========================================================
        # NON-OUTLIER MODE
        # ========================================================

        else:

            idx = n_groups - 2

            tensor = grouped_tensors[idx]

            minval_tensor = torch.min(
                tensor,
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                tensor,
                dim=-1
            ).values.unsqueeze(-1)

            grouped_tensors[idx] = (
                cls.uniform_quantization_threshold(
                    tensor,
                    cls.QUANTIZE_BITS,
                    minval_tensor,
                    maxval_tensor
                )
            )

        # ========================================================
        # REBUILD COMPLETE QUANTIZED FP16 TENSOR
        # ========================================================

        for tensor, mask in zip(
            grouped_tensors,
            masks
        ):

            result_tensor += (
                tensor * mask
            )

        # ========================================================
        # GROUP FRACTIONS
        # ========================================================

        val_frac = [
            (
                torch.count_nonzero(mask)
                / torch.numel(mask)
            ).item()
            for mask in masks
        ]

        # ========================================================
        # HEAT MAP
        # ========================================================

        heat_map = None

        # ========================================================
        # QUANTIZATION PARAMETER DICTIONARY
        # ========================================================

        quant_params = {
            "min": quant_min,
            "max": quant_max
        }

        # ========================================================
        # RETURN
        # ========================================================

        return (
            result_tensor,
            val_frac,
            heat_map,
            quant_indices,
            group_ids,
            quant_bits,
            quant_params
        )
