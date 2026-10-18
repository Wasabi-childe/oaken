
import torch
from typing import Optional


class FP4E4M3Quantizer:
    BLOCK_SIZE = 16
    FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    @classmethod
    def quantize_dequantize(cls, tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        if tensor.numel() == 0:
            return tensor.clone()

        if tensor.dim() == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.dim() == 1:
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.reshape(-1, tensor.shape[-1])

        result = torch.zeros_like(tensor)
        levels = torch.tensor(
            cls.FP4_LEVELS,
            device=tensor.device,
            dtype=torch.float32
        )

        # Quantize each row in blocks of 16 values.
        for row_idx in range(tensor.shape[0]):
            for start in range(0, tensor.shape[1], cls.BLOCK_SIZE):
                end = min(start + cls.BLOCK_SIZE, tensor.shape[1])

                block = tensor[row_idx, start:end].float()
                max_abs = torch.max(torch.abs(block))

                if max_abs == 0:
                    continue

                # E4M3 block scale: maximum FP4 magnitude is 6.
                scale = max_abs / 6.0

                # Round the scale to a finite E4M3 value.
                scale = torch.clamp(scale, min=2.0 ** -9, max=448.0)
                scale = scale.to(torch.float8_e4m3fn).float()

                # Guard against a scale that rounds to zero.
                if scale == 0:
                    scale = torch.tensor(
                        2.0 ** -9,
                        device=tensor.device,
                        dtype=torch.float32
                    )

                normalized = torch.abs(block) / scale

                # Map each value to its nearest E2M1 level.
                distances = torch.abs(
                    normalized.unsqueeze(-1) - levels
                )
                nearest_idx = torch.argmin(distances, dim=-1)
                quantized = levels[nearest_idx] * scale

                # Restore the sign.
                quantized = torch.sign(block) * quantized

                result[row_idx, start:end] = quantized

        return result.reshape(original_shape).to(original_dtype)


class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        outlier_mask = torch.logical_or(
            input_tensor <= threshold_lower,
            threshold_upper <= input_tensor
        )

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        group_masks = list()
        group_tensors = list()
        prev_thr_low, prev_thr_up = None, None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):
            if idx == len(threshold_lowers) - 1:
                # Inner-most Group
                group_masks.append(
                    mask := torch.logical_and(
                        input_tensor > prev_thr_low,
                        input_tensor < prev_thr_up
                    )
                )

            elif (
                prev_thr_low is not None
                and prev_thr_up is not None
            ):
                group_masks.append(
                    mask := torch.logical_or(
                        torch.logical_and(
                            prev_thr_low < input_tensor,
                            input_tensor <= thr_low
                        ),
                        torch.logical_and(
                            thr_up <= input_tensor,
                            input_tensor < prev_thr_up
                        )
                    )
                )

            else:
                # Outer-most Group
                group_masks.append(
                    mask := torch.logical_or(
                        input_tensor <= thr_low,
                        thr_up <= input_tensor
                    )
                )

            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_tensors.append(input_tensor * mask)

        assert (
            len(threshold_lowers)
            == len(threshold_uppers)
            == len(group_tensors)
            == len(group_masks)
        )

        return group_tensors, group_masks

    @staticmethod
    def uniform_quantization_threshold(
        tensor,
        bits: int,
        minval: torch.Tensor,
        maxval: torch.Tensor
    ):
        rangeval = maxval - minval
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx

        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(
            quantized,
            nan=2 ** bits - 1
        )

        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()

        return OakenQuantizer.uniform_quantization_threshold(
            tensor, bits, minval, maxval
        )

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

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(input_tensor).to(
            input_tensor.device
        ).half()

        if quantize_outlier:
            # Quantize inner-most group using the original INT5 path.
            minval_tensor = torch.min(
                grouped_tensors[-1], dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                grouped_tensors[-1], dim=-1
            ).values.unsqueeze(-1)

            grouped_tensors[-1] = cls.uniform_quantization_threshold(
                grouped_tensors[-1],
                cls.OUTLIER_BITS,
                minval_tensor,
                maxval_tensor
            )

            # Quantize outer groups.
            for idx in range(len(threshold_lowers) - 1):
                threshold_lower_tensor = torch.tensor(
                    threshold_lowers[idx]
                ).to(input_tensor.device).half()

                threshold_upper_tensor = torch.tensor(
                    threshold_uppers[idx]
                ).to(input_tensor.device).half()

                higher_mask = grouped_tensors[idx] > 0
                lower_mask = grouped_tensors[idx] < 0

                higher_outlier = grouped_tensors[idx] * higher_mask
                lower_outlier = grouped_tensors[idx] * lower_mask

                if use_group_shift:
                    higher_outlier -= threshold_upper_tensor
                    lower_outlier -= threshold_lower_tensor

                if idx == len(threshold_lowers) - 2:
                    # REPLACED: INT4 -> FP4 + E4M3
                    total_outlier = (
                        FP4E4M3Quantizer.quantize_dequantize(
                            higher_outlier * higher_mask
                            + lower_outlier * lower_mask
                        )
                    )
                else:
                    # Keep the original INT5 path unchanged.
                    total_outlier = cls.uniform_quantization(
                        higher_outlier * higher_mask
                        + lower_outlier * lower_mask,
                        cls.OUTLIER_BITS
                    )

                higher_outlier = total_outlier * higher_mask
                lower_outlier = total_outlier * lower_mask

                if use_group_shift:
                    higher_outlier += threshold_upper_tensor
                    lower_outlier += threshold_lower_tensor

                grouped_tensors[idx] = (
                    higher_outlier * higher_mask
                    + lower_outlier * lower_mask
                )

        else:
            # Quantize middle group.
            # REPLACED: INT4 -> FP4 + E4M3
            grouped_tensors[-2] = (
                FP4E4M3Quantizer.quantize_dequantize(
                    grouped_tensors[-2]
                )
            )

        # Reconstruct the result using the original group masks.
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
