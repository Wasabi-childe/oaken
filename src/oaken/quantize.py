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

        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(
            input_tensor
        ).to(input_tensor.device).half()

        if quantize_outlier:

            # ====================================================
            # INNER GROUP
            # ====================================================

            minval_tensor = torch.min(
                grouped_tensors[-1],
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                grouped_tensors[-1],
                dim=-1
            ).values.unsqueeze(-1)

            grouped_tensors[-1] = (
                cls.uniform_quantization_threshold(
                    grouped_tensors[-1],
                    cls.OUTLIER_BITS,
                    minval_tensor,
                    maxval_tensor
                )
            )

            # ====================================================
            # OUTER GROUPS
            # ====================================================

            for idx in range(
                len(threshold_lowers) - 1
            ):

                threshold_lower_tensor = torch.tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device
                ).half()

                threshold_upper_tensor = torch.tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device
                ).half()

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

                # Group shifting
                if use_group_shift:

                    higher_outlier -= (
                        threshold_upper_tensor
                    )

                    lower_outlier -= (
                        threshold_lower_tensor
                    )

                if idx == len(threshold_lowers) - 2:

                    total_outlier = cls.uniform_quantization(
                        higher_outlier * higher_mask
                        + lower_outlier * lower_mask,
                        cls.QUANTIZE_BITS
                    )

                else:

                    total_outlier = cls.uniform_quantization(
                        higher_outlier * higher_mask
                        + lower_outlier * lower_mask,
                        cls.OUTLIER_BITS
                    )

                higher_outlier = (
                    total_outlier
                    * higher_mask
                )

                lower_outlier = (
                    total_outlier
                    * lower_mask
                )

                # Undo group shift
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

        else:

            # ====================================================
            # MIDDLE GROUP
            # ====================================================

            minval_tensor = torch.min(
                grouped_tensors[-2],
                dim=-1
            ).values.unsqueeze(-1)

            maxval_tensor = torch.max(
                grouped_tensors[-2],
                dim=-1
            ).values.unsqueeze(-1)

            grouped_tensors[-2] = (
                cls.uniform_quantization_threshold(
                    grouped_tensors[-2],
                    cls.QUANTIZE_BITS,
                    minval_tensor,
                    maxval_tensor
                )
            )

        # ========================================================
        # RECONSTRUCT RESULT
        # ========================================================

        for tensor, mask in zip(
            grouped_tensors,
            masks
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

        # No delta encoding.
        # No additional processing of result_tensor.

        return (
            result_tensor,
            val_frac,
            heat_map
        )
