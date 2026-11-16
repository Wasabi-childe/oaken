
from functools import partial
import json

from src.oaken.quantize import *


def multi_group_oaken_main(args, model, tokenizer, device, runner):

    # ------------------------------------------------
    # Load offline calibration statistics
    # ------------------------------------------------

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_layer = len(model.get_decoder().layers)

    # Check that calibration statistics exist for every layer
    for kv_type in ["key", "value"]:

        required_fields = [
            "lower_threshold",
            "upper_threshold",
            "mean_abs_outlier"
        ]

        for field in required_fields:

            if field not in quantizer_stat[kv_type]:
                raise ValueError(
                    "Missing " + field +
                    " in " + kv_type + " calibration statistics."
                )

            if len(quantizer_stat[kv_type][field]) != n_layer:
                raise ValueError(
                    "Incorrect number of " + field +
                    " entries for " + kv_type + "."
                )

    # ------------------------------------------------
    # Initialize sparsity statistics
    # ------------------------------------------------

    sparsity_information = {
        "key": [0.0 for _ in range(n_layer)],
        "value": [0.0 for _ in range(n_layer)],

        "key_counter": [0 for _ in range(n_layer)],
        "value_counter": [0 for _ in range(n_layer)]
    }

    # ------------------------------------------------
    # V projection quantization hook
    # ------------------------------------------------

    def value_quantize_activation_hook(i, module, input, output):

        # Load calibrated thresholds
        value_lower = quantizer_stat["value"]["lower_threshold"][i][0]
        value_upper = quantizer_stat["value"]["upper_threshold"][i][0]

        # Load offline-calibrated layer-wise mean
        value_mean = quantizer_stat["value"]["mean_abs_outlier"][i][0]

        # Quantize and reconstruct V
        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                [value_lower],
                [value_upper],
                cmean_abs_outlier=value_mean
            )
        )

        # Accumulate sparsity
        sparsity_information["value"][i] += sparsity[0]
        sparsity_information["value_counter"][i] += 1

        return tensor.half()

    # ------------------------------------------------
    # K projection quantization hook
    # ------------------------------------------------

    def key_quantize_activation_hook(i, module, input, output):

        # Load calibrated thresholds
        key_lower = quantizer_stat["key"]["lower_threshold"][i][0]
        key_upper = quantizer_stat["key"]["upper_threshold"][i][0]

        # Load offline-calibrated layer-wise mean
        key_mean = quantizer_stat["key"]["mean_abs_outlier"][i][0]

        # Quantize and reconstruct K
        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                [key_lower],
                [key_upper],
                mean_abs_outlier=key_mean
            )
        )

        # Accumulate sparsity
        sparsity_information["key"][i] += sparsity[0]
        sparsity_information["key_counter"][i] += 1

        return tensor.half()

    # ------------------------------------------------
    # Register hooks
    # ------------------------------------------------

    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.v_proj.register_forward_hook(
            partial(value_quantize_activation_hook, i)
        )

        decoder.self_attn.k_proj.register_forward_hook(
            partial(key_quantize_activation_hook, i)
        )

    # ------------------------------------------------
    # Run model
    # ------------------------------------------------

    runner(
        args,
        model,
        tokenizer,
        device
    )

    # ------------------------------------------------
    # Calculate average sparsity
    # ------------------------------------------------

    key_sparsity_sum = 0.0
    value_sparsity_sum = 0.0

    key_layers_count = 0
    value_layers_count = 0

    for i in range(n_layer):

        key_counter = sparsity_information["key_counter"][i]
        value_counter = sparsity_information["value_counter"][i]

        if key_counter > 0:

            key_sparsity = (
                sparsity_information["key"][i] / key_counter
            )

            print(
                "Decoder", i,
                "Key sparsity:", key_sparsity
            )

            key_sparsity_sum += key_sparsity
            key_layers_count += 1

        if value_counter > 0:

            value_sparsity = (
                sparsity_information["value"][i] / value_counter
            )

            print(
                "Decoder", i,
                "Value sparsity:", value_sparsity
            )

            value_sparsity_sum += value_sparsity
            value_layers_count += 1

    # ------------------------------------------------
    # Print overall average sparsity
    # ------------------------------------------------

    if key_layers_count > 0:

        print(
            "Average Key Sparsity:",
            key_sparsity_sum / key_layers_count
        )

    if value_layers_count > 0:

        print(
            "Average Value Sparsity:",
            value_sparsity_sum / value_layers_count
        )
