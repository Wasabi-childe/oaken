from functools import partial
import json
from statistics import mean

from src.oaken.quantize import *
import pandas as pd


def multi_group_oaken_main(args, model, tokenizer, device, runner):

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_layer = len(model.get_decoder().layers)

    # Only one outlier group is now used
    sparsity_information = {
        "key": [[0.0] for _ in range(n_layer)],
        "value": [[0.0] for _ in range(n_layer)],
        "counter": [0.0 for _ in range(n_layer)]
    }

    def value_quantize_activation_hook(i, module, input, output):

        # Use the first (outermost) threshold pair
        value_lower = quantizer_stat["value"]["lower_threshold"][i][:1]
        value_upper = quantizer_stat["value"]["upper_threshold"][i][:1]

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                value_lower,
                value_upper
            )
        )

        sparsity_information["value"][i] = [
            a + b for a, b in zip(
                sparsity_information["value"][i],
                sparsity
            )
        ]

        sparsity_information["counter"][i] += 0.5

        return tensor.half()

    def key_quantize_activation_hook(i, module, input, output):

        # Use the first (outermost) threshold pair
        key_lower = quantizer_stat["key"]["lower_threshold"][i][:1]
        key_upper = quantizer_stat["key"]["upper_threshold"][i][:1]

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                key_lower,
                key_upper
            )
        )

        sparsity_information["key"][i] = [
            a + b for a, b in zip(
                sparsity_information["key"][i],
                sparsity
            )
        ]

        sparsity_information["counter"][i] += 0.5

        return tensor.half()

    # Register hooks
    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.v_proj.register_forward_hook(
            partial(value_quantize_activation_hook, i)
        )

        decoder.self_attn.k_proj.register_forward_hook(
            partial(key_quantize_activation_hook, i)
        )

    # Run model
    runner(args, model, tokenizer, device)

    # Calculate average sparsity
    key_sparsity_sum = 0.0
    value_sparsity_sum = 0.0

    for i in range(n_layer):

        counter = sparsity_information["counter"][i]

        if counter == 0:
            continue

        key_sparsity = (
            sparsity_information["key"][i][0] / counter
        )

        value_sparsity = (
            sparsity_information["value"][i][0] / counter
        )

        print(
            f"Decoder {i} Sparsity: "
            f"Key - {key_sparsity}, "
            f"Value - {value_sparsity}"
        )

        key_sparsity_sum += key_sparsity
        value_sparsity_sum += value_sparsity

    print(
        f"Total Sparsity: "
        f"Key - {key_sparsity_sum / n_layer}, "
        f"Value - {value_sparsity_sum / n_layer}"
    )
