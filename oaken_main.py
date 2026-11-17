
from functools import partial
import json

from src.oaken.quantize import *


def multi_group_oaken_main(args, model, tokenizer, device, runner):

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    if args.model == "gpt2":
        n_layer = len(model.transformer.h)
    else:
        n_layer = len(model.get_decoder().layers)

    # Only one outlier group is used
    sparsity_information = {
        "key": [[0.0] for _ in range(n_layer)],
        "value": [[0.0] for _ in range(n_layer)],
        "counter": [0.0 for _ in range(n_layer)]
    }

    # ------------------------------------------------
    # V PROJECTION: Direct FP4, without mean compensation
    # ------------------------------------------------

    def value_quantize_activation_hook(i, module, input, output):

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                use_mean_compensation=False
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

    # ------------------------------------------------
    # K PROJECTION: FP4 with mean compensation
    # ------------------------------------------------

    def key_quantize_activation_hook(i, module, input, output):

        # Use the first (outermost) threshold pair
        key_lower = quantizer_stat["key"]["lower_threshold"][i][:1]
        key_upper = quantizer_stat["key"]["upper_threshold"][i][:1]

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                key_lower,
                key_upper,
                use_mean_compensation=True
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

    runner(args, model, tokenizer, device)

    # ------------------------------------------------
    # Calculate average sparsity
    # ------------------------------------------------

    key_sparsity_sum = 0.0
    value_sparsity_sum = 0.0
    valid_layers = 0

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
        valid_layers += 1

    if valid_layers > 0:
        print(
            f"Total Sparsity: "
            f"Key - {key_sparsity_sum / valid_layers}, "
            f"Value - {value_sparsity_sum / valid_layers}"
        )
    else:
        print("No sparsity statistics were collected.")
