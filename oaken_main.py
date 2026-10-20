from functools import partial
import json
from statistics import mean

from src.oaken.quantize import MultiThresholdTokenwiseQuantizer


def multi_group_oaken_main(args, model, tokenizer, device, runner):
    # ------------------------------------------------------------
    # Load calibrated thresholds and learned codebooks
    # ------------------------------------------------------------
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_quant_group = quantizer_stat["n_quant_group"]
    n_layer = len(model.get_decoder().layers)

    if n_quant_group != 3:
        raise ValueError(
            "Expected three groups: outer, middle, and inner."
        )

    if "codebooks" not in quantizer_stat:
        raise ValueError(
            "No learned codebooks found in the calibration file. "
            "Run the calibration script first."
        )

    codebooks = quantizer_stat["codebooks_q3_2"]

    required_codebooks = [
        "K_outer",
        "K_middle",
        "K_inner",
        "V_outer",
        "V_middle",
        "V_inner",
    ]

    for name in required_codebooks:
        if name not in codebooks:
            raise ValueError("Missing codebook: " + name)

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------
    sparsity_information = {
        "key": [
            [0.0 for _ in range(n_quant_group)]
            for _ in range(n_layer)
        ],
        "value": [
            [0.0 for _ in range(n_quant_group)]
            for _ in range(n_layer)
        ],
        "counter": [0.0 for _ in range(n_layer)],
    }

    handles = []

    # ------------------------------------------------------------
    # V projection: use V-specific learned codebooks
    # ------------------------------------------------------------
    def tokenwise_quantize_activation_hook(
        i, module, inputs, output
    ):
        tensor, sparsity, _ = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                quantizer_stat["value"]["lower_threshold"][i],
                quantizer_stat["value"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
                codebooks=codebooks,
                tensor_kind="value",
                layer_idx=i,
            )
        )

        sparsity_information["value"][i] = [
            a + b
            for a, b in zip(
                sparsity_information["value"][i],
                sparsity,
            )
        ]

        sparsity_information["counter"][i] += 0.5

        return tensor.half()

    # ------------------------------------------------------------
    # K projection: use K-specific learned codebooks
    # ------------------------------------------------------------
    def channelwise_quantize_activation_hook(
        i, module, inputs, output
    ):
        tensor, sparsity, _ = (
            MultiThresholdTokenwiseQuantizer.downsample(
                output,
                quantizer_stat["key"]["lower_threshold"][i],
                quantizer_stat["key"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
                codebooks=codebooks,
                tensor_kind="key",
                layer_idx=i,
            )
        )

        sparsity_information["key"][i] = [
            a + b
            for a, b in zip(
                sparsity_information["key"][i],
                sparsity,
            )
        ]

        sparsity_information["counter"][i] += 0.5

        return tensor.half()

    # ------------------------------------------------------------
    # Register hooks
    # ------------------------------------------------------------
    for i, decoder in enumerate(model.get_decoder().layers):
        handles.append(
            decoder.self_attn.v_proj.register_forward_hook(
                partial(tokenwise_quantize_activation_hook, i)
            )
        )

        handles.append(
            decoder.self_attn.k_proj.register_forward_hook(
                partial(channelwise_quantize_activation_hook, i)
            )
        )

    # ------------------------------------------------------------
    # Run evaluation and remove hooks afterwards
    # ------------------------------------------------------------
    try:
        runner(args, model, tokenizer, device)
    finally:
        for handle in handles:
            handle.remove()

    # ------------------------------------------------------------
    # Report per-layer and overall group fractions
    # ------------------------------------------------------------
    key_sparsity = []
    value_sparsity = []

    key_sparsity_sum = [0.0 for _ in range(n_quant_group)]
    value_sparsity_sum = [0.0 for _ in range(n_quant_group)]

    for i in range(n_layer):
        count = sparsity_information["counter"][i]

        if count == 0:
            print(
                "Warning: no K/V projection activations captured "
                "for decoder", i
            )
            continue

        key_layer = [
            x / count
            for x in sparsity_information["key"][i]
        ]

        value_layer = [
            x / count
            for x in sparsity_information["value"][i]
        ]

        key_sparsity.append(key_layer)
        value_sparsity.append(value_layer)

        print(
            "Decoder", i,
            "Sparsity: Key -", key_layer,
            "Value -", value_layer,
        )

        for idx in range(n_quant_group):
            key_sparsity_sum[idx] += key_layer[idx]
            value_sparsity_sum[idx] += value_layer[idx]

    n_valid_layers = len(key_sparsity)

    if n_valid_layers:
        print(
            "Total Sparsity: Key -",
            [x / n_valid_layers for x in key_sparsity_sum],
            "Value -",
            [x / n_valid_layers for x in value_sparsity_sum],
        )


def key_channelwise_value_tokenwise_main(
    args, model, tokenizer, device, runner
):
    # Both K and V use the same learned, multi-group quantizer.
    return multi_group_oaken_main(
        args, model, tokenizer, device, runner
    )

def key_channelwise_value_tokenwise_main(args, model, tokenizer, device, runner):
    sparsity_information = {
        "key": [0.0 for _ in range(len(model.get_decoder().layers))],
        "value": [0.0 for _ in range(len(model.get_decoder().layers))],
        "counter": [0.0 for _ in range(len(model.get_decoder().layers))]
    }

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

        def tokenwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity = TokenwiseQuantizer.downsample(
                output,
                quantizer_stat["value"]["lower_threshold"][i],
                quantizer_stat["value"]["upper_threshold"][i],
                args.quant_outlier,
            )
            sparsity_information["value"][i] += sparsity
            sparsity_information["counter"][i] += 0.5
            return tensor.half()

        def channelwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity = ChannelwiseQuantizer.downsample(
                output,
                quantizer_stat["key"]["minval"][i],
                quantizer_stat["key"]["maxval"][i],
                quantizer_stat["key"]["lower_threshold"][i],
                quantizer_stat["key"]["upper_threshold"][i],
                args.quant_outlier,
            )
            sparsity_information["key"][i] += sparsity
            sparsity_information["counter"][i] += 0.5
            return tensor.half()

    if args.model in ["opt", "llama"]:
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(
                partial(tokenwise_quantize_activation_hook, i)
            )
            decoder.self_attn.k_proj.register_forward_hook(
                partial(channelwise_quantize_activation_hook, i)
            )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    runner(args, model, tokenizer, device)

    key_sparsity = []
    value_sparsity = []

    for i in range(len(model.get_decoder().layers)):
        key_sparsity.append(
            sparsity_information["key"][i]
            / sparsity_information["counter"][i]
        )
        value_sparsity.append(
            sparsity_information["value"][i]
            / sparsity_information["counter"][i]
        )
        print(
            f"Decoder {i} Sparsity: "
            f"Key - {key_sparsity[i]}, Value - {value_sparsity[i]}"
        )

    print(
        f"Total Sparsity: Key - {mean(key_sparsity)}, "
        f"Value - {mean(value_sparsity)}"
    )
