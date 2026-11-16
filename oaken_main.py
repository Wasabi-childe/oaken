from functools import partial
import json
from statistics import mean

from src.oaken.quantize import *
import pandas as pd


from functools import partial


def multi_group_oaken_main(args, model, tokenizer, device, runner):

    print("[INFO] Running Configuration C:")
    print("[INFO] K: Dynamic top-4 FP8 outliers + FP4")
    print("[INFO] V: FP4 only")
    print("[INFO] Group size: 128 (two attention heads)")

    num_heads = model.config.num_attention_heads
    n_layer = len(model.get_decoder().layers)
    num_heads = model.config.num_key_value_heads  # 4
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 64
    
    heads_per_group = 2
    group_size = heads_per_group * head_dim  # 128

    sparsity_information = {
        "key": [0.0 for _ in range(n_layer)],
        "value": [0.0 for _ in range(n_layer)],
        "counter": [0.0 for _ in range(n_layer)]
    }

    # ------------------------------------------------
    # K quantization hook
    # ------------------------------------------------

    def key_quantization_hook(i, module, input, output):

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample_config_c(
                output,
                num_heads=num_heads,
                is_key=True
            )
        )

        sparsity_information["key"][i] += sparsity[0]
        sparsity_information["counter"][i] += 0.5

        return tensor

    # ------------------------------------------------
    # V quantization hook
    # ------------------------------------------------

    def value_quantization_hook(i, module, input, output):

        tensor, sparsity, heatmap = (
            MultiThresholdTokenwiseQuantizer.downsample_config_c(
                output,
                num_heads=num_heads,
                is_key=False
            )
        )

        sparsity_information["value"][i] += sparsity[0]
        sparsity_information["counter"][i] += 0.5

        return tensor

    # ------------------------------------------------
    # Register hooks
    # ------------------------------------------------

    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.k_proj.register_forward_hook(
            partial(key_quantization_hook, i)
        )

        decoder.self_attn.v_proj.register_forward_hook(
            partial(value_quantization_hook, i)
        )

    # ------------------------------------------------
    # Run inference
    # ------------------------------------------------

    runner(args, model, tokenizer, device)

    # ------------------------------------------------
    # Print statistics
    # ------------------------------------------------

    key_sparsity = []
    value_sparsity = []

    for i in range(n_layer):

        count = sparsity_information["counter"][i]

        key_value = (
            sparsity_information["key"][i] / (count * 2)
            if count > 0 else 0.0
        )

        value_value = (
            sparsity_information["value"][i] / (count * 2)
            if count > 0 else 0.0
        )

        key_sparsity.append(key_value)
        value_sparsity.append(value_value)

        print(
            f"Decoder {i}: "
            f"Key outlier fraction = {key_value:.5f}, "
            f"Value outlier fraction = {value_value:.5f}"
        )

    print(
        f"Average K outlier fraction: {sum(key_sparsity) / n_layer:.5f}"
    )

    print(
        f"Average V outlier fraction: {sum(value_sparsity) / n_layer:.5f}"
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
            decoder.self_attn.v_proj.register_forward_hook(partial(tokenwise_quantize_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(channelwise_quantize_activation_hook, i))
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    runner(args, model, tokenizer, device)

    key_sparsity = []
    value_sparsity = []
    for i in range(len(model.get_decoder().layers)):
        key_sparsity.append(sparsity_information['key'][i] / sparsity_information['counter'][i])
        value_sparsity.append(sparsity_information['value'][i] / sparsity_information['counter'][i])
        print(f"Decoder {i} Sparsity: Key - {key_sparsity[i]}, Value - {value_sparsity[i]}")

    print(f"Total Sparsity: Key - {mean(key_sparsity)}, Value - {mean(value_sparsity)}")
