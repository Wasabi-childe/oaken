from functools import partial
import json
from statistics import mean

from src.oaken.quantize import *
import pandas as pd

from functools import partial
import json
from statistics import mean
import torch

from src.oaken.quantize import *


def multi_group_oaken_main(args, model, tokenizer, device, runner):

    # ============================================================
    # LOAD QUANTIZER STATISTICS
    # ============================================================

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_quant_group = quantizer_stat["n_quant_group"]
    n_layer = len(model.get_decoder().layers)

    sparsity_information = {
        "key": [
            [0.0 for _ in range(n_quant_group)]
            for _ in range(n_layer)
        ],
        "value": [
            [0.0 for _ in range(n_quant_group)]
            for _ in range(n_layer)
        ],
        "counter": [0.0 for _ in range(n_layer)]
    }

    # ============================================================
    # STORAGE FOR QUANTIZED INDICES
    # ============================================================

    # Each layer will contain a list of tensors.
    #
    # key_indices[layer]   -> quantized K values
    # value_indices[layer] -> quantized V values
    #
    # The tensors contain integer quantization indices:
    #   4-bit -> 0 ... 15
    #   5-bit -> 0 ... 31

    key_indices = [[] for _ in range(n_layer)]
    value_indices = [[] for _ in range(n_layer)]

    # ============================================================
    # VALUE HOOK
    # ============================================================

    def tokenwise_quantize_activation_hook(i, module, input, output):

        quantized_tensor, val_frac, _, quant_indices = \
            MultiThresholdTokenwiseQuantizer.downsample(
                input_tensor=output,
                threshold_lowers=quantizer_stat["value"]["lower_threshold"][i],
                threshold_uppers=quantizer_stat["value"]["upper_threshold"][i],
                quantize_outlier=True,
                use_group_shift=True
            )

        # --------------------------------------------------------
        # Store quantization indices
        # --------------------------------------------------------

        value_indices[i].append(
            quant_indices.detach().cpu()
        )

        # --------------------------------------------------------
        # Update sparsity statistics
        # --------------------------------------------------------

        sparsity_information["value"][i] = [
            a + b
            for a, b in zip(
                sparsity_information["value"][i],
                val_frac
            )
        ]

        sparsity_information["counter"][i] += 0.5

        # --------------------------------------------------------
        # Return reconstructed FP16 tensor to the model
        # --------------------------------------------------------

        return quantized_tensor.half()

    # ============================================================
    # KEY HOOK
    # ============================================================

    def channelwise_quantize_activation_hook(i, module, input, output):

        quantized_tensor, val_frac, _, quant_indices = \
            MultiThresholdTokenwiseQuantizer.downsample(
                input_tensor=output,
                threshold_lowers=quantizer_stat["key"]["lower_threshold"][i],
                threshold_uppers=quantizer_stat["key"]["upper_threshold"][i],
                quantize_outlier=True,
                use_group_shift=True
            )

        # --------------------------------------------------------
        # Store quantization indices
        # --------------------------------------------------------

        key_indices[i].append(
            quant_indices.detach().cpu()
        )

        # --------------------------------------------------------
        # Update sparsity statistics
        # --------------------------------------------------------

        sparsity_information["key"][i] = [
            a + b
            for a, b in zip(
                sparsity_information["key"][i],
                val_frac
            )
        ]

        sparsity_information["counter"][i] += 0.5

        # --------------------------------------------------------
        # Return reconstructed FP16 tensor to the model
        # --------------------------------------------------------

        return quantized_tensor.half()

    # ============================================================
    # REGISTER HOOKS
    # ============================================================

    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.v_proj.register_forward_hook(
            partial(
                tokenwise_quantize_activation_hook,
                i
            )
        )

        decoder.self_attn.k_proj.register_forward_hook(
            partial(
                channelwise_quantize_activation_hook,
                i
            )
        )

    # ============================================================
    # RUN MODEL
    # ============================================================

    runner(
        args,
        model,
        tokenizer,
        device
    )

    # ============================================================
    # COMBINE STORED QUANTIZED INDICES
    # ============================================================

    saved_key_indices = {}
    saved_value_indices = {}

    for i in range(n_layer):

        if len(key_indices[i]) > 0:
            saved_key_indices[f"layer_{i}_key"] = torch.cat(
                key_indices[i],
                dim=0
            )

        if len(value_indices[i]) > 0:
            saved_value_indices[f"layer_{i}_value"] = torch.cat(
                value_indices[i],
                dim=0
            )

    # ============================================================
    # SAVE QUANTIZED INDICES
    # ============================================================

    quantized_data = {}

    quantized_data.update(saved_key_indices)
    quantized_data.update(saved_value_indices)

    output_file = "/content/oaken/quantized_kv_indices.pt"

    torch.save(
        quantized_data,
        output_file
    )

    print()
    print("=" * 60)
    print("QUANTIZED INDICES SAVED")
    print("=" * 60)
    print(f"File: {output_file}")

    for name, tensor in quantized_data.items():

        print(
            f"{name}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"values={tensor.numel()}"
        )

    # ============================================================
    # SPARSITY RESULTS
    # ============================================================

    key_sparsity = []
    value_sparsity = []

    key_sparsity_sum = [
        0.0 for _ in range(n_quant_group)
    ]

    value_sparsity_sum = [
        0.0 for _ in range(n_quant_group)
    ]

    for i in range(n_layer):

        key_sparsity.append(
            [
                x / sparsity_information["counter"][i]
                for x in sparsity_information["key"][i]
            ]
        )

        value_sparsity.append(
            [
                x / sparsity_information["counter"][i]
                for x in sparsity_information["value"][i]
            ]
        )

        print(
            f"Decoder {i} Sparsity: "
            f"Key - {key_sparsity[i]}, "
            f"Value - {value_sparsity[i]}"
        )

        for idx, item in enumerate(key_sparsity[i]):
            key_sparsity_sum[idx] += item

        for idx, item in enumerate(value_sparsity[i]):
            value_sparsity_sum[idx] += item

    print(
        f"Total Sparsity: "
        f"Key - {[x / n_layer for x in key_sparsity_sum]}, "
        f"Value - {[x / n_layer for x in value_sparsity_sum]}"
    )

    return quantized_data

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

