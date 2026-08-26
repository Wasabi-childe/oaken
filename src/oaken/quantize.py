from functools import partial
import json
from statistics import mean

import torch
from src.oaken.quantize import *
import pandas as pd

def multi_group_oaken_main(args, model, tokenizer, device, runner):
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)
        n_quant_group = quantizer_stat["n_quant_group"]
        n_layer = len(model.get_decoder().layers)

        sparsity_information = {
            "key": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
            "value": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
            "counter": [0.0 for j in range(n_layer)]
        }

        # ---------------- NEW: accumulate every forward call's codes_info per layer (full KV cache, not just last token) ----------------
        kv_capture = {
            "key": [[] for _ in range(n_layer)],
            "value": [[] for _ in range(n_layer)],
        }

        key_counter = 0
        value_counter = 0

        def tokenwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity, heatmap, codes_info = MultiThresholdTokenwiseQuantizer.downsample_with_codes(
                output,
                quantizer_stat["value"]["lower_threshold"][i],
                quantizer_stat["value"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
            )
            sparsity_information["value"][i] = [sum(x) for x in zip(sparsity_information["value"][i], sparsity)]
            sparsity_information["counter"][i] += 0.5

            # ---------------- NEW: append (not overwrite) so every token's codes for this layer are kept ----------------
            kv_capture["value"][i].append(codes_info)

            return tensor.half()

        def channelwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity, heatmap, codes_info = MultiThresholdTokenwiseQuantizer.downsample_with_codes(
                output,
                quantizer_stat["key"]["lower_threshold"][i],
                quantizer_stat["key"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
            )
            sparsity_information["key"][i] = [sum(x) for x in zip(sparsity_information["key"][i], sparsity)]
            sparsity_information["counter"][i] += 0.5

            # ---------------- NEW: append (not overwrite) so every token's codes for this layer are kept ----------------
            kv_capture["key"][i].append(codes_info)

            return tensor.half()
    
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(partial(tokenwise_quantize_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(channelwise_quantize_activation_hook, i))
        
        runner(args, model, tokenizer, device)

        # ---------------- NEW: merge each layer's per-call codes_info entries into one combined tensor per group, ----------------
        # ---------------- concatenated along the sequence/token dimension, then save the full KV cache to a .pt file ----------------
        def merge_layer_codes(layer_calls):
            """
            layer_calls: list of codes_info dicts, one per forward call (one per token/chunk).
            Returns: dict of {group_label: merged_codes_info} where each group's "codes"
            (and "mask", if present) are concatenated along dim=-2 (the sequence dimension)
            across all calls, preserving one entry per group across the whole sequence.
            """
            if len(layer_calls) == 0:
                return {}

            merged = {}
            group_labels = layer_calls[0].keys()
            for label in group_labels:
                codes_list = [call[label]["codes"] for call in layer_calls]
                merged_codes = torch.cat(codes_list, dim=-2)

                entry = dict(layer_calls[0][label])  # copy static metadata (bits, qx, offset, etc.)
                entry["codes"] = merged_codes

                if "mask" in layer_calls[0][label]:
                    mask_list = [call[label]["mask"] for call in layer_calls]
                    entry["mask"] = torch.cat(mask_list, dim=-2)
                if "higher_mask" in layer_calls[0][label]:
                    entry["higher_mask"] = torch.cat([call[label]["higher_mask"] for call in layer_calls], dim=-2)
                if "lower_mask" in layer_calls[0][label]:
                    entry["lower_mask"] = torch.cat([call[label]["lower_mask"] for call in layer_calls], dim=-2)

                merged[label] = entry
            return merged

        full_kv_cache = {
            "key": [merge_layer_codes(kv_capture["key"][i]) for i in range(n_layer)],
            "value": [merge_layer_codes(kv_capture["value"][i]) for i in range(n_layer)],
        }

        kv_save_path = getattr(args, "kv_capture_path", "quantized_kv.pt")
        torch.save(full_kv_cache, kv_save_path)
        print(f"Saved FULL quantized KV cache (all layers, all tokens) to {kv_save_path}")

        key_sparsity = []
        value_sparsity = []
        key_sparsity_sum = [0.0 for _ in range(n_quant_group)]
        value_sparsity_sum = [0.0 for _ in range(n_quant_group)]
        for i in range(n_layer):
            key_sparsity.append(
                list(map(lambda x: x / sparsity_information['counter'][i], sparsity_information['key'][i]))
            )
            value_sparsity.append(
                list(map(lambda x: x / sparsity_information['counter'][i], sparsity_information['value'][i]))
            )
            print(f"Decoder {i} Sparsity: Key - {key_sparsity[i]}, Value - {value_sparsity[i]}")
            for idx, item in enumerate(key_sparsity[i]):
                key_sparsity_sum[idx] += item
            for idx, item in enumerate(value_sparsity[i]):
                value_sparsity_sum[idx] += item

        print(f"Total Sparsity: Key - {[x / n_layer for x in key_sparsity_sum]}, Value - {[x / n_layer for x in value_sparsity_sum]}")

def key_channelwise_value_tokenwise_main(args, model, tokenizer, device, runner):
    sparsity_information = {
        "key": [0.0 for _ in range(len(model.get_decoder().layers))],
        "value": [0.0 for _ in range(len(model.get_decoder().layers))],
        "counter": [0.0 for _ in range(len(model.get_decoder().layers))]
    }

    # ---------------- NEW: accumulate every forward call's tensor per layer (full KV cache, not just last token) ----------------
    n_layer = len(model.get_decoder().layers)
    kv_capture = {
        "key": [[] for _ in range(n_layer)],
        "value": [[] for _ in range(n_layer)],
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

            # ---------------- NEW: append, not overwrite ----------------
            # NOTE: TokenwiseQuantizer.downsample() doesn't return integer codes,
            # so this still captures dequantized fp16 (see earlier note).
            kv_capture["value"][i].append(tensor.detach().half().cpu())

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

            # ---------------- NEW: append, not overwrite ----------------
            kv_capture["key"][i].append(tensor.detach().half().cpu())

            return tensor.half()
    
    if args.model in ["opt", "llama"]:
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(partial(tokenwise_quantize_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(channelwise_quantize_activation_hook, i))
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    runner(args, model, tokenizer, device)

    # ---------------- NEW: concatenate each layer's per-call tensors along the sequence dim, then save the full KV cache ----------------
    full_kv_cache = {
        "key": [torch.cat(kv_capture["key"][i], dim=-2) if len(kv_capture["key"][i]) > 0 else None for i in range(n_layer)],
        "value": [torch.cat(kv_capture["value"][i], dim=-2) if len(kv_capture["value"][i]) > 0 else None for i in range(n_layer)],
    }

    kv_save_path = getattr(args, "kv_capture_path", "quantized_kv.pt")
    torch.save(full_kv_cache, kv_save_path)
    print(f"Saved FULL KV cache (all layers, all tokens) to {kv_save_path}")

    key_sparsity = []
    value_sparsity = []
    for i in range(len(model.get_decoder().layers)):
        key_sparsity.append(sparsity_information['key'][i] / sparsity_information['counter'][i])
        value_sparsity.append(sparsity_information['value'][i] / sparsity_information['counter'][i])
        print(f"Decoder {i} Sparsity: Key - {key_sparsity[i]}, Value - {value_sparsity[i]}")

    print(f"Total Sparsity: Key - {mean(key_sparsity)}, Value - {mean(value_sparsity)}")
