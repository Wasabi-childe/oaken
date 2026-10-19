
from functools import partial
import json
from statistics import mean

from src.oaken.quantize import *
import pandas as pd



from functools import partial
import torch


def calibrate_k_channel_masks(args, model, tokenizer, device):
    import torch
    from datasets import load_dataset

    calibration = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )

    encodings = tokenizer(
        "\n\n".join(calibration["text"]),
        return_tensors="pt"
    )

    max_length = 2048
    seq_len = encodings.input_ids.size(1)
    n_layer = len(model.get_decoder().layers)

    score_sums = [None for _ in range(n_layer)]
    score_counts = [0 for _ in range(n_layer)]
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            x = output.detach().float()
            reduce_dims = tuple(range(x.ndim - 1))
            channel_sum = x.abs().sum(dim=reduce_dims)
            count = x.numel() // x.shape[-1]

            if score_sums[layer_idx] is None:
                score_sums[layer_idx] = channel_sum
            else:
                score_sums[layer_idx] += channel_sum

            score_counts[layer_idx] += count
            return output

        return hook

    for i, layer in enumerate(model.get_decoder().layers):
        handles.append(
            layer.self_attn.k_proj.register_forward_hook(
                make_hook(i)
            )
        )

    try:
        model.eval()

        with torch.no_grad():
            for start in range(0, seq_len, max_length):
                input_ids = encodings.input_ids[
                    :, start:start + max_length
                ].to(device)

                if input_ids.shape[1] == 0:
                    continue

                model(input_ids)

    finally:
        for handle in handles:
            handle.remove()

    channel_masks = []

    for i in range(n_layer):
        if score_sums[i] is None or score_counts[i] == 0:
            raise RuntimeError(
                "No calibration data collected for layer " + str(i)
            )

        scores = score_sums[i] / score_counts[i]
        n_channels = scores.numel()
        n_fp4 = max(1, int(n_channels * 0.25 + 0.5))

        selected = torch.topk(scores, k=n_fp4).indices

        mask = torch.zeros(
            n_channels, dtype=torch.bool, device=scores.device
        )
        mask[selected] = True
        channel_masks.append(mask)

        print(
            "Calibration layer", i,
            "| FP4 channels:", int(mask.sum().item()),
            "| FP2 channels:", int((~mask).sum().item())
        )

    return channel_masks


def multi_group_oaken_main(args, model, tokenizer, device, runner):
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_quant_group = quantizer_stat["n_quant_group"]
    n_layer = len(model.get_decoder().layers)

    # Phase 1: Calibrate K-channel masks using WikiText-2 train.
    channel_masks = calibrate_k_channel_masks(
        args, model, tokenizer, device
    )

    # Phase 2: Define quantization hooks.

    def tokenwise_quantize_activation_hook(
        i, module, inputs, output
    ):
        tensor = FP4E4M3Quantizer.quantize_dequantize(
            output, block_size=64
        )
        return tensor.to(output.dtype)

    def make_mixed_precision_k_hook(mask):
        def mixed_precision_k_hook(module, inputs, output):
            return MixedPrecisionKQuantizer.quantize_dequantize(
                output, mask
            ).to(output.dtype)

        return mixed_precision_k_hook

    # Phase 3: Apply quantization during evaluation.
    handles = []

    try:
        for layer_idx, decoder in enumerate(
            model.get_decoder().layers
        ):
            k_handle = decoder.self_attn.k_proj.register_forward_hook(
                make_mixed_precision_k_hook(
                    channel_masks[layer_idx]
                )
            )
            handles.append(k_handle)

            v_handle = decoder.self_attn.v_proj.register_forward_hook(
                partial(
                    tokenwise_quantize_activation_hook,
                    layer_idx
                )
            )
            handles.append(v_handle)

        # Uses the existing eval_perplexity runner.
        runner(args, model, tokenizer, device)

    finally:
        for handle in handles:
            handle.remove()

    return channel_masks

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
