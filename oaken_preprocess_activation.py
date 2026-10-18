
import torch
import argparse
from functools import partial
import json
import time

from src.model import get_model
from src.tokenizer import get_tokenizer
from eval_perplexity import eval_perplexity
from eval_workload import run_model


def common_main(args):
    print("= Run Configuration ===================")
    print("Model name: {}".format(args.model))
    print("Model size: {}".format(args.model_size))
    print("Activation outlier list: {}".format(args.list_fracs))
    print("Task: {}".format(args.task))
    print("Batch size: {}".format(args.batch_size))
    print("Output: {}".format(args.output_file))
    print("=======================================")

    torch_device = "cuda"
    tokenizer = get_tokenizer(args.model, args.model_size)
    model = get_model(
        torch_device,
        tokenizer,
        args.model,
        args.model_size,
        args.gpu_count,
        args.gpu_start_idx
    )

    return torch_device, tokenizer, model


def multi_key_value_tokenwise_main(args, runner):
    print("Called K-only threshold profiling")

    torch_device, tokenizer, model = common_main(args)

    n_layer = len(model.get_decoder().layers)
    n_quant_group = len(args.list_fracs)

    assert n_quant_group > 1
    assert abs(sum(args.list_fracs) - 1.0) < 1e-6
    assert 0.0 < args.sample_rate <= 1.0

    start_time = time.monotonic()

    stat = {
        "key": [{
            "upper_threshold": [0.0 for _ in range(n_quant_group)],
            "lower_threshold": [0.0 for _ in range(n_quant_group)],
            "counter": 0,
        } for _ in range(n_layer)]
    }

    sample_rate = args.sample_rate

    def get_key_activation_stat_hook(i, module, inputs, output):
        if sample_rate != 1.0:
            n_tokens = output.size(1)
            n_sampled = max(1, int(sample_rate * n_tokens))
            sampled_indices = torch.randperm(
                n_tokens, device=output.device
            )[:n_sampled]

            output_float32 = output[:, sampled_indices, :].float()
        else:
            output_float32 = output.float()

        accumulated_frac = 0.0

        for idx, frac in enumerate(args.list_fracs):
            accumulated_frac += frac

            upper = torch.quantile(
                output_float32,
                1.0 - accumulated_frac / 2.0,
                dim=None,
                interpolation="higher"
            ).half().cpu().item()

            lower = torch.quantile(
                output_float32,
                accumulated_frac / 2.0,
                dim=None,
                interpolation="lower"
            ).half().cpu().item()

            stat["key"][i]["upper_threshold"][idx] += upper
            stat["key"][i]["lower_threshold"][idx] += lower

        stat["key"][i]["counter"] += 1

        return output

    # Register hooks only on K projections.
    for i, decoder in enumerate(model.get_decoder().layers):
        decoder.self_attn.k_proj.register_forward_hook(
            partial(get_key_activation_stat_hook, i)
        )

    # Run profiling inference.
    runner(args, model, tokenizer, torch_device)

    # Average the K thresholds across hook invocations.
    quantizer_stat = {
        "n_quant_group": n_quant_group,
        "key": {
            "upper_threshold": [
                [0.0 for _ in range(n_quant_group)]
                for _ in range(n_layer)
            ],
            "lower_threshold": [
                [0.0 for _ in range(n_quant_group)]
                for _ in range(n_layer)
            ]
        }
    }

    for i in range(n_layer):
        counter = stat["key"][i]["counter"]

        if counter == 0:
            raise RuntimeError(
                "No K-projection activations collected for layer {}".format(i)
            )

        for j in range(n_quant_group):
            quantizer_stat["key"]["upper_threshold"][i][j] = (
                stat["key"][i]["upper_threshold"][j] / counter
            )

            quantizer_stat["key"]["lower_threshold"][i][j] = (
                stat["key"][i]["lower_threshold"][j] / counter
            )

    end_time = time.monotonic()

    print("Elapsed time: {} seconds".format(end_time - start_time))

    # Save only K thresholds.
    with open(args.output_file, mode="w") as f:
        json.dump(quantizer_stat, f, indent=2)

    print("K-only thresholds saved to {}".format(args.output_file))
def fit_v_codebook_main(args, runner):
    print("Called V-only 3-bit codebook calibration")

    torch_device, tokenizer, model = common_main(args)

    n_layer = len(model.get_decoder().layers)

    CODEBOOK_SIZE = 8
    BLOCK_SIZE = 32
    MAX_VALUES_PER_LAYER = 100000

    samples_by_layer = [[] for _ in range(n_layer)]
    sample_counts = [0 for _ in range(n_layer)]

    def collect_v_samples(i, module, inputs, output):
        if sample_counts[i] >= MAX_VALUES_PER_LAYER:
            return output

        # Match the normalization used by FP3E4M3Quantizer.
        x = output.detach().reshape(-1, output.shape[-1]).float()

        width = x.shape[-1]
        pad = (-width) % BLOCK_SIZE

        if pad:
            x = torch.nn.functional.pad(x, (0, pad))

        blocks = x.reshape(x.shape[0], -1, BLOCK_SIZE)

        amax = blocks.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 2.0).clamp(
            min=2.0 ** -9,
            max=448.0
        )

        # Match the E4M3 scale rounding used by the quantizer.
        scale = scale.to(torch.float8_e4m3fn).float()
        normalized = blocks / scale

        values = normalized.reshape(-1)

        remaining = MAX_VALUES_PER_LAYER - sample_counts[i]
        n_take = min(remaining, 2048, values.numel())

        if n_take > 0:
            indices = torch.randperm(
                values.numel(),
                device=values.device
            )[:n_take]

            samples_by_layer[i].append(
                values[indices].cpu()
            )
            sample_counts[i] += n_take

        return output

    # Collect samples only from V projections.
    for i, decoder in enumerate(model.get_decoder().layers):
        decoder.self_attn.v_proj.register_forward_hook(
            partial(collect_v_samples, i)
        )

    # Run the existing calibration workload.
    runner(args, model, tokenizer, torch_device)

    collected = [
        torch.cat(layer_samples)
        for layer_samples in samples_by_layer
        if len(layer_samples) > 0
    ]

    if not collected:
        raise RuntimeError(
            "No V-projection samples collected."
        )

    all_samples = torch.cat(collected).float()

    print("Total normalized V samples:", all_samples.numel())

    # Lloyd-Max-style 1-D clustering into eight reconstruction levels.
    levels = torch.quantile(
        all_samples,
        torch.linspace(
            0.5 / CODEBOOK_SIZE,
            1.0 - 0.5 / CODEBOOK_SIZE,
            CODEBOOK_SIZE
        )
    )

    for iteration in range(100):
        boundaries = (levels[:-1] + levels[1:]) / 2.0

        assignments = torch.bucketize(
            all_samples,
            boundaries
        )

        new_levels = levels.clone()

        for j in range(CODEBOOK_SIZE):
            assigned = all_samples[assignments == j]

            if assigned.numel() > 0:
                new_levels[j] = assigned.mean()

        if torch.max(torch.abs(new_levels - levels)) < 1e-6:
            levels = new_levels
            break

        levels = new_levels

    levels = torch.sort(levels).values

    codebook = {
        "format": "data_fitted_v_codebook",
        "method": "lloyd_max_1d",
        "codebook_size": CODEBOOK_SIZE,
        "block_size": BLOCK_SIZE,
        "scale_format": "E4M3",
        "normalization": "block_amax_div_2",
        "scope": "shared_across_v_layers",
        "levels": levels.tolist(),
        "samples_per_layer": sample_counts
    }

    with open(args.output_file, "w") as f:
        json.dump(codebook, f, indent=2)

    print("\nFitted V codebook:")
    for i, level in enumerate(levels.tolist()):
        print("Index {}: {:.6f}".format(i, level))

    print("\nSaved codebook to {}".format(args.output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile K-projection activation thresholds only"
    )

    parser.add_argument(
        "-t", "--task",
        type=str, required=True, dest="task"
    )

    parser.add_argument(
        "-m", "--model",
        type=str, required=True, dest="model"
    )

    parser.add_argument(
        "-s", "--size",
        type=str, required=True, dest="model_size"
    )

    parser.add_argument(
        "-o", "--output",
        type=str, required=True, dest="output_file"
    )

    parser.add_argument(
        "-f", "--list_fracs",
        nargs="+", type=float, required=True, dest="list_fracs"
    )

    parser.add_argument(
        "-b", "--batch-size",
        default=1, type=int, dest="batch_size"
    )

    parser.add_argument(
        "--sample-rate",
        default=1.0, type=float, dest="sample_rate"
    )

    parser.add_argument(
        "--gpu-start-idx",
        default=0, type=int, dest="gpu_start_idx"
    )

    parser.add_argument(
        "--gpu-count",
        default=1, type=int, dest="gpu_count"
    )

    args = parser.parse_args()

    if args.task == "wikitext":
        fit_v_codebook_main(args, eval_perplexity)
    else:
        fit_v_codebook_main(args, run_model)
