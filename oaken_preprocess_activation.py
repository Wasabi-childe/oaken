```python
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
        multi_key_value_tokenwise_main(args, eval_perplexity)
    else:
        multi_key_value_tokenwise_main(args, run_model)
```
