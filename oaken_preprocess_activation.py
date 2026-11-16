
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
    print("Model name:", args.model)
    print("Model size:", args.model_size)
    print("Outlier fraction:", args.list_fracs[0])
    print("Task:", args.task)
    print("Batch size:", args.batch_size)
    print("Output:", args.output_file)
    print("=======================================")

    torch_device = "cuda"

    tokenizer = get_tokenizer(
        args.model,
        args.model_size
    )

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

    print("Called multi_key_value_tokenwise_main")

    # This algorithm uses exactly one outlier fraction
    if len(args.list_fracs) != 1:
        raise ValueError(
            "Provide exactly one outlier fraction using -f. "
            "Example: -f 0.02"
        )

    outlier_frac = args.list_fracs[0]

    if not 0.0 < outlier_frac < 1.0:
        raise ValueError(
            "Outlier fraction must be between 0 and 1."
        )

    torch_device, tokenizer, model = common_main(args)

    n_layer = len(model.get_decoder().layers)

    start_time = time.monotonic()

    # One lower and upper threshold per layer
    stat = {
        "key": [
            {
                "upper_threshold": 0.0,
                "lower_threshold": 0.0,
                "counter": 0
            }
            for _ in range(n_layer)
        ],

        "value": [
            {
                "upper_threshold": 0.0,
                "lower_threshold": 0.0,
                "counter": 0
            }
            for _ in range(n_layer)
        ]
    }

    sample_rate = args.sample_rate

    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(
            "Sample rate must be greater than 0 and at most 1."
        )

    # ---------------------------------------------
    # V projection threshold profiling
    # ---------------------------------------------

    def get_value_activation_stat_hook(
        i, module, input, output
    ):

        if sample_rate != 1.0:
            sampled_indices = torch.randperm(
                output.size(1),
                device=output.device
            )[:int(sample_rate * output.size(1))]

            output_float32 = output[:, sampled_indices, :].float()

        else:
            output_float32 = output.float()

        # Upper and lower thresholds
        upper = torch.quantile(
            output_float32,
            1.0 - outlier_frac / 2.0,
            dim=None,
            interpolation="higher"
        )

        lower = torch.quantile(
            output_float32,
            outlier_frac / 2.0,
            dim=None,
            interpolation="lower"
        )

        stat["value"][i]["upper_threshold"] += upper.half().cpu().item()
        stat["value"][i]["lower_threshold"] += lower.half().cpu().item()

        stat["value"][i]["counter"] += 1

        return output

    # ---------------------------------------------
    # K projection threshold profiling
    # ---------------------------------------------

    def get_key_activation_stat_hook(
        i, module, input, output
    ):

        if sample_rate != 1.0:
            sampled_indices = torch.randperm(
                output.size(1),
                device=output.device
            )[:int(sample_rate * output.size(1))]

            output_float32 = output[:, sampled_indices, :].float()

        else:
            output_float32 = output.float()

        # Upper and lower thresholds
        upper = torch.quantile(
            output_float32,
            1.0 - outlier_frac / 2.0,
            dim=None,
            interpolation="higher"
        )

        lower = torch.quantile(
            output_float32,
            outlier_frac / 2.0,
            dim=None,
            interpolation="lower"
        )

        stat["key"][i]["upper_threshold"] += upper.half().cpu().item()
        stat["key"][i]["lower_threshold"] += lower.half().cpu().item()

        stat["key"][i]["counter"] += 1

        return output

    # ---------------------------------------------
    # Register hooks
    # ---------------------------------------------

    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.k_proj.register_forward_hook(
            partial(get_key_activation_stat_hook, i)
        )

        decoder.self_attn.v_proj.register_forward_hook(
            partial(get_value_activation_stat_hook, i)
        )

    # ---------------------------------------------
    # Run inference for profiling
    # ---------------------------------------------

    runner(
        args,
        model,
        tokenizer,
        torch_device
    )

    # ---------------------------------------------
    # Average thresholds
    # ---------------------------------------------

    quantizer_stat = {
        "n_quant_group": 1,

        "key": {
            "upper_threshold": [],
            "lower_threshold": []
        },

        "value": {
            "upper_threshold": [],
            "lower_threshold": []
        }
    }

    for i in range(n_layer):

        key_counter = stat["key"][i]["counter"]
        value_counter = stat["value"][i]["counter"]

        if key_counter == 0 or value_counter == 0:
            raise RuntimeError(
                "No activation statistics collected for layer "
                + str(i)
            )

        key_upper = (
            stat["key"][i]["upper_threshold"] / key_counter
        )

        key_lower = (
            stat["key"][i]["lower_threshold"] / key_counter
        )

        value_upper = (
            stat["value"][i]["upper_threshold"] / value_counter
        )

        value_lower = (
            stat["value"][i]["lower_threshold"] / value_counter
        )

        quantizer_stat["key"]["upper_threshold"].append(
            [key_upper]
        )

        quantizer_stat["key"]["lower_threshold"].append(
            [key_lower]
        )

        quantizer_stat["value"]["upper_threshold"].append(
            [value_upper]
        )

        quantizer_stat["value"]["lower_threshold"].append(
            [value_lower]
        )

        print(
            "Decoder", i,
            "K thresholds:", key_lower, key_upper,
            "V thresholds:", value_lower, value_upper
        )

    end_time = time.monotonic()

    print(
        "Elapsed time:",
        end_time - start_time,
        "seconds"
    )

    # ---------------------------------------------
    # Save threshold statistics
    # ---------------------------------------------

    with open(args.output_file, mode="w") as f:
        json.dump(
            quantizer_stat,
            f,
            indent=2
        )

    print("Threshold statistics saved to:", args.output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the profiling of the model"
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        dest="task"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        dest="model"
    )

    parser.add_argument(
        "-s", "--size",
        type=str,
        required=True,
        dest="model_size"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        dest="output_file"
    )

    parser.add_argument(
        "-f", "--list_fracs",
        nargs="+",
        type=float,
        required=True,
        dest="list_fracs",
        help="One total outlier fraction, e.g. 0.02"
    )

    parser.add_argument(
        "-b", "--batch-size",
        default=1,
        type=int,
        dest="batch_size"
    )

    parser.add_argument(
        "--sample-rate",
        default=1.0,
        type=float,
        dest="sample_rate"
    )

    parser.add_argument(
        "--gpu-start-idx",
        default=0,
        type=int,
        dest="gpu_start_idx"
    )

    parser.add_argument(
        "--gpu-count",
        default=1,
        type=int,
        dest="gpu_count"
    )

    args = parser.parse_args()

    if args.task == "wikitext":
        multi_key_value_tokenwise_main(
            args,
            eval_perplexity
        )
    else:
        multi_key_value_tokenwise_main(
            args,
            run_model
        )
