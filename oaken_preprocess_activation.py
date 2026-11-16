
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

    if len(args.list_fracs) != 1:
        raise ValueError(
            "Provide exactly one outlier fraction using -f. "
            "Example: -f 0.05"
        )

    outlier_frac = args.list_fracs[0]

    if not 0.0 < outlier_frac < 1.0:
        raise ValueError(
            "Outlier fraction must be between 0 and 1."
        )

    torch_device, tokenizer, model = common_main(args)

    n_layer = len(model.get_decoder().layers)

    start_time = time.monotonic()

    # Statistics collected independently for every layer.
    # K and V are kept separate.

    stat = {
        "key": [],
        "value": []
    }

    for _ in range(n_layer):
        stat["key"].append({
            "upper_threshold": 0.0,
            "lower_threshold": 0.0,
            "threshold_counter": 0,
            "outlier_abs_sum": 0.0,
            "outlier_count": 0
        })

        stat["value"].append({
            "upper_threshold": 0.0,
            "lower_threshold": 0.0,
            "threshold_counter": 0,
            "outlier_abs_sum": 0.0,
            "outlier_count": 0
        })

    sample_rate = args.sample_rate

    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(
            "Sample rate must be greater than 0 and at most 1."
        )

    # ------------------------------------------------
    # Shared function for K and V statistics
    # ------------------------------------------------

    def collect_activation_stats(output, layer_stat):

        if sample_rate != 1.0:
            n_tokens = max(
                1,
                int(sample_rate * output.size(1))
            )

            sampled_indices = torch.randperm(
                output.size(1),
                device=output.device
            )[:n_tokens]

            output_float32 = output[:, sampled_indices, :].float()

        else:
            output_float32 = output.float()

        # Calculate thresholds for this batch.

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

        # Accumulate thresholds.

        layer_stat["upper_threshold"] += upper.item()
        layer_stat["lower_threshold"] += lower.item()
        layer_stat["threshold_counter"] += 1

        # Identify outliers using this batch's thresholds.

        outlier_mask = torch.logical_or(
            output_float32 <= lower,
            output_float32 >= upper
        )

        # Accumulate absolute outlier values.
        # Only sum and count are retained, not the activations.

        outlier_abs_sum = (
            output_float32.abs() * outlier_mask
        ).sum()

        outlier_count = outlier_mask.sum()

        layer_stat["outlier_abs_sum"] += outlier_abs_sum.item()
        layer_stat["outlier_count"] += outlier_count.item()

        return output

    # ------------------------------------------------
    # K projection hook
    # ------------------------------------------------

    def get_key_activation_stat_hook(
        i, module, input, output
    ):

        return collect_activation_stats(
            output,
            stat["key"][i]
        )

    # ------------------------------------------------
    # V projection hook
    # ------------------------------------------------

    def get_value_activation_stat_hook(
        i, module, input, output
    ):

        return collect_activation_stats(
            output,
            stat["value"][i]
        )

    # ------------------------------------------------
    # Register hooks
    # ------------------------------------------------

    for i, decoder in enumerate(model.get_decoder().layers):

        decoder.self_attn.k_proj.register_forward_hook(
            partial(get_key_activation_stat_hook, i)
        )

        decoder.self_attn.v_proj.register_forward_hook(
            partial(get_value_activation_stat_hook, i)
        )

    # ------------------------------------------------
    # Run calibration
    # ------------------------------------------------

    runner(
        args,
        model,
        tokenizer,
        torch_device
    )

    # ------------------------------------------------
    # Calculate final statistics
    # ------------------------------------------------

    quantizer_stat = {
        "n_quant_group": 1,

        "key": {
            "upper_threshold": [],
            "lower_threshold": [],
            "mean_abs_outlier": []
        },

        "value": {
            "upper_threshold": [],
            "lower_threshold": [],
            "mean_abs_outlier": []
        }
    }

    for i in range(n_layer):

        for kv_type in ["key", "value"]:

            layer_stat = stat[kv_type][i]

            threshold_counter = layer_stat["threshold_counter"]
            outlier_count = layer_stat["outlier_count"]

            if threshold_counter == 0:
                raise RuntimeError(
                    "No activation statistics collected for "
                    + kv_type + " layer " + str(i)
                )

            if outlier_count == 0:
                raise RuntimeError(
                    "No outliers collected for "
                    + kv_type + " layer " + str(i)
                )

            # Average the batch thresholds.

            upper_threshold = (
                layer_stat["upper_threshold"] / threshold_counter
            )

            lower_threshold = (
                layer_stat["lower_threshold"] / threshold_counter
            )

            # One mean absolute outlier value per layer.

            mean_abs_outlier = (
                layer_stat["outlier_abs_sum"] / outlier_count
            )

            quantizer_stat[kv_type]["upper_threshold"].append(
                [upper_threshold]
            )

            quantizer_stat[kv_type]["lower_threshold"].append(
                [lower_threshold]
            )

            quantizer_stat[kv_type]["mean_abs_outlier"].append(
                [mean_abs_outlier]
            )

        print(
            "Decoder", i,
            "K thresholds:",
            quantizer_stat["key"]["lower_threshold"][i][0],
            quantizer_stat["key"]["upper_threshold"][i][0],
            "K mean:",
            quantizer_stat["key"]["mean_abs_outlier"][i][0],
            "| V thresholds:",
            quantizer_stat["value"]["lower_threshold"][i][0],
            quantizer_stat["value"]["upper_threshold"][i][0],
            "V mean:",
            quantizer_stat["value"]["mean_abs_outlier"][i][0]
        )

    end_time = time.monotonic()

    print(
        "Elapsed time:",
        end_time - start_time,
        "seconds"
    )

    # ------------------------------------------------
    # Save statistics
    # ------------------------------------------------

    with open(args.output_file, mode="w") as f:
        json.dump(
            quantizer_stat,
            f,
            indent=2
        )

    print("Quantizer statistics saved to:", args.output_file)


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
        help="One total outlier fraction, e.g. 0.05"
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
