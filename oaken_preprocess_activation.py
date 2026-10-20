
import torch
import argparse
import json
import time

from src.model import get_model
from src.tokenizer import get_tokenizer
from eval_perplexity import eval_perplexity
from eval_workload import run_model

import matplotlib.pyplot as plt
import torch


def plot_shifted_normalized_distributions(
    model,
    tokenizer,
    args,
    torch_device,
    runner,
    quantizer_stat,
    output_dir="quantizer/oaken/group_distributions",
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    distributions = {
        "K_outer": [],
        "K_middle": [],
        "K_inner": [],
        "V_outer": [],
        "V_middle": [],
        "V_inner": [],
    }

    handles = []

    def make_hook(layer_idx, tensor_kind):
        def hook(module, inputs, output):
            x = output.detach().float()

            lower = quantizer_stat[tensor_kind][
                "lower_threshold"
            ][layer_idx]

            upper = quantizer_stat[tensor_kind][
                "upper_threshold"
            ][layer_idx]

            rows = x.reshape(-1, x.shape[-1])

            masks = get_group_masks(rows, lower, upper)

            for group_idx, mask in enumerate(masks):
                if group_idx == 0:
                    group_name = "outer"
                elif group_idx == len(masks) - 1:
                    group_name = "inner"
                else:
                    group_name = "middle"

                shifted = shift_group(
                    rows,
                    mask,
                    group_idx,
                    lower,
                    upper,
                )

                scale, amax = tokenwise_e4m3_scale(
                    shifted, mask
                )

                normalized = shifted / scale

                valid = mask & (amax > 0)
                values = normalized[valid]

                if values.numel():
                    # Subsample to limit memory usage.
                    if values.numel() > 100000:
                        indices = torch.randperm(
                            values.numel(),
                            device=values.device,
                        )[:100000]
                        values = values[indices]

                    distributions[
                        ("K_" if tensor_kind == "key" else "V_")
                        + group_name
                    ].append(values.cpu())

            return output

        return hook

    for layer_idx, decoder in enumerate(
        model.get_decoder().layers
    ):
        handles.append(
            decoder.self_attn.k_proj.register_forward_hook(
                make_hook(layer_idx, "key")
            )
        )
        handles.append(
            decoder.self_attn.v_proj.register_forward_hook(
                make_hook(layer_idx, "value")
            )
        )

    try:
        runner(args, model, tokenizer, torch_device)
    finally:
        for handle in handles:
            handle.remove()

    # One histogram per group, with K and V plotted separately.
    for tensor_name in ("K", "V"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, group_name in zip(
            axes, ("outer", "middle", "inner")
        ):
            key = tensor_name + "_" + group_name
            chunks = distributions[key]

            if not chunks:
                ax.set_title(group_name + " (no samples)")
                continue

            values = torch.cat(chunks).numpy()

            ax.hist(
                values,
                bins=200,
                range=(-7, 7),
                density=True,
            )

            ax.set_title(
                tensor_name + " " + group_name
            )
            ax.set_xlabel("Shifted value / E4M3 scale")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            tensor_name + " distributions after group shift and scaling"
        )
        fig.tight_layout()

        path = os.path.join(
            output_dir,
            tensor_name + "_shifted_normalized.png",
        )
        fig.savefig(path, dpi=200)
        plt.show()
        plt.close(fig)

        print("Saved:", path)
# ================================================================
# CONFIGURATION
# ================================================================

HIST_BINS = 4096
MAX_NORMALIZED_VALUE = 6.0
LLOYD_ITERATIONS = 50

CODEBOOK_CONFIG = {
    "K_inner": {
        "tensor": "key",
        "group": "inner",
        "bits": 5,
    },
    "K_outer": {
        "tensor": "key",
        "group": "outer",
        "bits": 5,
    },
    "K_middle": {
        "tensor": "key",
        "group": "middle",
        "bits": 4,
    },
    "V_inner": {
        "tensor": "value",
        "group": "inner",
        "bits": 5,
    },
    "V_outer": {
        "tensor": "value",
        "group": "outer",
        "bits": 5,
    },
    "V_middle": {
        "tensor": "value",
        "group": "middle",
        "bits": 4,
    },
}


# ================================================================
# MODEL SETUP
# ================================================================

def common_main(args):
    print("= Run Configuration ===================")
    print("Model name:", args.model)
    print("Model size:", args.model_size)
    print("Activation fractions:", args.list_fracs)
    print("Task:", args.task)
    print("Batch size:", args.batch_size)
    print("Output:", args.output_file)
    print("=======================================")

    torch_device = "cuda"
    tokenizer = get_tokenizer(args.model, args.model_size)

    model = get_model(
        torch_device,
        tokenizer,
        args.model,
        args.model_size,
        args.gpu_count,
        args.gpu_start_idx,
    )

    return torch_device, tokenizer, model


# ================================================================
# OAKEN GROUP MASKS
#
# This reproduces the group-mask ordering in the supplied
# MultiThresholdTokenwiseQuantizer.get_multigroup_threshold().
#
# For three groups:
#   index 0 = outer
#   index 1 = middle
#   index 2 = inner
# ================================================================

def get_group_masks(x, threshold_lowers, threshold_uppers):
    if len(threshold_lowers) != len(threshold_uppers):
        raise ValueError("Lower and upper threshold counts differ.")

    masks = []
    previous_lower = None
    previous_upper = None

    for idx, (lower, upper) in enumerate(
        zip(threshold_lowers, threshold_uppers)
    ):
        if idx == len(threshold_lowers) - 1:
            # Innermost group
            mask = (
                (x > previous_lower)
                & (x < previous_upper)
            )

        elif previous_lower is not None:
            # Intermediate group
            mask = (
                (
                    (previous_lower < x)
                    & (x <= lower)
                )
                |
                (
                    (upper <= x)
                    & (x < previous_upper)
                )
            )

        else:
            # Outermost group
            mask = (
                (x <= lower)
                | (upper <= x)
            )

        masks.append(mask)

        previous_lower = lower
        previous_upper = upper

    return masks


# ================================================================
# TOKEN-WISE SCALE
#
# One E4M3 scale for each token and selected group.
# Scale convention matches the supplied FP4E4M3Quantizer:
#
#     scale = amax / 6
#
# The scale is rounded to E4M3 before being used.
# ================================================================

def tokenwise_e4m3_scale(x, mask):
    selected = x * mask

    amax = selected.abs().amax(
        dim=-1,
        keepdim=True,
    )

    scale = amax / MAX_NORMALIZED_VALUE

    # Handle empty/all-zero groups.
    scale = torch.where(
        amax > 0,
        scale,
        torch.ones_like(scale),
    )

    scale = scale.to(torch.float8_e4m3fn).float()
    scale = scale.clamp_min(1e-8)

    return scale, amax


# ================================================================
# APPLY OAKEN GROUP SHIFT
#
# In the supplied implementation, all groups before the
# innermost group are shifted when quantize_outlier=True.
# Thus, with three groups, outer and middle are shifted;
# inner is not.
# ================================================================

def shift_group(
    x,
    mask,
    group_idx,
    threshold_lowers,
    threshold_uppers,
):
    selected = x * mask

    # The innermost group is not shifted.
    if group_idx == len(threshold_lowers) - 1:
        return selected

    positive_mask = selected > 0
    negative_mask = selected < 0

    # Match the original code's half-precision threshold values.
    lower = torch.tensor(
        threshold_lowers[group_idx],
        device=x.device,
        dtype=torch.float16,
    ).float()

    upper = torch.tensor(
        threshold_uppers[group_idx],
        device=x.device,
        dtype=torch.float16,
    ).float()

    shifted = selected.clone()

    shifted = torch.where(
        positive_mask,
        selected - upper,
        shifted,
    )

    shifted = torch.where(
        negative_mask,
        selected - lower,
        shifted,
    )

    return shifted * mask


# ================================================================
# CODEBOOK HISTOGRAM COLLECTION
#
# Histogram values are absolute normalized magnitudes in [0, 6].
# Each histogram pools samples across all model layers.
# ================================================================

def make_histograms():
    return {
        name: torch.zeros(
            HIST_BINS,
            dtype=torch.float64,
        )
        for name in CODEBOOK_CONFIG
    }


def accumulate_codebook_histograms(
    histograms,
    tensor_kind,
    layer_idx,
    output,
    quantizer_stat,
):
    x = output.detach().float()

    lower_thresholds = (
        quantizer_stat[tensor_kind]["lower_threshold"][layer_idx]
    )

    upper_thresholds = (
        quantizer_stat[tensor_kind]["upper_threshold"][layer_idx]
    )

    # Each row represents one token's complete projected vector.
    original_shape = x.shape
    feature_dim = original_shape[-1]
    rows = x.reshape(-1, feature_dim)

    masks = get_group_masks(
        rows,
        lower_thresholds,
        upper_thresholds,
    )

    for group_idx, mask in enumerate(masks):
        if group_idx == 0:
            group_name = "outer"
        elif group_idx == len(masks) - 1:
            group_name = "inner"
        else:
            group_name = "middle"

        if tensor_kind == "key":
            class_name = "K_" + group_name
        else:
            class_name = "V_" + group_name

        shifted = shift_group(
            rows,
            mask,
            group_idx,
            lower_thresholds,
            upper_thresholds,
        )

        # One scale per token for this group's selected channels.
        scale, amax = tokenwise_e4m3_scale(shifted, mask)

        normalized = shifted / scale
        normalized = normalized.abs().clamp(
            0.0,
            MAX_NORMALIZED_VALUE,
        )

        valid = mask & (amax > 0)

        values = normalized[valid]

        if values.numel() == 0:
            continue

        batch_hist = torch.histc(
            values.float(),
            bins=HIST_BINS,
            min=0.0,
            max=MAX_NORMALIZED_VALUE,
        )

        histograms[class_name] += batch_hist.cpu().double()


# ================================================================
# FIT A ONE-DIMENSIONAL CODEBOOK USING WEIGHTED LLOYD ITERATIONS
#
# FP5: 16 magnitude levels, plus sign handled separately.
# FP4:  8 magnitude levels, plus sign handled separately.
#
# Zero and the maximum representable normalized magnitude are
# fixed endpoints. The remaining levels are fitted to the data.
# ================================================================

def fit_codebook(histogram, bits):
    number_of_levels = 2 ** (bits - 1)

    counts = histogram.double()

    if counts.sum().item() == 0:
        raise RuntimeError(
            "No samples collected for a codebook. "
            "Check the model hooks, runner, and group masks."
        )

    bin_width = MAX_NORMALIZED_VALUE / HIST_BINS

    bin_centers = (
        torch.arange(HIST_BINS, dtype=torch.float64) + 0.5
    ) * bin_width

    # Initialize levels over the normalized range.
    levels = torch.linspace(
        0.0,
        MAX_NORMALIZED_VALUE,
        number_of_levels,
        dtype=torch.float64,
    )

    for _ in range(LLOYD_ITERATIONS):
        edges = (levels[:-1] + levels[1:]) / 2.0

        assignments = torch.bucketize(
            bin_centers,
            edges,
        )

        weighted_counts = torch.bincount(
            assignments,
            weights=counts,
            minlength=number_of_levels,
        )

        weighted_sums = torch.bincount(
            assignments,
            weights=counts * bin_centers,
            minlength=number_of_levels,
        )

        new_levels = levels.clone()

        for idx in range(1, number_of_levels - 1):
            if weighted_counts[idx] > 0:
                new_levels[idx] = (
                    weighted_sums[idx] / weighted_counts[idx]
                )

        # Keep zero and the maximum level fixed.
        new_levels[0] = 0.0
        new_levels[-1] = MAX_NORMALIZED_VALUE

        # Preserve sorted levels.
        new_levels = torch.cummax(
            new_levels,
            dim=0,
        ).values

        new_levels[0] = 0.0
        new_levels[-1] = MAX_NORMALIZED_VALUE

        if torch.max(torch.abs(new_levels - levels)) < 1e-7:
            levels = new_levels
            break

        levels = new_levels

    return levels.float()


# ================================================================
# THRESHOLD PROFILING
# ================================================================

def profile_thresholds(args, model, tokenizer, torch_device, runner):
    n_layer = len(model.get_decoder().layers)
    n_groups = len(args.list_fracs)

    stat = {
        "key": [
            {
                "upper_threshold": [0.0] * n_groups,
                "lower_threshold": [0.0] * n_groups,
                "counter": 0,
            }
            for _ in range(n_layer)
        ],
        "value": [
            {
                "upper_threshold": [0.0] * n_groups,
                "lower_threshold": [0.0] * n_groups,
                "counter": 0,
            }
            for _ in range(n_layer)
        ],
    }

    def make_threshold_hook(layer_idx, tensor_kind):
        def hook(module, inputs, output):
            if args.sample_rate < 1.0:
                sample_count = max(
                    1,
                    int(args.sample_rate * output.size(1)),
                )

                sampled_indices = torch.randperm(
                    output.size(1),
                    device=output.device,
                )[:sample_count]

                values = output[:, sampled_indices, :].float()
            else:
                values = output.float()

            cumulative_fraction = 0.0

            for idx, fraction in enumerate(args.list_fracs):
                cumulative_fraction += fraction

                upper = torch.quantile(
                    values,
                    1.0 - cumulative_fraction / 2.0,
                    interpolation="higher",
                ).half().cpu().item()

                lower = torch.quantile(
                    values,
                    cumulative_fraction / 2.0,
                    interpolation="lower",
                ).half().cpu().item()

                stat[tensor_kind][layer_idx]["upper_threshold"][idx] += upper
                stat[tensor_kind][layer_idx]["lower_threshold"][idx] += lower

            stat[tensor_kind][layer_idx]["counter"] += 1

            return output

        return hook

    handles = []

    for idx, decoder in enumerate(model.get_decoder().layers):
        handles.append(
            decoder.self_attn.k_proj.register_forward_hook(
                make_threshold_hook(idx, "key")
            )
        )

        handles.append(
            decoder.self_attn.v_proj.register_forward_hook(
                make_threshold_hook(idx, "value")
            )
        )

    print("\n[1/2] Profiling K/V thresholds...")

    try:
        runner(args, model, tokenizer, torch_device)
    finally:
        for handle in handles:
            handle.remove()

    quantizer_stat = {
        "n_quant_group": n_groups,
        "key": {
            "upper_threshold": [
                [0.0] * n_groups for _ in range(n_layer)
            ],
            "lower_threshold": [
                [0.0] * n_groups for _ in range(n_layer)
            ],
        },
        "value": {
            "upper_threshold": [
                [0.0] * n_groups for _ in range(n_layer)
            ],
            "lower_threshold": [
                [0.0] * n_groups for _ in range(n_layer)
            ],
        },
    }

    for tensor_kind in ("key", "value"):
        for layer_idx in range(n_layer):
            count = stat[tensor_kind][layer_idx]["counter"]

            if count == 0:
                raise RuntimeError(
                    "No activations captured for "
                    + tensor_kind
                    + " layer "
                    + str(layer_idx)
                )

            for group_idx in range(n_groups):
                quantizer_stat[tensor_kind]["upper_threshold"][
                    layer_idx
                ][group_idx] = (
                    stat[tensor_kind][layer_idx]["upper_threshold"][
                        group_idx
                    ] / count
                )

                quantizer_stat[tensor_kind]["lower_threshold"][
                    layer_idx
                ][group_idx] = (
                    stat[tensor_kind][layer_idx]["lower_threshold"][
                        group_idx
                    ] / count
                )

    return quantizer_stat


# ================================================================
# CODEBOOK CALIBRATION PASS
# ================================================================

def profile_codebooks(
    args,
    model,
    tokenizer,
    torch_device,
    runner,
    quantizer_stat,
):
    histograms = make_histograms()
    handles = []

    def make_codebook_hook(layer_idx, tensor_kind):
        def hook(module, inputs, output):
            accumulate_codebook_histograms(
                histograms,
                tensor_kind,
                layer_idx,
                output,
                quantizer_stat,
            )
            return output

        return hook

    for idx, decoder in enumerate(model.get_decoder().layers):
        handles.append(
            decoder.self_attn.k_proj.register_forward_hook(
                make_codebook_hook(idx, "key")
            )
        )

        handles.append(
            decoder.self_attn.v_proj.register_forward_hook(
                make_codebook_hook(idx, "value")
            )
        )

    print("\n[2/2] Collecting normalized group distributions...")

    try:
        # The model is run again because the final thresholds
        # are needed before group masks can be applied.
        runner(args, model, tokenizer, torch_device)
    finally:
        for handle in handles:
            handle.remove()

    codebooks = {}
    histogram_counts = {}

    print("\nFitting six shared codebooks...")

    for class_name, config in CODEBOOK_CONFIG.items():
        total = histograms[class_name].sum().item()

        if total == 0:
            raise RuntimeError(
                "No calibration samples for " + class_name
            )

        codebooks[class_name] = fit_codebook(
            histograms[class_name],
            config["bits"],
        ).tolist()

        histogram_counts[class_name] = int(total)

        print(
            class_name,
            "| bits =", config["bits"],
            "| magnitude levels =", len(codebooks[class_name]),
            "| histogram samples =", histogram_counts[class_name],
        )

    return codebooks, histogram_counts


# ================================================================
# MAIN
# ================================================================

def multi_key_value_tokenwise_main(args, runner):
    if len(args.list_fracs) != 3:
        raise ValueError(
            "This configuration expects exactly three fractions "
            "for outer, middle, and inner groups."
        )

    if abs(sum(args.list_fracs) - 1.0) >= 1e-6:
        raise ValueError(
            "The fractions in --list_fracs must sum to 1."
        )

    if not 0.0 < args.sample_rate <= 1.0:
        raise ValueError("--sample-rate must be in (0, 1].")

    torch_device, tokenizer, model = common_main(args)

    start_time = time.monotonic()

    # Pass 1: determine Oaken's threshold boundaries.
# Pass 1: determine Oaken's threshold boundaries.
    quantizer_stat = profile_thresholds(
        args,
        model,
        tokenizer,
        torch_device,
        runner,
    )
    
    # Plot K/V distributions after group shifting and scale division.
    plot_shifted_normalized_distributions(
        model,
        tokenizer,
        args,
        torch_device,
        runner,
        quantizer_stat,
    )
    
    # Pass 2: collect normalized group values and fit codebooks.
    codebooks, histogram_counts = profile_codebooks(
        args,
        model,
        tokenizer,
        torch_device,
        runner,
    )

    # Keep the original threshold JSON structure and add codebooks.
    quantizer_stat["codebook_config"] = CODEBOOK_CONFIG
    quantizer_stat["codebooks"] = codebooks
    quantizer_stat["codebook_histogram_counts"] = histogram_counts

    quantizer_stat["codebook_metadata"] = {
        "codebook_sharing": "one codebook per class shared across layers",
        "scale_mode": "token-wise per group",
        "scale_format": "E4M3",
        "scale_formula": "amax / 6",
        "normalized_magnitude_range": [0.0, 6.0],
        "codebook_fitting": "weighted one-dimensional Lloyd iterations",
        "key_calibration_point": "k_proj output before RoPE",
        "value_calibration_point": "v_proj output",
        "note": (
            "This file calibrates thresholds and codebooks. "
            "It does not quantize the KV cache during evaluation."
        ),
    }

    with open(args.output_file, "w") as file:
        json.dump(quantizer_stat, file, indent=2)

    elapsed = time.monotonic() - start_time

    print("\nCalibration complete.")
    print("Saved:", args.output_file)
    print("Elapsed seconds:", round(elapsed, 2))


# ================================================================
# ARGUMENTS
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile KV thresholds and fit token-wise codebooks."
    )

    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        dest="task",
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        dest="model",
    )

    parser.add_argument(
        "-s", "--size",
        type=str,
        required=True,
        dest="model_size",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        dest="output_file",
    )

    parser.add_argument(
        "-f", "--list_fracs",
        nargs="+",
        type=float,
        required=True,
        dest="list_fracs",
    )

    parser.add_argument(
        "-b", "--batch-size",
        default=1,
        type=int,
        dest="batch_size",
    )

    parser.add_argument(
        "--sample-rate",
        default=1.0,
        type=float,
        dest="sample_rate",
    )

    parser.add_argument(
        "--gpu-start-idx",
        default=0,
        type=int,
        dest="gpu_start_idx",
    )

    parser.add_argument(
        "--gpu-count",
        default=1,
        type=int,
        dest="gpu_count",
    )

    args = parser.parse_args()

    if args.task == "wikitext":
        selected_runner = eval_perplexity
    else:
        selected_runner = run_model

    multi_key_value_tokenwise_main(
        args,
        selected_runner,
    )
