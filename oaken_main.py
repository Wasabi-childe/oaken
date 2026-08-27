from functools import partial
import json
from statistics import mean

import torch
from src.oaken.quantize import *
import pandas as pd
from src.oaken.huffman import HuffmanCollector

def multi_group_oaken_main(args, model, tokenizer, device, runner):
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)
        huffman_collector = HuffmanCollector()
        n_quant_group = quantizer_stat["n_quant_group"]
        n_layer = len(model.get_decoder().layers)

        sparsity_information = {
            "key": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
            "value": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
            "counter": [0.0 for j in range(n_layer)]
        }



        def tokenwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity, heatmap, codes_info = MultiThresholdTokenwiseQuantizer.downsample_with_codes(
                output,
                quantizer_stat["value"]["lower_threshold"][i],
                quantizer_stat["value"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
                kv_type="value",
                huffman_collector=huffman_collector,
            )
            sparsity_information["value"][i] = [sum(x) for x in zip(sparsity_information["value"][i], sparsity)]
            sparsity_information["counter"][i] += 0.5

            # ---------------- NEW: capture latest quantized codes (not dequantized fp16) for this layer ----------------

            # nonlocal value_counter
            # if value_counter < 10:
            #     df = pd.DataFrame(heatmap.squeeze().int().cpu().numpy())
            #     df.to_csv(f"heatmap/value_{i}_{value_counter}.csv", index=False)
            #     value_counter += 1

            return tensor.half()

        def channelwise_quantize_activation_hook(i, module, input, output):
            #tensor, sparsity, heatmap = MultiThresholdChannelwiseQuantizer.downsample( #
            tensor, sparsity, heatmap, codes_info = MultiThresholdTokenwiseQuantizer.downsample_with_codes( #
                output,
                quantizer_stat["key"]["lower_threshold"][i],
                quantizer_stat["key"]["upper_threshold"][i],
                args.quant_outlier,
                use_group_shift=True,
                kv_type="key",
                huffman_collector=huffman_collector,
            )
            sparsity_information["key"][i] = [sum(x) for x in zip(sparsity_information["key"][i], sparsity)]
            sparsity_information["counter"][i] += 0.5

            # ---------------- NEW: capture latest quantized codes (not dequantized fp16) for this layer ----------------

            # nonlocal key_counter
            # if key_counter < 10:
            #     df = pd.DataFrame(heatmap.squeeze().int().cpu().numpy())
            #     df.to_csv(f"heatmap/key_{i}_{key_counter}.csv", index=False)
            #     key_counter += 1

            return tensor.half()
    
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(partial(tokenwise_quantize_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(channelwise_quantize_activation_hook, i))
        
        runner(args, model, tokenizer, device)
        print("\nCalibration complete.")

        huffman_collector.print_summary()
        
        codebooks = huffman_collector.build_all_codebooks()
        
        huffman_save_path = "/content/oaken/huffman_codebooks.json"
        
        with open(huffman_save_path, "w") as f:
            json.dump(codebooks, f, indent=2)
        
        print(f"Saved Huffman codebooks to {huffman_save_path}")
        
        print("\n" + "=" * 70)
        print("HUFFMAN CODEBOOKS")
        print("=" * 70)
        
        for name, codebook in codebooks.items():
            print(f"\n{name}:")
            for symbol, code in sorted(codebook.items()):
                print(f"  {symbol:2d} -> {code}")

        print()
        print("=" * 70)
        print("HUFFMAN RESULT")
        print("=" * 70)

        # ------------------------------------------------------------
        # Calculate Huffman storage size
        # ------------------------------------------------------------

        total_values = 0
        total_fixed_bits = 0
        total_huffman_bits = 0

        for kv_type in huffman_collector.counts:
            for group_name in huffman_collector.counts[kv_type]:

                counts = huffman_collector.counts[kv_type][group_name]

                # Number of symbols in this group
                total_codes = sum(counts.values())

                # Corresponding Huffman codebook
                codebook_name = f"{kv_type}_{group_name}"
                codebook = codebooks[codebook_name]

                # Fixed-length representation
                #
                # inner       -> 5 bits
                # outer_0     -> 5 bits
                # outer_1     -> 4 bits
                #
                # We can determine this directly from the number of
                # unique possible symbols.
                if group_name == "outer_1":
                    fixed_bits_per_code = 4
                else:
                    fixed_bits_per_code = 5

                fixed_bits = total_codes * fixed_bits_per_code

                # Huffman bits = frequency × Huffman code length
                huffman_bits = sum(
                    count * len(codebook[symbol])
                    for symbol, count in counts.items()
                )

                avg_huffman_bits = huffman_bits / total_codes

                savings = (
                    1 - huffman_bits / fixed_bits
                ) * 100

                print()
                print(f"{kv_type.upper()} / {group_name}")
                print("-" * 50)
                print(f"Values:                 {total_codes:,}")
                print(f"Fixed bits/code:        {fixed_bits_per_code}")
                print(f"Huffman bits/code:      {avg_huffman_bits:.4f}")
                print(f"Fixed size:             {fixed_bits / 8 / 1024 / 1024:.4f} MB")
                print(f"Huffman size:           {huffman_bits / 8 / 1024 / 1024:.4f} MB")
                print(f"Savings:                {savings:.2f}%")

                total_values += total_codes
                total_fixed_bits += fixed_bits
                total_huffman_bits += huffman_bits

        # ------------------------------------------------------------
        # Overall result
        # ------------------------------------------------------------

        overall_avg = total_huffman_bits / total_values

        overall_savings = (
            1 - total_huffman_bits / total_fixed_bits
        ) * 100

        print()
        print("=" * 70)
        print("OVERALL")
        print("=" * 70)

        print(f"Total values:           {total_values:,}")

        print(
            f"Fixed quantized size:   "
            f"{total_fixed_bits / 8 / 1024 / 1024:.4f} MB"
        )

        print(
            f"Huffman size:           "
            f"{total_huffman_bits / 8 / 1024 / 1024:.4f} MB"
        )

        print(
            f"Average Huffman rate:   "
            f"{overall_avg:.4f} bits/code"
        )

        print(
            f"Overall savings:        "
            f"{overall_savings:.2f}%"
        )

        # ---------------- NEW: save captured quantized K/V codes to a .pt file ----------------


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

    # ---------------- NEW: storage for captured quantized K/V ----------------


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

            # ---------------- NEW ----------------
            # NOTE: TokenwiseQuantizer.downsample() here does not return integer
            # codes (only MultiThresholdTokenwiseQuantizer.downsample_with_codes does).
            # Capturing the dequantized fp16 tensor as before; if you need true
            # integer codes for TokenwiseQuantizer/ChannelwiseQuantizer as well,
            # they'd need the same kind of *_with_codes addition made to them.

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

            # ---------------- NEW ----------------


            return tensor.half()
    
    if args.model in ["opt", "llama"]:
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(partial(tokenwise_quantize_activation_hook, i))
            decoder.self_attn.k_proj.register_forward_hook(partial(channelwise_quantize_activation_hook, i))
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    runner(args, model, tokenizer, device)


    # ---------------- NEW: save captured K/V to a .pt file ----------------

    key_sparsity = []
    value_sparsity = []
    for i in range(len(model.get_decoder().layers)):
        key_sparsity.append(sparsity_information['key'][i] / sparsity_information['counter'][i])
        value_sparsity.append(sparsity_information['value'][i] / sparsity_information['counter'][i])
        print(f"Decoder {i} Sparsity: Key - {key_sparsity[i]}, Value - {value_sparsity[i]}")

    print(f"Total Sparsity: Key - {mean(key_sparsity)}, Value - {mean(value_sparsity)}")
