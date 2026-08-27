import heapq
from collections import Counter


class HuffmanCollector:

    def __init__(self):
        # Six Oaken groups:
        #
        # key/inner
        # key/outer_0
        # key/outer_1
        # value/inner
        # value/outer_0
        # value/outer_1

        self.counts = {
            "key": {
                "inner": Counter(),
                "outer_0": Counter(),
                "outer_1": Counter(),
            },
            "value": {
                "inner": Counter(),
                "outer_0": Counter(),
                "outer_1": Counter(),
            }
        }

    def update(self, kv_type, group_name, codes, mask=None):

        if mask is not None:
            codes = codes[mask]

        codes = codes.reshape(-1).cpu().tolist()

        self.counts[kv_type][group_name].update(codes)

    def get_counts(self, kv_type, group_name):
        return self.counts[kv_type][group_name]

    def build_codebook(self, kv_type, group_name):

        counts = self.counts[kv_type][group_name]

        if len(counts) == 0:
            return {}

        # Only one symbol
        if len(counts) == 1:

            symbol = next(iter(counts))

            return {
                symbol: "0"
            }

        heap = []

        counter = 0

        # Create initial leaf nodes
        for symbol, frequency in counts.items():

            heapq.heappush(
                heap,
                (frequency, counter, symbol)
            )

            counter += 1

        next_node = counter

        # Internal tree nodes
        trees = {}

        # Build Huffman tree
        while len(heap) > 1:

            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)

            merged = next_node
            next_node += 1

            trees[merged] = (node1, node2)

            heapq.heappush(
                heap,
                (
                    freq1 + freq2,
                    counter,
                    merged
                )
            )

            counter += 1

        root = heap[0][2]

        codebook = {}

        def traverse(node, code):

            # Leaf node
            if node in counts:

                codebook[node] = code
                return

            # Internal node
            left, right = trees[node]

            traverse(left, code + "0")
            traverse(right, code + "1")

        traverse(root, "")

        return codebook

    def build_all_codebooks(self):

        """
        Build a separate Huffman codebook for every
        Oaken quantization group.

        Total = 6 codebooks.
        """

        codebooks = {}

        all_groups = [
            ("key", "inner"),
            ("key", "outer_0"),
            ("key", "outer_1"),
            ("value", "inner"),
            ("value", "outer_0"),
            ("value", "outer_1"),
        ]

        for kv_type, group_name in all_groups:

            key = f"{kv_type}_{group_name}"

            codebooks[key] = self.build_codebook(
                kv_type,
                group_name
            )

        return codebooks

    def print_summary(self):

        print()
        print("=" * 70)
        print("HUFFMAN FREQUENCY SUMMARY")
        print("=" * 70)

        for kv_type in self.counts:

            for group_name in self.counts[kv_type]:

                counts = self.counts[kv_type][group_name]

                total = sum(counts.values())

                print()
                print(
                    f"{kv_type.upper()} / {group_name}"
                )

                print(
                    f"Total codes: {total:,}"
                )

                print(
                    f"Unique codes: {len(counts)}"
                )

                if total > 0:

                    for symbol, count in counts.most_common():

                        percentage = \
                            100.0 * count / total

                        print(
                            f"  Code {symbol:2d}: "
                            f"{count:10,} "
                            f"({percentage:7.3f}%)"
                        )

    def print_codebooks(self, codebooks):

        print()
        print("=" * 70)
        print("HUFFMAN CODEBOOKS")
        print("=" * 70)

        for name, codebook in codebooks.items():

            print()
            print(f"{name}:")

            for symbol in sorted(codebook):

                print(
                    f"  {symbol:2d} -> {codebook[symbol]}"
                )

        print()
        print(f"Total Huffman codebooks: {len(codebooks)}")
