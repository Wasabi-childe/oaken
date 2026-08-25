import heapq
from collections import Counter


class HuffmanNode:
    def __init__(self, frequency, symbol=None, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency


class HuffmanCodec:

    # ============================================================
    # BUILD HUFFMAN TREE
    # ============================================================

    @staticmethod
    def build_tree(symbols):
        """
        Build a Huffman tree from a list of integer symbols.
        """

        if len(symbols) == 0:
            return None

        frequencies = Counter(symbols)

        heap = []

        for symbol, frequency in frequencies.items():
            node = HuffmanNode(
                frequency=frequency,
                symbol=symbol
            )
            heapq.heappush(heap, node)

        # Only one unique symbol
        if len(heap) == 1:
            return heap[0]

        while len(heap) > 1:

            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )

            heapq.heappush(heap, merged)

        return heap[0]

    # ============================================================
    # GENERATE HUFFMAN CODES
    # ============================================================

    @staticmethod
    def build_codes(root):
        """
        Generate:
            symbol -> binary Huffman code
        """

        codes = {}

        if root is None:
            return codes

        # Single-symbol case
        if root.symbol is not None:
            codes[root.symbol] = "0"
            return codes

        def traverse(node, code):

            if node is None:
                return

            if node.symbol is not None:
                codes[node.symbol] = code
                return

            traverse(
                node.left,
                code + "0"
            )

            traverse(
                node.right,
                code + "1"
            )

        traverse(root, "")

        return codes

    # ============================================================
    # ENCODE
    # ============================================================

    @classmethod
    def encode(cls, symbols):
        """
        Huffman encode a list of integer symbols.

        Returns:
            encoded_data
            codes
            original_symbol_count
        """

        if len(symbols) == 0:
            return b"", {}, 0

        root = cls.build_tree(symbols)

        codes = cls.build_codes(root)

        # --------------------------------------------------------
        # Convert Huffman bits into actual bytes
        # --------------------------------------------------------

        output = bytearray()

        current_byte = 0
        bit_count = 0
        total_bits = 0

        for symbol in symbols:

            code = codes[symbol]

            for bit in code:

                current_byte <<= 1

                if bit == "1":
                    current_byte |= 1

                bit_count += 1
                total_bits += 1

                if bit_count == 8:

                    output.append(current_byte)

                    current_byte = 0
                    bit_count = 0

        # --------------------------------------------------------
        # Handle final incomplete byte
        # --------------------------------------------------------

        padding_bits = 0

        if bit_count > 0:

            padding_bits = 8 - bit_count

            current_byte <<= padding_bits

            output.append(current_byte)

        return (
            bytes(output),
            codes,
            total_bits
        )

    # ============================================================
    # DECODE
    # ============================================================

    @staticmethod
    def decode(
        encoded_data,
        codes,
        total_bits,
        num_symbols
    ):
        """
        Decode Huffman-compressed bytes.

        Args:
            encoded_data:
                Compressed byte stream.

            codes:
                Dictionary:
                    symbol -> Huffman code

            total_bits:
                Number of valid bits in encoded_data.

            num_symbols:
                Number of original symbols.

        Returns:
            List of decoded integer symbols.
        """

        if num_symbols == 0:
            return []

        if not codes:
            return []

        # Reverse dictionary:
        # code -> symbol

        reverse_codes = {
            code: symbol
            for symbol, code in codes.items()
        }

        decoded = []

        current_code = ""

        bits_read = 0

        for byte in encoded_data:

            for bit_position in range(7, -1, -1):

                if bits_read >= total_bits:
                    break

                bit = (
                    byte >> bit_position
                ) & 1

                current_code += str(bit)

                bits_read += 1

                if current_code in reverse_codes:

                    decoded.append(
                        reverse_codes[current_code]
                    )

                    current_code = ""

                    if len(decoded) == num_symbols:
                        return decoded

        return decoded

    # ============================================================
    # COMPRESSION STATISTICS
    # ============================================================

    @staticmethod
    def compression_stats(
        original_bits,
        compressed_bits
    ):
        """
        Calculate compression ratio and memory reduction.
        """

        if original_bits <= 0:
            return {
                "compression_ratio": 0.0,
                "memory_reduction": 0.0
            }

        if compressed_bits <= 0:
            return {
                "compression_ratio": 0.0,
                "memory_reduction": 0.0
            }

        compression_ratio = (
            original_bits / compressed_bits
        )

        memory_reduction = (
            1.0
            - compressed_bits / original_bits
        ) * 100.0

        return {
            "compression_ratio": compression_ratio,
            "memory_reduction": memory_reduction
        }

    # ============================================================
    # COMPLETE COMPRESSION FUNCTION
    # ============================================================

    @classmethod
    def compress(
        cls,
        symbols,
        bits_per_symbol
    ):
        """
        Complete Huffman compression.

        Args:
            symbols:
                List of integer quantization indices.

            bits_per_symbol:
                Number of bits used by Oaken quantization
                (8 or 9 in your implementation).

        Returns:
            dictionary containing compressed data,
            Huffman codes and compression statistics.
        """

        encoded_data, codes, total_bits = cls.encode(
            symbols
        )

        original_bits = (
            len(symbols)
            * bits_per_symbol
        )

        compressed_bits = total_bits

        stats = cls.compression_stats(
            original_bits,
            compressed_bits
        )

        return {
            "data": encoded_data,
            "codes": codes,
            "num_symbols": len(symbols),
            "total_bits": total_bits,
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": stats[
                "compression_ratio"
            ],
            "memory_reduction": stats[
                "memory_reduction"
            ]
        }

    # ============================================================
    # COMPLETE DECOMPRESSION FUNCTION
    # ============================================================

    @classmethod
    def decompress(
        cls,
        compressed_data,
        codes,
        total_bits,
        num_symbols
    ):
        """
        Complete Huffman decompression.
        """

        return cls.decode(
            encoded_data=compressed_data,
            codes=codes,
            total_bits=total_bits,
            num_symbols=num_symbols
        )


# ================================================================
# SIMPLE TEST
# ================================================================

if __name__ == "__main__":

    test_symbols = [
        3, 3, 3, 3,
        2, 2, 2,
        1, 1,
        0
    ]

    print("Original symbols:")
    print(test_symbols)

    # ------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------

    result = HuffmanCodec.compress(
        symbols=test_symbols,
        bits_per_symbol=8
    )

    print("\nHuffman codes:")
    print(result["codes"])

    print("\nOriginal bits:")
    print(result["original_bits"])

    print("\nCompressed bits:")
    print(result["compressed_bits"])

    print("\nCompressed bytes:")
    print(len(result["data"]))

    print("\nCompression ratio:")
    print(
        f"{result['compression_ratio']:.3f}x"
    )

    print("\nMemory reduction:")
    print(
        f"{result['memory_reduction']:.2f}%"
    )

    # ------------------------------------------------------------
    # Decompress
    # ------------------------------------------------------------

    decoded = HuffmanCodec.decompress(
        compressed_data=result["data"],
        codes=result["codes"],
        total_bits=result["total_bits"],
        num_symbols=result["num_symbols"]
    )

    print("\nDecoded symbols:")
    print(decoded)

    print("\nCorrect:")
    print(
        decoded == test_symbols
    )
