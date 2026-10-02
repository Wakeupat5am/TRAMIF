from collections import Counter
from math import log2


def shannon_entropy(data: bytes) -> float:
    """Return byte-distribution entropy in bits, ranging from 0 to 8."""

    if not data:
        raise ValueError("Input data must not be empty.")

    counts = Counter(data)
    total = len(data)
    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)

    return entropy


def local_entropy_values(
    data: bytes,
    window_size: int = 256,
    stride: int = 128,
) -> list[float]:
    """Return normalized local entropy for each real byte offset.

    Average the normalized entropy of all windows covering an offset.
    Use only existing bytes in short or trailing windows.
    Stop after the first window that reaches the end of the input.
    """

    if not data:
        raise ValueError("Input data must not be empty.")

    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if type(stride) is not int or not 1 <= stride <= window_size:
        raise ValueError("stride must be between 1 and window_size.")

    entropy_sums = [0.0] * len(data)
    coverage_counts = [0] * len(data)

    for start in range(0, len(data), stride):
        stop = min(start + window_size, len(data))
        window = data[start:stop]
        normalized_entropy = shannon_entropy(window) / 8.0

        for offset in range(start, stop):
            entropy_sums[offset] += normalized_entropy
            coverage_counts[offset] += 1

        if stop == len(data):
            break

    if any(count == 0 for count in coverage_counts):
        raise RuntimeError("Some byte offsets were not covered.")

    return [
        total / count
        for total, count in zip(entropy_sums, coverage_counts)
    ]