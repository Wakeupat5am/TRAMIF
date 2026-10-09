import numpy as np

from scripts.entropy_core import shannon_entropy


class _BufferedInput:
    """Read a declared number of bytes with a bounded input buffer."""

    def __init__(self, stream, file_size, chunk_size):
        self.stream = stream
        self.remaining = file_size
        self.chunk_size = chunk_size
        self.buffer = b""
        self.position = 0

    def read_exact(self, count):
        output = bytearray()

        while len(output) < count:
            if self.position == len(self.buffer):
                requested = min(self.chunk_size, self.remaining)

                if requested == 0:
                    raise ValueError(
                        "Not enough bytes for the requested window."
                    )

                self.buffer = self.stream.read(requested)
                self.position = 0

                if not self.buffer:
                    raise ValueError(
                        "Input ended before the declared file size."
                    )

                if len(self.buffer) > requested:
                    raise ValueError(
                        "Stream returned more bytes than requested."
                    )

                self.remaining -= len(self.buffer)

            available = len(self.buffer) - self.position
            take = min(count - len(output), available)
            stop = self.position + take

            output.extend(self.buffer[self.position:stop])
            self.position = stop

        return bytes(output)

    def finish(self):
        if self.remaining != 0 or self.position != len(self.buffer):
            raise RuntimeError("Some declared input bytes were not consumed.")

        if self.stream.read(1):
            raise ValueError(
                "Input contains more than the declared file size."
            )


def iter_local_entropy(
    stream,
    file_size,
    window_size=256,
    stride=128,
    chunk_size=65536,
):
    """Yield finalized local entropy values in their original byte order.

    Each yielded block is a one-dimensional float64 NumPy array.
    Across all blocks, there is exactly one value per input byte.

    Values are normalized by 8 and averaged across covering windows.
    The first window reaching EOF is the last window.

    Consume the iterator completely to perform the final size check.
    Memory usage is bounded by chunk_size and window_size.
    """
    for name, value in (
        ("file_size", file_size),
        ("window_size", window_size),
        ("chunk_size", chunk_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if type(stride) is not int or not 1 <= stride <= window_size:
        raise ValueError("stride must be between 1 and window_size.")

    reader = _BufferedInput(stream, file_size, chunk_size)
    window = reader.read_exact(min(window_size, file_size))

    entropy_sums = np.zeros(window_size, dtype=np.float64)
    coverage_counts = np.zeros(window_size, dtype=np.int64)
    start = 0

    while True:
        length = len(window)

        # Reuse the already-tested entropy definition.
        normalized_entropy = shannon_entropy(window) / 8.0
        entropy_sums[:length] += normalized_entropy
        coverage_counts[:length] += 1

        reaches_end = start + length == file_size

        # The next window starts at start + stride.
        # Positions before that point cannot receive another contribution.
        finalized = length if reaches_end else stride

        if (coverage_counts[:finalized] == 0).any():
            raise RuntimeError("An output position has no covering window.")

        # Division creates a separate array, safe to retain after yielding.
        values = (
            entropy_sums[:finalized]
            / coverage_counts[:finalized]
        )
        yield values

        if reaches_end:
            reader.finish()
            return

        # Retain only positions that overlap the next window.
        carry = length - stride

        entropy_sums[:carry] = entropy_sums[stride:length].copy()
        coverage_counts[:carry] = coverage_counts[stride:length].copy()

        entropy_sums[carry:] = 0.0
        coverage_counts[carry:] = 0

        start += stride
        new_count = min(stride, file_size - (start + carry))

        window = window[stride:] + reader.read_exact(new_count)