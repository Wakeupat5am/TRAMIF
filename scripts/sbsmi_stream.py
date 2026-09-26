from pathlib import Path


def file_to_sbsmi(
    path: str | Path,
    l: int = 6,
    chunk_size: int = 65536,
) -> list[list[int]]:
    """Build an SBSMI by reading a file in chunks."""

    if not isinstance(l, int) or not 1 <= l <= 8:
        raise ValueError("l must be an integer between 1 and 8.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    size = 2 ** l
    mask = size - 1
    counts = [[0 for _ in range(size)] for _ in range(size)]

    buffer = 0
    bits_available = 0
    previous = None
    state_count = 0

    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            for byte in chunk:
                buffer = (buffer << 8) | byte
                bits_available += 8

                while bits_available >= l:
                    bits_available -= l
                    current = (buffer >> bits_available) & mask

                    if previous is not None:
                        counts[previous][current] += 1

                    previous = current
                    state_count += 1

                # Retain unused bits, including across chunk boundaries.
                buffer &= (1 << bits_available) - 1

    # Handle the partial block only after reaching the end of the file.
    if bits_available:
        current = buffer

        if previous is not None:
            counts[previous][current] += 1

        state_count += 1

    if state_count < 2:
        raise ValueError("At least two states are needed for a transition.")

    pixels = []

    for row in counts:
        row_total = sum(row)

        if row_total == 0:
            pixels.append([0] * size)
        else:
            pixels.append([
                count * 255 // row_total
                for count in row
            ])

    return pixels